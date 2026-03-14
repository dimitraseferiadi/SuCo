/**
 * IndexSuCo.h
 *
 * FAISS integration of the SuCo (Subspace Collision) index for high-dimensional
 * Approximate Nearest Neighbor (ANN) search.
 *
 * Based on the paper:
 *   "Subspace Collision: An Efficient and Accurate Framework for
 *    High-dimensional Approximate Nearest Neighbor Search"
 *   Jiuqi Wei et al., SIGMOD 2025.
 *   https://doi.org/10.1145/3709729
 *
 * Algorithm overview:
 *   1. The d-dimensional space is divided into `nsubspaces` subspaces of
 *      dimension `subspace_dim = d / nsubspaces`.
 *   2. Each subspace is further split into two halves of dimension
 *      `half_dim = subspace_dim / 2`.
 *   3. K-means clustering (with `ncentroids_half` centroids) is run
 *      independently on each half, yielding an Inverted Multi-Index (IMI)
 *      with `ncentroids_half^2` cells per subspace.
 *   4. At query time the Dynamic Activation algorithm (Algorithm 3 in the
 *      paper) traverses IMI cells in ascending distance order until
 *      `collision_ratio * ntotal` points have been collected as "collisions"
 *      for a subspace.  Points that collide in the most subspaces form the
 *      candidate set (SC-score selection, controlled by `candidate_ratio`).
 *   5. The top-k results are returned from the candidate set via exact L2
 *      re-ranking.
 *
 * Key parameters:
 *   nsubspaces       Number of subspaces Ns (default 8).
 *                    d must be divisible by nsubspaces.
 *   ncentroids_half  sqrt(K), the number of K-means centroids per half-
 *                    subspace (default 50).  Total IMI cells per subspace = K
 *                    = ncentroids_half^2.
 *   collision_ratio  alpha: fraction of dataset retrieved per subspace as
 *                    collisions (default 0.05).
 *   candidate_ratio  beta: fraction of dataset used as the re-rank candidate
 *                    pool (default 0.005).
 *   niter            Number of K-means iterations (default 10).
 *   verbose          Print progress during train/search.
 *
 * Persistence:
 *   write_index(IOWriter*) / read_index(IOReader*) are the canonical
 *   serialisation methods and work with any FAISS IO backend (file, memory,
 *   etc.).  File-path convenience wrappers are also provided.
 *
 *   NOTE: This index is wired into FAISS global I/O dispatch, so both
 *   member methods and faiss::write_index() / faiss::read_index() work.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/io.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

// ---------------------------------------------------------------------------
// SearchParametersSuCo
// ---------------------------------------------------------------------------

/**
 * Per-search parameter overrides for IndexSuCo.
 *
 * Pass to IndexSuCo::search() to vary collision_ratio or candidate_ratio for
 * a single search call without mutating the index.  This is the idiomatic
 * FAISS approach and is thread-safe: multiple threads can query the same index
 * with different parameters simultaneously.
 *
 * A value <= 0 means "use the index member variable as the default".
 */
struct SearchParametersSuCo : SearchParameters {
    /// Override collision_ratio (alpha) for this search; <= 0 → index default.
    float collision_ratio = -1.0f;
    /// Override candidate_ratio (beta) for this search; <= 0 → index default.
    float candidate_ratio = -1.0f;

    explicit SearchParametersSuCo(
            float collision_ratio = -1.0f,
            float candidate_ratio = -1.0f)
            : collision_ratio(collision_ratio),
              candidate_ratio(candidate_ratio) {}
    ~SearchParametersSuCo() override = default;
};

// ---------------------------------------------------------------------------
// IndexSuCo
// ---------------------------------------------------------------------------
struct IndexSuCo : Index {
    // -----------------------------------------------------------------------
    // Hyper-parameters (set before train/add/search)
    // -----------------------------------------------------------------------

    /// Number of subspaces  Ns   (d must be divisible by nsubspaces)
    int nsubspaces;

    /// Number of K-means centroids per *half*-subspace  sqrt(K)
    int ncentroids_half;

    /// Fraction of the dataset to retrieve per subspace as collisions  (alpha)
    float collision_ratio;

    /// Fraction of the dataset to use as the re-rank candidate pool  (beta)
    float candidate_ratio;

    /// Number of K-means training iterations
    int niter;

    // -----------------------------------------------------------------------
    // Derived dimensions (set during train)
    // -----------------------------------------------------------------------

    /// d / nsubspaces  (must be even)
    int subspace_dim;

    /// subspace_dim / 2
    int half_dim;

    // -----------------------------------------------------------------------
    // Index storage
    // -----------------------------------------------------------------------

    /**
     * Centroids array, shape [nsubspaces * 2 * ncentroids_half * half_dim].
     * Layout: centroids of subspace i, first  half → offset
     *         i * 2 * ncentroids_half * half_dim
     *         centroids of subspace i, second half → offset
     *         (i * 2 + 1) * ncentroids_half * half_dim
     * Each centroid is half_dim floats stored row-major.
     */
    std::vector<float> centroids;

    /**
     * Flat Inverted Multi-Indexes, one per subspace.
     * inv_lists[s] is a flat array of ncentroids_half^2 inverted lists.
     * The list for cluster (c1, c2) in subspace s is at index
     *   c1 * ncentroids_half + c2.
     */
    std::vector<std::vector<std::vector<idx_t>>> inv_lists;

    /**
     * Raw added vectors, stored as a flat row-major array of size ntotal * d.
     * Kept in memory to support exact L2 re-ranking at query time.
     */
    std::vector<float> xb;

    // -----------------------------------------------------------------------
    // Constructor / destructor
    // -----------------------------------------------------------------------

    /**
     * @param d                 Vector dimensionality.
     * @param nsubspaces        Number of subspaces (default 8).
     * @param ncentroids_half   K-means centroids per half-subspace (default 50).
     * @param collision_ratio   alpha (default 0.05).
     * @param candidate_ratio   beta  (default 0.005).
     * @param niter             K-means iterations (default 10).
     */
    explicit IndexSuCo(
            idx_t d,
            int   nsubspaces      = 8,
            int   ncentroids_half = 50,
            float collision_ratio = 0.05f,
            float candidate_ratio = 0.005f,
            int   niter           = 10);

    ~IndexSuCo() override = default;

    // -----------------------------------------------------------------------
    // Core Index interface
    // -----------------------------------------------------------------------

    /**
     * Train the index: run K-means on each half of each subspace.
     *
     * Re-training is explicitly supported.  When called on an already-trained
     * (or even indexed) instance, all existing centroids, IMI buckets, and
     * stored vectors are discarded before the new training run begins.  This
     * matches the behaviour of faiss::IndexIVF and other re-trainable FAISS
     * indices.
     *
     * @param n   Number of training vectors.
     * @param x   Training vectors, shape [n, d], row-major float32.
     */
    void train(idx_t n, const float* x) override;

    /**
     * Add vectors to the index (after train).
     * Assigns each vector to its nearest centroid in every half-subspace and
     * inserts the vector ID into the corresponding IMI bucket.
     * Also stores the raw vectors for re-ranking.
     * @param n   Number of vectors to add.
     * @param x   Vectors, shape [n, d], row-major float32.
     */
    void add(idx_t n, const float* x) override;

    /**
     * Approximate k-NN search using the Subspace Collision framework.
     *
     * Batch queries are distributed across available OpenMP threads (one
     * thread per query).  search_one() itself is single-threaded, so there
     * are no nested parallel regions and no contention between threads.
     *
     * @param n        Number of query vectors.
     * @param x        Query vectors, shape [n, d], row-major float32.
     * @param k        Number of nearest neighbors to return.
     * @param distances Output distances,  shape [n, k].
     * @param labels    Output global IDs,  shape [n, k].
     * @param params    Optional SearchParametersSuCo for per-call ratio
     *                  overrides; nullptr → use index defaults.
     */
    void search(
            idx_t                   n,
            const float*            x,
            idx_t                   k,
            float*                  distances,
            idx_t*                  labels,
            const SearchParameters* params = nullptr) const override;

    /**
     * Reset the index to an untrained, empty state.
     */
    void reset() override;

    // -----------------------------------------------------------------------
    // Extra search modes
    // -----------------------------------------------------------------------

    /**
     * Index-free (SC-Linear) search: Algorithm 1 from the paper.
     *
     * For each subspace, computes exact distances to all database points to
     * determine collisions.  Has the same asymptotic cost as a linear scan
     * (O(n·d·Ns) per query) but delivers very high recall.  Intended as a
     * correctness / accuracy baseline against which the indexed SuCo can be
     * compared (cf. Table 2 and Table 4 in the paper).
     *
     * Like search(), batch queries are parallelised across OpenMP threads.
     *
     * @param cr_override  collision_ratio override; <= 0 → use index default.
     * @param cdr_override candidate_ratio override; <= 0 → use index default.
     */
    void search_linear(
            idx_t        n,
            const float* x,
            idx_t        k,
            float*       distances,
            idx_t*       labels,
            float        cr_override  = -1.0f,
            float        cdr_override = -1.0f) const;

    /**
     * Compute SC-scores for a single query without final reranking.
     * Runs Dynamic Activation across all subspaces and returns the raw
     * per-point SC-score (number of subspaces where the point appeared in
     * the activated cells).  Useful for analysing the Pareto property.
     *
     * @param xq         Query vector, length d.
     * @param cr         collision_ratio override; <= 0 → use index default.
     * @param out_scores Output buffer of length ntotal (int32_t).
     *                   out_scores[i] = SC-score of database point i.
     */
    void get_sc_scores(
            const float* xq,
            float        cr,
            int32_t*     out_scores) const;

    /**
     * Like search() but uses the classical Multi-sequence IMI traversal
     * (heap with visited-pair set) instead of Dynamic Activation.
     * The two algorithms produce identical results but with different
     * computational profiles; useful for Figure 6 efficiency comparison.
     *
     * Batch queries are parallelised across OpenMP threads identically to
     * search().
     *
     * @param cr_override  collision_ratio override; <= 0 → use index default.
     * @param cdr_override candidate_ratio override; <= 0 → use index default.
     */
    void search_multisequence(
            idx_t        n,
            const float* x,
            idx_t        k,
            float*       distances,
            idx_t*       labels,
            float        cr_override  = -1.0f,
            float        cdr_override = -1.0f) const;

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    /**
     * Serialise the index to an IOWriter (any FAISS IO backend: file, memory,
     * etc.).  This is the canonical persistence entry point.
     *
     * The binary format is tagged with a 'SuCo' magic word and version number
     * so that read_index() can validate the stream.
     *
        * This method interoperates with FAISS global I/O dispatch:
        * faiss::write_index() / faiss::read_index() also recognise IndexSuCo.
     */
    void write_index(IOWriter* f) const;

    /**
     * Deserialise an index from an IOReader.  Any state currently held by
     * this object is overwritten.
     */
    void read_index(IOReader* f);

    /** Convenience wrapper: write to a file at the given path. */
    void write_index(const char* fname) const;

    /** Convenience wrapper: read from a file at the given path. */
    void read_index(const char* fname);

private:
    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /**
     * Assign `n` vectors of dimension `dim` to their nearest centroid among
     * `ncentroids` centroids stored row-major in `cents`.
     * Writes integer assignments (0-based) into `out_assign`.
     */
    static void assign_to_centroids(
            idx_t        n,
            int          dim,
            const float* vecs,
            int          ncentroids,
            const float* cents,
            int32_t*     out_assign);

    /**
     * Dynamic Activation algorithm (Algorithm 3 in the paper).
     * Retrieves IMI cells in ascending (dist1[c1] + dist2[c2]) order from
     * inv_lists[subspace_idx] until at least `collision_num` data points have
     * been collected, and directly increments sc_scores for each point.
     *
     * Fully single-threaded; called from search_one() which is itself invoked
     * inside an outer OpenMP parallel-for loop in search().
     */
    void dynamic_activate(
            int            subspace_idx,
            const float*   dists1,
            const int32_t* idx1,
            const float*   dists2,
            const int32_t* idx2,
            idx_t          collision_num,
            uint8_t*       sc_scores) const;

    /**
     * Multi-sequence IMI traversal (Babenko & Lempitsky 2012).
     * Uses a min-heap of (combined_dist, i, j) with a visited bitset,
     * yielding cells in globally optimal (d1+d2)-ascending order.
     * Produces identical SC-score increments to dynamic_activate but with
     * strictly more heap operations for the same retrieved count.
     */
    void dynamic_activate_multisequence(
            int            subspace_idx,
            const float*   dists1,
            const int32_t* idx1,
            const float*   dists2,
            const int32_t* idx2,
            idx_t          collision_num,
            uint8_t*       sc_scores) const;

    /**
     * Shared SC-score selection, candidate gathering, and exact L2 re-ranking
     * used by both search_one() and search_one_multisequence() after collision
     * counting is complete.
     *
     * Selects *exactly* min(candidate_num, ntotal) candidates: all points
     * whose SC-score exceeds the threshold are included unconditionally, and
     * points at the threshold score fill the remainder of the budget in index
     * order (no overshoot).  The top-k nearest candidates are then returned.
     *
     * @param xq            Query vector, length d.
     * @param k             Number of NNs to return.
     * @param candidate_num Desired candidate pool size (beta * ntotal).
     * @param sc_scores     Per-point SC-scores, length ntotal (uint8_t).
     * @param out_dist      Output distances, length k.
     * @param out_labels    Output global IDs, length k.
     */
    void rerank(
            const float*   xq,
            idx_t          k,
            idx_t          candidate_num,
            const uint8_t* sc_scores,
            float*         out_dist,
            idx_t*         out_labels) const;

    /**
     * Single-query search using Dynamic Activation.
     *
     * Fully single-threaded.  Batch-level parallelism (one query per thread)
     * is handled by the calling search() method.  The caller must supply a
     * pre-allocated scratch_buf of length >= ntotal; it is zeroed at the start
     * of each call, eliminating a per-query heap allocation.
     *
     * @param xq          Query vector, length d.
     * @param k           Number of NNs to return.
     * @param out_dist    Output distances, length k.
     * @param out_labels  Output global IDs, length k.
     * @param scratch_buf Per-thread uint8_t buffer of length >= ntotal.
     * @param cr          collision_ratio to use.
     * @param cdr         candidate_ratio to use.
     */
    void search_one(
            const float* xq,
            idx_t        k,
            float*       out_dist,
            idx_t*       out_labels,
            uint8_t*     scratch_buf,
            float        cr,
            float        cdr) const;

    /** Single-query wrapper that uses Multi-sequence traversal. */
    void search_one_multisequence(
            const float* xq,
            idx_t        k,
            float*       out_dist,
            idx_t*       out_labels,
            uint8_t*     scratch_buf,
            float        cr,
            float        cdr) const;
};

} // namespace faiss