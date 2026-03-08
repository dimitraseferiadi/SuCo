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
 */

#pragma once

#include <cstdint>
#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

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
     * @param n        Number of query vectors.
     * @param x        Query vectors, shape [n, d], row-major float32.
     * @param k        Number of nearest neighbors to return.
     * @param distances Output distances,  shape [n, k].
     * @param labels    Output global IDs,  shape [n, k].
     */
    void search(
            idx_t                  n,
            const float*           x,
            idx_t                  k,
            float*                 distances,
            idx_t*                 labels,
            const SearchParameters* params = nullptr) const override;

    /**
     * Reset the index to an untrained, empty state.
     */
    void reset() override;

    // -----------------------------------------------------------------------
    // Persistence helpers
    // -----------------------------------------------------------------------

    /** Write the index (centroids + IMI) to a binary file. */
    void write_index(const char* fname) const;

    /** Read the index from a binary file produced by write_index(). */
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
     * @param subspace_idx   Which subspace's inv_lists to query.
     * @param dists1         Distances from query to each first-half centroid.
     * @param idx1           Argsort of dists1 ascending.
     * @param dists2         Distances from query to each second-half centroid.
     * @param idx2           Argsort of dists2 ascending.
     * @param collision_num  Stop after this many points have been collected.
     * @param sc_scores      SC-score accumulator (length ntotal); incremented
     *                       in-place for every collected point.
     */
    void dynamic_activate(
            int          subspace_idx,
            const float*   dists1,
            const int32_t* idx1,
            const float*   dists2,
            const int32_t* idx2,
            idx_t          collision_num,
            uint8_t*       sc_scores) const;

    /**
     * Search a single query vector.
     * @param xq          Query vector, length d.
     * @param k           Number of NNs to return.
     * @param out_dist    Output distances, length k.
     * @param out_labels  Output global IDs, length k.
     */
    void search_one(
            const float* xq,
            idx_t        k,
            float*       out_dist,
            idx_t*       out_labels) const;
};

} // namespace faiss