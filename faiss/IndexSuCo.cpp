/**
 * IndexSuCo.cpp
 *
 * FAISS implementation of the SuCo index.
 * See IndexSuCo.h for full algorithm description.
 *
 * Changes from original implementation
 * =====================================
 * 1. SC-Linear baseline (Algorithm 1) added as search_linear().
 *
 * 2. train() no longer throws when called on an already-trained instance.
 *    Re-training is now idiomatic FAISS behaviour: existing centroids, IMI
 *    buckets, and stored vectors are cleared before the new run begins.
 *
 * 3. Candidate count is now exact.  The original histogram threshold could
 *    include all points sharing the boundary SC-score, over-shooting beta*n by
 *    up to score_hist[threshold] - 1.  The new rerank() helper selects exactly
 *    candidate_num points: all with score > threshold unconditionally, then
 *    enough at score == threshold to fill the budget.
 *
 * 4. Batch query parallelism is now at the query level.  search() and
 *    search_multisequence() allocate one scratch buffer per OpenMP thread and
 *    run an outer parallel-for over queries.  search_one() is fully
 *    single-threaded; there are no nested parallel regions.
 *
 * 5. SC-score selection and re-ranking are factored into a shared rerank()
 *    helper, removing the code duplication between search_one() and
 *    search_one_multisequence().
 *
 * 6. Persistence uses FAISS IOWriter / IOReader.  The binary format is
 *    unchanged (version 3), so existing index files remain readable.  The
 *    file-path write_index(const char*) / read_index(const char*) methods are
 *    now thin wrappers over the IOWriter / IOReader variants.
 */

#include <faiss/IndexSuCo.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/utils/distances.h>

namespace faiss {

// ============================================================================
// IO helper macros  (modelled on FAISS internal convention)
// ============================================================================

// Write a single POD value
#define SUCO_WRITE1(x) \
    f->operator()(&(x), sizeof(x), 1)

// Write a std::vector: length (uint64_t) followed by the raw elements
#define SUCO_WRITEVEC(vec)                                           \
    do {                                                             \
        uint64_t _sz = static_cast<uint64_t>((vec).size());         \
        SUCO_WRITE1(_sz);                                            \
        if (_sz)                                                     \
            f->operator()(                                           \
                    (vec).data(), sizeof((vec)[0]),                  \
                    static_cast<size_t>(_sz));                       \
    } while (0)

// Read a single POD value
#define SUCO_READ1(x) \
    f->operator()(&(x), sizeof(x), 1)

// Read a std::vector written by SUCO_WRITEVEC
#define SUCO_READVEC(vec)                                            \
    do {                                                             \
        uint64_t _sz;                                                \
        SUCO_READ1(_sz);                                             \
        (vec).resize(static_cast<size_t>(_sz));                      \
        if (_sz)                                                     \
            f->operator()(                                           \
                    (vec).data(), sizeof((vec)[0]),                  \
                    static_cast<size_t>(_sz));                       \
    } while (0)

// ============================================================================
// Constructor
// ============================================================================

IndexSuCo::IndexSuCo(
        idx_t d,
        int   nsubspaces,
        int   ncentroids_half,
        float collision_ratio,
        float candidate_ratio,
        int   niter)
        : Index(d, METRIC_L2),
          nsubspaces(nsubspaces),
          ncentroids_half(ncentroids_half),
          collision_ratio(collision_ratio),
          candidate_ratio(candidate_ratio),
          niter(niter),
          subspace_dim(0),
          half_dim(0) {
    FAISS_THROW_IF_NOT_MSG(d > 0, "IndexSuCo: d must be > 0");
    FAISS_THROW_IF_NOT_MSG(nsubspaces > 0, "IndexSuCo: nsubspaces must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            ncentroids_half > 0,
            "IndexSuCo: ncentroids_half must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            nsubspaces <= 255,
            "IndexSuCo: nsubspaces must be <= 255 (SC-scores are uint8)");
    FAISS_THROW_IF_NOT_MSG(
            d % nsubspaces == 0,
            "IndexSuCo: d must be divisible by nsubspaces");
    FAISS_THROW_IF_NOT_MSG(
            (d / nsubspaces) % 2 == 0,
            "IndexSuCo: d/nsubspaces must be even "
            "(each subspace is split into two halves)");
    FAISS_THROW_IF_NOT_MSG(
            collision_ratio > 0.0f && collision_ratio < 1.0f,
            "IndexSuCo: collision_ratio must be in (0,1)");
    FAISS_THROW_IF_NOT_MSG(
            candidate_ratio > 0.0f && candidate_ratio < 1.0f,
            "IndexSuCo: candidate_ratio must be in (0,1)");

    subspace_dim = static_cast<int>(d) / nsubspaces;
    half_dim     = subspace_dim / 2;
    is_trained   = false;
}

// ============================================================================
// reset
// ============================================================================

void IndexSuCo::reset() {
    centroids.clear();
    inv_lists.clear();
    xb.clear();
    ntotal     = 0;
    is_trained = false;
}

// ============================================================================
// Internal: assign vectors to nearest centroid
// ============================================================================

void IndexSuCo::assign_to_centroids(
        idx_t        n,
        int          dim,
        const float* vecs,
        int          ncentroids,
        const float* cents,
        int32_t*     out_assign) {
    // knn_L2sqr(k=1) leverages FAISS BLAS/SIMD vectorisation.
    std::vector<float> dists(n);
    std::vector<idx_t> labels(n);
    knn_L2sqr(
            vecs, cents,
            static_cast<size_t>(dim),
            static_cast<size_t>(n),
            static_cast<size_t>(ncentroids),
            static_cast<size_t>(1),
            dists.data(), labels.data());
    for (idx_t i = 0; i < n; ++i)
        out_assign[i] = static_cast<int32_t>(labels[i]);
}

// ============================================================================
// train
// ============================================================================

void IndexSuCo::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n > 0, "IndexSuCo: need at least one training vector");

    // --- Re-training: discard all previously learned state ------------------
    // Changing the centroids invalidates all IMI assignments, so we reset to
    // an empty-but-about-to-be-trained state.  This mirrors faiss::IndexIVF.
    centroids.clear();
    inv_lists.clear();
    xb.clear();
    ntotal     = 0;
    is_trained = false;

    if (verbose) {
        printf("IndexSuCo::train: d=%ld  nsubspaces=%d  ncentroids_half=%d  "
               "n=%ld\n",
               (long)d, nsubspaces, ncentroids_half, (long)n);
    }

    // Allocate centroid storage:
    //   nsubspaces * 2 halves  *  ncentroids_half centroids  *  half_dim floats
    centroids.assign(
            static_cast<size_t>(nsubspaces) * 2 * ncentroids_half * half_dim,
            0.0f);

    // Temporary contiguous buffer for one half-subspace's training data
    std::vector<float> half_buf(static_cast<size_t>(n) * half_dim);

        // Prepare an empty, fully-shaped IMI so trained-but-empty indexes can be
        // serialized and searched consistently before add().
        inv_lists.resize(nsubspaces);
        for (int s = 0; s < nsubspaces; ++s) {
                inv_lists[s].resize(
                                static_cast<size_t>(ncentroids_half) * ncentroids_half);
        }

    for (int s = 0; s < nsubspaces; ++s) {
        for (int half = 0; half < 2; ++half) {
            // --- extract half-subspace training vectors ----------------------
            int col_start = s * subspace_dim + half * half_dim;
            for (idx_t i = 0; i < n; ++i) {
                std::memcpy(
                        half_buf.data() + i * half_dim,
                        x + i * d + col_start,
                        half_dim * sizeof(float));
            }

            // --- K-means -----------------------------------------------------
            ClusteringParameters cp;
            cp.niter   = niter;
            cp.verbose = verbose;
            cp.seed    = 42 + s * 2 + half; // deterministic per half

            Clustering clus(half_dim, ncentroids_half, cp);
            IndexFlatL2 assign_idx(half_dim);
            clus.train(n, half_buf.data(), assign_idx);

            size_t cent_offset = static_cast<size_t>(s * 2 + half)
                    * ncentroids_half * half_dim;
            std::memcpy(
                    centroids.data() + cent_offset,
                    clus.centroids.data(),
                    static_cast<size_t>(ncentroids_half) * half_dim *
                            sizeof(float));

            if (verbose)
                printf("  trained subspace %d half %d\n", s, half);
        }
    }

    is_trained = true;
}

// ============================================================================
// add
// ============================================================================

void IndexSuCo::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "IndexSuCo: must call train() before add()");
    FAISS_THROW_IF_NOT_MSG(n > 0, "IndexSuCo: n must be > 0");

    if (verbose) {
        printf("IndexSuCo::add: adding %ld vectors (current ntotal=%ld)\n",
               (long)n, (long)ntotal);
    }

    // Store raw vectors (needed for re-ranking)
    size_t prev_xb_size = xb.size();
    xb.resize(prev_xb_size + static_cast<size_t>(n) * d);
    std::memcpy(
            xb.data() + prev_xb_size,
            x,
            static_cast<size_t>(n) * d * sizeof(float));

    // Initialise inv_lists on first add
    if (inv_lists.empty()) {
        inv_lists.resize(nsubspaces);
        for (int s = 0; s < nsubspaces; ++s)
            inv_lists[s].resize(
                    static_cast<size_t>(ncentroids_half) * ncentroids_half);
    }

    std::vector<float> half_buf(static_cast<size_t>(n) * half_dim);

    for (int s = 0; s < nsubspaces; ++s) {
        std::vector<int32_t> asgn1(n), asgn2(n);

        for (int half = 0; half < 2; ++half) {
            int col_start = s * subspace_dim + half * half_dim;
            for (idx_t i = 0; i < n; ++i) {
                std::memcpy(
                        half_buf.data() + i * half_dim,
                        x + i * d + col_start,
                        half_dim * sizeof(float));
            }

            const float* cents = centroids.data() +
                    static_cast<size_t>(s * 2 + half) * ncentroids_half *
                            half_dim;
            assign_to_centroids(
                    n, half_dim, half_buf.data(),
                    ncentroids_half, cents,
                    (half == 0) ? asgn1.data() : asgn2.data());
        }

        idx_t base_id = ntotal;
        for (idx_t i = 0; i < n; ++i) {
            size_t bucket =
                    static_cast<size_t>(asgn1[i]) * ncentroids_half + asgn2[i];
            inv_lists[s][bucket].push_back(base_id + i);
        }
    }

    ntotal += n;
}

// ============================================================================
// Dynamic Activation  (Algorithm 3 from the paper)
// ============================================================================

void IndexSuCo::dynamic_activate(
        int            subspace_idx,
        const float*   dists1,
        const int32_t* idx1,
        const float*   dists2,
        const int32_t* idx2,
        idx_t          collision_num,
        uint8_t*       sc_scores) const {
    struct ActivatedEntry {
        float   combined_dist;
        int32_t idx2_ptr; // current position in the sorted idx2 array
    };

    const auto& lists = inv_lists[subspace_idx];
    std::vector<ActivatedEntry> active;
    active.reserve(ncentroids_half);

    // Activate the first row of idx1 (Algorithm 3, lines 3–4)
    active.push_back({dists1[idx1[0]] + dists2[idx2[0]], 0});

    idx_t retrieved_num   = 0;
    int   exhausted_count = 0; // entries whose combined_dist == FLT_MAX

    while (exhausted_count < static_cast<int>(active.size())) {
        // Find activated entry with minimum combined distance (linear scan –
        // active is small, at most ncentroids_half entries, Algorithm 3 line 6)
        int   pos  = 0;
        float best = active[0].combined_dist;
        for (int z = 1; z < static_cast<int>(active.size()); ++z) {
            if (active[z].combined_dist < best) {
                best = active[z].combined_dist;
                pos  = z;
            }
        }

        const int32_t c1 = idx1[pos];
        const int32_t c2 = idx2[active[pos].idx2_ptr];

        // Retrieve the IMI bucket and increment sc_scores (lines 7–9)
        const size_t  bucket = static_cast<size_t>(c1) * ncentroids_half + c2;
        const auto&   lst    = lists[bucket];
        for (idx_t vid : lst)
            sc_scores[vid]++;
        retrieved_num += static_cast<idx_t>(lst.size());
        if (retrieved_num >= collision_num) // lines 10–11
            break;

        // Activate next row of idx1 when pos is visited for the first time
        // (Algorithm 3 lines 12–14).  The guard `pos+1 == active.size()`
        // ensures we only push a new entry once, even if pos is revisited.
        if (active[pos].idx2_ptr == 0
                && pos + 1 < ncentroids_half
                && pos + 1 == static_cast<int>(active.size())) {
            active.push_back(
                    {dists1[idx1[pos + 1]] + dists2[idx2[0]], 0});
        }

        // Advance this row's idx2 pointer, or mark it exhausted (lines 15–17)
        if (active[pos].idx2_ptr < ncentroids_half - 1) {
            active[pos].idx2_ptr++;
            active[pos].combined_dist =
                    dists1[idx1[pos]] +
                    dists2[idx2[active[pos].idx2_ptr]];
        } else {
            active[pos].combined_dist = FLT_MAX;
            exhausted_count++;
        }
    }
}

// ============================================================================
// rerank  – shared SC-score selection and exact L2 re-ranking
// ============================================================================

void IndexSuCo::rerank(
        const float*   xq,
        idx_t          k,
        idx_t          candidate_num,
        const uint8_t* sc_scores,
        float*         out_dist,
        idx_t*         out_labels) const {
    // -------------------------------------------------------------------------
    // Degenerate case: request more candidates than exist → compare everything.
    // -------------------------------------------------------------------------
    if (candidate_num >= ntotal) {
        std::vector<float> all_dists(ntotal);
        for (idx_t j = 0; j < ntotal; ++j)
            all_dists[j] = fvec_L2sqr(
                    xq,
                    xb.data() + static_cast<size_t>(j) * d,
                    static_cast<size_t>(d));

        const idx_t result_k = std::min(k, ntotal);
        std::vector<idx_t> order(ntotal);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(
                order.begin(), order.begin() + result_k, order.end(),
                [&](idx_t a, idx_t b) {
                    return all_dists[a] < all_dists[b];
                });
        for (idx_t j = 0; j < result_k; ++j) {
            out_dist[j]   = all_dists[order[j]];
            out_labels[j] = order[j];
        }
        for (idx_t j = result_k; j < k; ++j) {
            out_dist[j]   = FLT_MAX;
            out_labels[j] = -1;
        }
        return;
    }

    // -------------------------------------------------------------------------
    // Build a histogram of SC-scores  (O(ntotal), scores in [0, nsubspaces]).
    // -------------------------------------------------------------------------
    const int max_score = nsubspaces;
    std::vector<idx_t> score_hist(max_score + 1, 0);
    for (idx_t i = 0; i < ntotal; ++i) {
        const int sc = std::min(static_cast<int>(sc_scores[i]), max_score);
        score_hist[sc]++;
    }

    // -------------------------------------------------------------------------
    // Find the threshold score T:
    //   • all points with score > T are included unconditionally
    //     (their count is stored in cumul_above)
    //   • points at score == T fill the remainder of the budget
    //
    // Because candidate_num < ntotal, at least one score bucket is non-empty
    // with sum == ntotal, so the loop always breaks before sc reaches -1.
    // -------------------------------------------------------------------------
    idx_t cumul_above    = 0;
    int   threshold_score = 0;

    for (int sc = max_score; sc >= 0; --sc) {
        if (cumul_above + score_hist[sc] <= candidate_num) {
            cumul_above += score_hist[sc];
        } else {
            threshold_score = sc;
            break;
        }
    }

    // How many slots remain at the threshold tier
    const idx_t budget_at = candidate_num - cumul_above;

    // -------------------------------------------------------------------------
    // Gather exactly candidate_num candidates in a single O(ntotal) pass.
    //   • sc > threshold_score  → always include
    //   • sc == threshold_score → include up to budget_at (in index order)
    //   • sc < threshold_score  → never include
    // This eliminates the original overshoot of up to score_hist[T]-1 points.
    // -------------------------------------------------------------------------
    std::vector<idx_t> candidates;
    candidates.reserve(candidate_num);
    idx_t at_count = 0;

    for (idx_t i = 0; i < ntotal; ++i) {
        const int sc = static_cast<int>(sc_scores[i]);
        if (sc > threshold_score) {
            candidates.push_back(i);
        } else if (sc == threshold_score && at_count < budget_at) {
            candidates.push_back(i);
            ++at_count;
        }
    }

    // -------------------------------------------------------------------------
    // Exact L2 re-ranking over the candidate set.
    // -------------------------------------------------------------------------
    const idx_t nc = static_cast<idx_t>(candidates.size());
    std::vector<float> cand_dists(nc);

    for (idx_t j = 0; j < nc; ++j)
        cand_dists[j] = fvec_L2sqr(
                xq,
                xb.data() + static_cast<size_t>(candidates[j]) * d,
                static_cast<size_t>(d));

    const idx_t result_k = std::min(k, nc);
    std::vector<idx_t> order(nc);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
            order.begin(), order.begin() + result_k, order.end(),
            [&](idx_t a, idx_t b) { return cand_dists[a] < cand_dists[b]; });

    for (idx_t j = 0; j < result_k; ++j) {
        out_dist[j]   = cand_dists[order[j]];
        out_labels[j] = candidates[order[j]];
    }
    for (idx_t j = result_k; j < k; ++j) {
        out_dist[j]   = FLT_MAX;
        out_labels[j] = -1;
    }
}

// ============================================================================
// search_one  (single-query Dynamic Activation search – fully sequential)
// ============================================================================

void IndexSuCo::search_one(
        const float* xq,
        idx_t        k,
        float*       out_dist,
        idx_t*       out_labels,
        uint8_t*     scratch_buf,
        float        cr,
        float        cdr) const {
    FAISS_THROW_IF_NOT_MSG(ntotal > 0, "IndexSuCo: index is empty");

    const idx_t collision_num = static_cast<idx_t>(
            std::max(1.0f, cr * static_cast<float>(ntotal)));
    const idx_t candidate_num = static_cast<idx_t>(
            std::max(static_cast<float>(k),
                     cdr * static_cast<float>(ntotal)));

    // ---- 1. Collision counting: Dynamic Activation per subspace -------------
    std::fill(scratch_buf, scratch_buf + ntotal, uint8_t(0));
    uint8_t* sc_scores = scratch_buf;

    std::vector<float>   dists1(ncentroids_half), dists2(ncentroids_half);
    std::vector<int32_t> idx1(ncentroids_half),   idx2(ncentroids_half);

    for (int s = 0; s < nsubspaces; ++s) {
        const int col_start = s * subspace_dim;

        // Batch centroid distance computation (vectorised via fvec_L2sqr_ny)
        const float* cents1 = centroids.data() +
                static_cast<size_t>(s * 2) * ncentroids_half * half_dim;
        fvec_L2sqr_ny(
                dists1.data(),
                xq + col_start,
                cents1,
                static_cast<size_t>(half_dim),
                static_cast<size_t>(ncentroids_half));

        const float* cents2 = centroids.data() +
                static_cast<size_t>(s * 2 + 1) * ncentroids_half * half_dim;
        fvec_L2sqr_ny(
                dists2.data(),
                xq + col_start + half_dim,
                cents2,
                static_cast<size_t>(half_dim),
                static_cast<size_t>(ncentroids_half));

        std::iota(idx1.begin(), idx1.end(), 0);
        std::sort(idx1.begin(), idx1.end(),
                  [&](int a, int b) { return dists1[a] < dists1[b]; });
        std::iota(idx2.begin(), idx2.end(), 0);
        std::sort(idx2.begin(), idx2.end(),
                  [&](int a, int b) { return dists2[a] < dists2[b]; });

        dynamic_activate(
                s,
                dists1.data(), idx1.data(),
                dists2.data(), idx2.data(),
                collision_num,
                sc_scores);
    }

    // ---- 2. Candidate selection + exact re-ranking --------------------------
    rerank(xq, k, candidate_num, sc_scores, out_dist, out_labels);
}

// ============================================================================
// search  (batch entry point – parallelises over queries)
// ============================================================================

void IndexSuCo::search(
        idx_t                   n,
        const float*            x,
        idx_t                   k,
        float*                  distances,
        idx_t*                  labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "IndexSuCo: must call train() before search()");
    FAISS_THROW_IF_NOT_MSG(k > 0, "IndexSuCo: k must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            k <= ntotal,
            "IndexSuCo: k cannot exceed the number of indexed vectors");

    // Extract per-search ratio overrides
    float cr  = collision_ratio;
    float cdr = candidate_ratio;
    if (params) {
        const auto* sp = dynamic_cast<const SearchParametersSuCo*>(params);
        FAISS_THROW_IF_NOT_MSG(
                sp,
                "IndexSuCo::search: params must be of type "
                "SearchParametersSuCo");
                if (sp->collision_ratio > 0.0f) {
                        FAISS_THROW_IF_NOT_MSG(
                                        sp->collision_ratio < 1.0f,
                                        "IndexSuCo::search: collision_ratio override must be in (0,1)");
                        cr = sp->collision_ratio;
                }
                if (sp->candidate_ratio > 0.0f) {
                        FAISS_THROW_IF_NOT_MSG(
                                        sp->candidate_ratio < 1.0f,
                                        "IndexSuCo::search: candidate_ratio override must be in (0,1)");
                        cdr = sp->candidate_ratio;
                }
    }

    // Allocate one scratch buffer per OpenMP thread.
    // search_one() is fully sequential; the outer parallel-for provides all
    // query-level parallelism with no nested OMP regions.
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    std::vector<std::vector<uint8_t>> scratch_bufs(
            nthreads,
            std::vector<uint8_t>(static_cast<size_t>(ntotal), 0));

#pragma omp parallel for schedule(dynamic, 1)
    for (idx_t i = 0; i < n; ++i) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        search_one(
                x + i * d,
                k,
                distances + i * k,
                labels    + i * k,
                scratch_bufs[tid].data(),
                cr, cdr);
    }
}

// ============================================================================
// get_sc_scores  (single-query, no re-ranking)
// ============================================================================

void IndexSuCo::get_sc_scores(
        const float* xq,
        float        cr,
        int32_t*     out_scores) const {
    FAISS_THROW_IF_NOT_MSG(
            is_trained && ntotal > 0,
            "IndexSuCo::get_sc_scores: index not trained or empty");

    const float eff_cr = (cr > 0.f) ? cr : collision_ratio;
    FAISS_THROW_IF_NOT_MSG(
            eff_cr > 0.f && eff_cr < 1.f,
            "IndexSuCo::get_sc_scores: collision_ratio must be in (0,1)");
    const idx_t collision_num = static_cast<idx_t>(
            std::max(1.f, eff_cr * static_cast<float>(ntotal)));

    std::vector<uint8_t>  sc_buf(ntotal, 0);
    std::vector<float>    dists1(ncentroids_half), dists2(ncentroids_half);
    std::vector<int32_t>  idx1(ncentroids_half),   idx2(ncentroids_half);

    for (int s = 0; s < nsubspaces; ++s) {
        const int col_start = s * subspace_dim;

        const float* cents1 = centroids.data() +
                static_cast<size_t>(s * 2) * ncentroids_half * half_dim;
        fvec_L2sqr_ny(
                dists1.data(), xq + col_start, cents1,
                static_cast<size_t>(half_dim),
                static_cast<size_t>(ncentroids_half));

        const float* cents2 = centroids.data() +
                static_cast<size_t>(s * 2 + 1) * ncentroids_half * half_dim;
        fvec_L2sqr_ny(
                dists2.data(), xq + col_start + half_dim, cents2,
                static_cast<size_t>(half_dim),
                static_cast<size_t>(ncentroids_half));

        std::iota(idx1.begin(), idx1.end(), 0);
        std::sort(idx1.begin(), idx1.end(),
                  [&](int a, int b) { return dists1[a] < dists1[b]; });
        std::iota(idx2.begin(), idx2.end(), 0);
        std::sort(idx2.begin(), idx2.end(),
                  [&](int a, int b) { return dists2[a] < dists2[b]; });

        dynamic_activate(
                s, dists1.data(), idx1.data(),
                dists2.data(), idx2.data(),
                collision_num, sc_buf.data());
    }

    for (idx_t i = 0; i < ntotal; ++i)
        out_scores[i] = static_cast<int32_t>(sc_buf[i]);
}

// ============================================================================
// dynamic_activate_multisequence  (classical heap-based IMI traversal)
// ============================================================================

void IndexSuCo::dynamic_activate_multisequence(
        int            subspace_idx,
        const float*   dists1,
        const int32_t* idx1,
        const float*   dists2,
        const int32_t* idx2,
        idx_t          collision_num,
        uint8_t*       sc_scores) const {
    using Entry = std::tuple<float, int32_t, int32_t>; // (dist, i, j)

    const int nc = ncentroids_half;

    // Flat nc×nc visited bitset (nc ≤ 100 → at most 10 000 bits)
    std::vector<bool> visited(static_cast<size_t>(nc) * nc, false);

    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    heap.push({dists1[idx1[0]] + dists2[idx2[0]], 0, 0});
    visited[0] = true;

    const auto& lists = inv_lists[subspace_idx];
    idx_t retrieved    = 0;

    while (!heap.empty()) {
        const auto [dist, i, j] = heap.top();
        heap.pop();

        const int32_t c1  = idx1[i], c2 = idx2[j];
        const auto&   lst = lists[static_cast<size_t>(c1) * nc + c2];
        for (idx_t vid : lst)
            sc_scores[vid]++;
        retrieved += static_cast<idx_t>(lst.size());
        if (retrieved >= collision_num)
            break;

        if (i + 1 < nc) {
            const size_t vi = static_cast<size_t>(i + 1) * nc + j;
            if (!visited[vi]) {
                visited[vi] = true;
                heap.push({dists1[idx1[i + 1]] + dists2[idx2[j]], i + 1, j});
            }
        }
        if (j + 1 < nc) {
            const size_t vj = static_cast<size_t>(i) * nc + j + 1;
            if (!visited[vj]) {
                visited[vj] = true;
                heap.push({dists1[idx1[i]] + dists2[idx2[j + 1]], i, j + 1});
            }
        }
    }
}

// ============================================================================
// search_one_multisequence  (single-query, heap traversal – fully sequential)
// ============================================================================

void IndexSuCo::search_one_multisequence(
        const float* xq,
        idx_t        k,
        float*       out_dist,
        idx_t*       out_labels,
        uint8_t*     scratch_buf,
        float        cr,
        float        cdr) const {
    FAISS_THROW_IF_NOT_MSG(ntotal > 0, "IndexSuCo: index is empty");

    const idx_t collision_num = static_cast<idx_t>(
            std::max(1.f, cr * static_cast<float>(ntotal)));
    const idx_t candidate_num = static_cast<idx_t>(
            std::max(static_cast<float>(k),
                     cdr * static_cast<float>(ntotal)));

    std::fill(scratch_buf, scratch_buf + ntotal, uint8_t(0));
    uint8_t* sc_scores = scratch_buf;

    std::vector<float>   dists1(ncentroids_half), dists2(ncentroids_half);
    std::vector<int32_t> idx1(ncentroids_half),   idx2(ncentroids_half);

    for (int s = 0; s < nsubspaces; ++s) {
        const int col_start = s * subspace_dim;

        const float* cents1 = centroids.data() +
                static_cast<size_t>(s * 2) * ncentroids_half * half_dim;
        fvec_L2sqr_ny(
                dists1.data(), xq + col_start, cents1,
                static_cast<size_t>(half_dim),
                static_cast<size_t>(ncentroids_half));

        const float* cents2 = centroids.data() +
                static_cast<size_t>(s * 2 + 1) * ncentroids_half * half_dim;
        fvec_L2sqr_ny(
                dists2.data(), xq + col_start + half_dim, cents2,
                static_cast<size_t>(half_dim),
                static_cast<size_t>(ncentroids_half));

        std::iota(idx1.begin(), idx1.end(), 0);
        std::sort(idx1.begin(), idx1.end(),
                  [&](int a, int b) { return dists1[a] < dists1[b]; });
        std::iota(idx2.begin(), idx2.end(), 0);
        std::sort(idx2.begin(), idx2.end(),
                  [&](int a, int b) { return dists2[a] < dists2[b]; });

        dynamic_activate_multisequence(
                s, dists1.data(), idx1.data(),
                dists2.data(), idx2.data(),
                collision_num, sc_scores);
    }

    rerank(xq, k, candidate_num, sc_scores, out_dist, out_labels);
}

// ============================================================================
// search_multisequence  (batch entry point – parallelises over queries)
// ============================================================================

void IndexSuCo::search_multisequence(
        idx_t        n,
        const float* x,
        idx_t        k,
        float*       distances,
        idx_t*       labels,
        float        cr_override,
        float        cdr_override) const {
    FAISS_THROW_IF_NOT_MSG(
            is_trained,
            "IndexSuCo: must call train() before search_multisequence()");
    FAISS_THROW_IF_NOT_MSG(k > 0 && k <= ntotal, "IndexSuCo: invalid k");

    const float cr  = (cr_override  > 0.f) ? cr_override  : collision_ratio;
    const float cdr = (cdr_override > 0.f) ? cdr_override : candidate_ratio;
    FAISS_THROW_IF_NOT_MSG(
            cr > 0.f && cr < 1.f,
            "IndexSuCo::search_multisequence: collision_ratio must be in (0,1)");
    FAISS_THROW_IF_NOT_MSG(
            cdr > 0.f && cdr < 1.f,
            "IndexSuCo::search_multisequence: candidate_ratio must be in (0,1)");

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    std::vector<std::vector<uint8_t>> scratch_bufs(
            nthreads,
            std::vector<uint8_t>(static_cast<size_t>(ntotal), 0));

#pragma omp parallel for schedule(dynamic, 1)
    for (idx_t i = 0; i < n; ++i) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        search_one_multisequence(
                x + i * d, k,
                distances + i * k, labels + i * k,
                scratch_bufs[tid].data(), cr, cdr);
    }
}

// ============================================================================
// search_linear  (Algorithm 1 from the paper – index-free SC baseline)
// ============================================================================

void IndexSuCo::search_linear(
        idx_t        n,
        const float* x,
        idx_t        k,
        float*       distances,
        idx_t*       labels,
        float        cr_override,
        float        cdr_override) const {
    FAISS_THROW_IF_NOT_MSG(
            is_trained && ntotal > 0,
            "IndexSuCo::search_linear: index not trained or empty");
    FAISS_THROW_IF_NOT_MSG(
            k > 0 && k <= ntotal,
            "IndexSuCo::search_linear: invalid k");

    const float cr  = (cr_override  > 0.f) ? cr_override  : collision_ratio;
    const float cdr = (cdr_override > 0.f) ? cdr_override : candidate_ratio;
    FAISS_THROW_IF_NOT_MSG(
            cr > 0.f && cr < 1.f,
            "IndexSuCo::search_linear: collision_ratio must be in (0,1)");
    FAISS_THROW_IF_NOT_MSG(
            cdr > 0.f && cdr < 1.f,
            "IndexSuCo::search_linear: candidate_ratio must be in (0,1)");

    // Pre-allocate per-thread working buffers to avoid repeated heap
    // allocations inside the parallel region.
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    // Per-thread: SC-score array (int32), subspace-distance array, sort order,
    // and a uint8 copy for rerank().
    std::vector<std::vector<int32_t>> tl_sc(
            nthreads, std::vector<int32_t>(ntotal, 0));
    std::vector<std::vector<float>>   tl_dists(
            nthreads, std::vector<float>(ntotal, 0.f));
    std::vector<std::vector<idx_t>>   tl_order(
            nthreads, std::vector<idx_t>(ntotal, 0));
    std::vector<std::vector<uint8_t>> tl_sc_u8(
            nthreads, std::vector<uint8_t>(ntotal, 0));

#pragma omp parallel for schedule(dynamic, 1)
    for (idx_t qi = 0; qi < n; ++qi) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif

        const float* xq       = x         + qi * d;
        float*       out_dist = distances  + qi * k;
        idx_t*       out_ids  = labels     + qi * k;

        const idx_t collision_num = static_cast<idx_t>(
                std::max(1.0f, cr * static_cast<float>(ntotal)));
        const idx_t candidate_num = static_cast<idx_t>(
                std::max(static_cast<float>(k),
                         cdr * static_cast<float>(ntotal)));

        // Borrow this thread's pre-allocated buffers
        std::vector<int32_t>& sc_scores = tl_sc[tid];
        std::vector<float>&   sub_dists = tl_dists[tid];
        std::vector<idx_t>&   order     = tl_order[tid];
        std::vector<uint8_t>& sc_u8     = tl_sc_u8[tid];

        std::fill(sc_scores.begin(), sc_scores.end(), 0);

        // ----- Algorithm 1, lines 4–10: per-subspace collision counting -----
        for (int s = 0; s < nsubspaces; ++s) {
            const int col_start = s * subspace_dim;

            // Exact subspace distance from each database vector to query
            for (idx_t j = 0; j < ntotal; ++j) {
                sub_dists[j] = fvec_L2sqr(
                        xq + col_start,
                        xb.data() + static_cast<size_t>(j) * d + col_start,
                        static_cast<size_t>(subspace_dim));
            }

            // Partial sort: find the collision_num closest in this subspace
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(
                    order.begin(),
                    order.begin() + collision_num,
                    order.end(),
                    [&](idx_t a, idx_t b) {
                        return sub_dists[a] < sub_dists[b];
                    });

            for (idx_t z = 0; z < collision_num; ++z)
                sc_scores[order[z]]++;
        }

        // Convert int32 SC-scores to uint8 for rerank() (max value = nsubspaces
        // which always fits in uint8 since nsubspaces ≤ 255 in practice)
        for (idx_t j = 0; j < ntotal; ++j)
            sc_u8[j] = static_cast<uint8_t>(
                    std::min(sc_scores[j],
                             static_cast<int32_t>(nsubspaces)));

        rerank(xq, k, candidate_num, sc_u8.data(), out_dist, out_ids);
    }
}

// ============================================================================
// write_index  (IOWriter – canonical serialisation)
// ============================================================================

void IndexSuCo::write_index(IOWriter* f) const {
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "IndexSuCo: cannot write untrained index");
        FAISS_THROW_IF_NOT_MSG(
                        ntotal == 0 || inv_lists.size() == static_cast<size_t>(nsubspaces),
                        "IndexSuCo::write_index: inv_lists size mismatch");

    // Header
    const uint32_t magic   = 0x5375436F; // 'SuCo'
    const uint32_t version = 3;
    SUCO_WRITE1(magic);
    SUCO_WRITE1(version);

    int64_t dd = d;
    SUCO_WRITE1(dd);
    SUCO_WRITE1(nsubspaces);
    SUCO_WRITE1(ncentroids_half);
    SUCO_WRITE1(collision_ratio);
    SUCO_WRITE1(candidate_ratio);
    SUCO_WRITE1(niter);
    int64_t ntotal_w = ntotal;
    SUCO_WRITE1(ntotal_w);

    // Centroids (size is implicit from the header fields)
    const size_t ncents =
            static_cast<size_t>(nsubspaces) * 2 * ncentroids_half * half_dim;
    f->operator()(centroids.data(), sizeof(float), ncents);

    // Inverted multi-indexes: for each bucket write a uint64 count then IDs
    const size_t nbuckets =
            static_cast<size_t>(ncentroids_half) * ncentroids_half;
    for (int s = 0; s < nsubspaces; ++s) {
        const std::vector<std::vector<idx_t>>* lists_s =
                (inv_lists.size() == static_cast<size_t>(nsubspaces))
                ? &inv_lists[s]
                : nullptr;
        FAISS_THROW_IF_NOT_MSG(
                !lists_s || lists_s->size() == nbuckets,
                "IndexSuCo::write_index: inv_lists bucket count mismatch");
        for (size_t b = 0; b < nbuckets; ++b) {
            const uint64_t nids = lists_s
                    ? static_cast<uint64_t>((*lists_s)[b].size())
                    : uint64_t(0);
            SUCO_WRITE1(nids);
            if (nids)
                f->operator()(
                        (*lists_s)[b].data(), sizeof(idx_t),
                        static_cast<size_t>(nids));
        }
    }

    // Raw vectors
    uint64_t nxb = static_cast<uint64_t>(xb.size());
    SUCO_WRITE1(nxb);
    if (nxb)
        f->operator()(xb.data(), sizeof(float), static_cast<size_t>(nxb));

    if (verbose)
        printf("IndexSuCo::write_index: serialised (ntotal=%ld)\n",
               (long)ntotal);
}

// ============================================================================
// read_index  (IOReader – canonical deserialisation)
// ============================================================================

void IndexSuCo::read_index(IOReader* f) {
    uint32_t magic, version;
    SUCO_READ1(magic);
    SUCO_READ1(version);
    FAISS_THROW_IF_NOT_MSG(
            magic == 0x5375436F,
            "IndexSuCo::read_index: bad magic, not a SuCo index stream");
    FAISS_THROW_IF_NOT_MSG(
            version >= 1 && version <= 3,
            "IndexSuCo::read_index: unsupported version "
            "(expected 1..3)");

    int64_t dd;
    SUCO_READ1(dd);
    SUCO_READ1(nsubspaces);
    SUCO_READ1(ncentroids_half);
    SUCO_READ1(collision_ratio);
    SUCO_READ1(candidate_ratio);
    SUCO_READ1(niter);
    int64_t ntotal_r;
    SUCO_READ1(ntotal_r);

    d            = static_cast<idx_t>(dd);
    ntotal       = static_cast<idx_t>(ntotal_r);
    metric_type  = METRIC_L2;

    FAISS_THROW_IF_NOT_MSG(d > 0, "IndexSuCo::read_index: invalid d");
    FAISS_THROW_IF_NOT_MSG(
            nsubspaces > 0,
            "IndexSuCo::read_index: invalid nsubspaces");
    FAISS_THROW_IF_NOT_MSG(
            nsubspaces <= 255,
            "IndexSuCo::read_index: nsubspaces must be <= 255");
    FAISS_THROW_IF_NOT_MSG(
            ncentroids_half > 0,
            "IndexSuCo::read_index: invalid ncentroids_half");
    FAISS_THROW_IF_NOT_MSG(
            d % nsubspaces == 0,
            "IndexSuCo::read_index: d must be divisible by nsubspaces");

    subspace_dim = static_cast<int>(d) / nsubspaces;
    half_dim     = subspace_dim / 2;

    FAISS_THROW_IF_NOT_MSG(
            subspace_dim % 2 == 0,
            "IndexSuCo::read_index: subspace_dim must be even");

    // Centroids
    const size_t ncents =
            static_cast<size_t>(nsubspaces) * 2 * ncentroids_half * half_dim;
    centroids.resize(ncents);
    FAISS_THROW_IF_NOT_MSG(
            f->operator()(centroids.data(), sizeof(float), ncents) == ncents,
            "IndexSuCo::read_index: unexpected EOF reading centroids");

    // Inverted multi-indexes
    const size_t nbuckets =
            static_cast<size_t>(ncentroids_half) * ncentroids_half;
    inv_lists.resize(nsubspaces);
    for (int s = 0; s < nsubspaces; ++s) {
        inv_lists[s].resize(nbuckets);
        for (size_t b = 0; b < nbuckets; ++b) {
            uint64_t nids;
            FAISS_THROW_IF_NOT_MSG(
                    SUCO_READ1(nids) == 1,
                    "IndexSuCo::read_index: unexpected EOF reading bucket size");
            inv_lists[s][b].resize(static_cast<size_t>(nids));
            if (nids) {
                FAISS_THROW_IF_NOT_MSG(
                        f->operator()(
                                inv_lists[s][b].data(), sizeof(idx_t),
                                static_cast<size_t>(nids)) ==
                                static_cast<size_t>(nids),
                        "IndexSuCo::read_index: unexpected EOF reading bucket");
            }
        }
    }

    // Raw vectors
    uint64_t nxb;
    FAISS_THROW_IF_NOT_MSG(
            SUCO_READ1(nxb) == 1,
            "IndexSuCo::read_index: unexpected EOF reading vector count");
    xb.resize(static_cast<size_t>(nxb));
    if (nxb) {
        FAISS_THROW_IF_NOT_MSG(
                f->operator()(xb.data(), sizeof(float),
                               static_cast<size_t>(nxb)) ==
                        static_cast<size_t>(nxb),
                "IndexSuCo::read_index: unexpected EOF reading vectors");
    }

    is_trained = true;

    if (verbose)
        printf("IndexSuCo::read_index: loaded "
               "(ntotal=%ld  d=%ld  nsubspaces=%d)\n",
               (long)ntotal, (long)d, nsubspaces);
}

// ============================================================================
// write_index / read_index  (file-path convenience wrappers)
// ============================================================================

void IndexSuCo::write_index(const char* fname) const {
    FileIOWriter writer(fname);
    write_index(static_cast<IOWriter*>(&writer));
}

void IndexSuCo::read_index(const char* fname) {
    FileIOReader reader(fname);
    read_index(static_cast<IOReader*>(&reader));
}

} // namespace faiss