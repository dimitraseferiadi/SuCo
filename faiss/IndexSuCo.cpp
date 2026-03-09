/**
 * IndexSuCo.cpp
 *
 * FAISS implementation of the SuCo index.
 * See IndexSuCo.h for full algorithm description.
 */

#include <faiss/IndexSuCo.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

namespace faiss {

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
    FAISS_THROW_IF_NOT_MSG(ncentroids_half > 0, "IndexSuCo: ncentroids_half must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            d % nsubspaces == 0,
            "IndexSuCo: d must be divisible by nsubspaces");
    FAISS_THROW_IF_NOT_MSG(
            (d / nsubspaces) % 2 == 0,
            "IndexSuCo: d/nsubspaces must be even (each subspace is split into two halves)");
    FAISS_THROW_IF_NOT_MSG(
            collision_ratio > 0.0f && collision_ratio < 1.0f,
            "IndexSuCo: collision_ratio must be in (0,1)");
    FAISS_THROW_IF_NOT_MSG(
            candidate_ratio > 0.0f && candidate_ratio < 1.0f,
            "IndexSuCo: candidate_ratio must be in (0,1)");

    subspace_dim = static_cast<int>(d) / nsubspaces;
    half_dim     = subspace_dim / 2;

    is_trained = false;
}

// ============================================================================
// reset
// ============================================================================

void IndexSuCo::reset() {
    centroids.clear();
    inv_lists.clear();
    xb.clear();
    ntotal = 0;
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
    // Use knn_L2sqr (k=1) to avoid a per-call IndexFlatL2 allocation while
    // still leveraging FAISS BLAS/SIMD vectorisation.
    std::vector<float> dists(n);
    std::vector<idx_t> labels(n);
    knn_L2sqr(
            vecs, cents,
            static_cast<size_t>(dim),
            static_cast<size_t>(n),
            static_cast<size_t>(ncentroids),
            static_cast<size_t>(1),
            dists.data(), labels.data());
    for (idx_t i = 0; i < n; ++i) {
        out_assign[i] = static_cast<int32_t>(labels[i]);
    }
}

// ============================================================================
// train
// ============================================================================

void IndexSuCo::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(!is_trained, "IndexSuCo: already trained");
    FAISS_THROW_IF_NOT_MSG(n > 0, "IndexSuCo: need at least one training vector");

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

    for (int s = 0; s < nsubspaces; ++s) {
        for (int half = 0; half < 2; ++half) {
            // ---- extract half-subspace training vectors --------------------
            int col_start = s * subspace_dim + half * half_dim;
            for (idx_t i = 0; i < n; ++i) {
                std::memcpy(
                        half_buf.data() + i * half_dim,
                        x + i * d + col_start,
                        half_dim * sizeof(float));
            }

            // ---- K-means ---------------------------------------------------
            ClusteringParameters cp;
            cp.niter    = niter;
            cp.verbose  = verbose;
            cp.seed     = 42 + s * 2 + half; // deterministic per half

            Clustering clus(half_dim, ncentroids_half, cp);

            // Use an exact flat index as the assignment oracle
            IndexFlatL2 assign_idx(half_dim);
            clus.train(n, half_buf.data(), assign_idx);

            // Centroids are stored in clus.centroids, row-major
            // [ncentroids_half * half_dim]
            size_t cent_offset = static_cast<size_t>(s * 2 + half)
                    * ncentroids_half * half_dim;
            std::memcpy(
                    centroids.data() + cent_offset,
                    clus.centroids.data(),
                    static_cast<size_t>(ncentroids_half) * half_dim *
                            sizeof(float));

            if (verbose) {
                printf("  trained subspace %d half %d\n", s, half);
            }
        }
    }

    is_trained = true;
}

// ============================================================================
// add
// ============================================================================

void IndexSuCo::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexSuCo: must call train() before add()");
    FAISS_THROW_IF_NOT_MSG(n > 0, "IndexSuCo: n must be > 0");

    if (verbose) {
        printf("IndexSuCo::add: adding %ld vectors (current ntotal=%ld)\n",
               (long)n, (long)ntotal);
    }

    // Store raw vectors (needed for re-ranking)
    size_t prev_xb_size = xb.size();
    xb.resize(prev_xb_size + static_cast<size_t>(n) * d);
    std::memcpy(xb.data() + prev_xb_size, x, static_cast<size_t>(n) * d * sizeof(float));

    // If this is the first add, initialise the inv_lists array
    if (inv_lists.empty()) {
        inv_lists.resize(nsubspaces);
        for (int s = 0; s < nsubspaces; ++s) {
            inv_lists[s].resize(
                    static_cast<size_t>(ncentroids_half) * ncentroids_half);
        }
    }

    // Temporary buffer for one half-subspace's vectors
    std::vector<float>   half_buf(static_cast<size_t>(n) * half_dim);

    for (int s = 0; s < nsubspaces; ++s) {
        // We need assignments for both halves simultaneously to build the IMI
        std::vector<int32_t> asgn1(n), asgn2(n);

        for (int half = 0; half < 2; ++half) {
            int col_start = s * subspace_dim + half * half_dim;

            // Extract half-subspace vectors
            for (idx_t i = 0; i < n; ++i) {
                std::memcpy(
                        half_buf.data() + i * half_dim,
                        x + i * d + col_start,
                        half_dim * sizeof(float));
            }

            // Centroids for this half
            const float* cents = centroids.data() +
                    static_cast<size_t>(s * 2 + half) * ncentroids_half * half_dim;

            assign_to_centroids(
                    n, half_dim,
                    half_buf.data(),
                    ncentroids_half, cents,
                    (half == 0) ? asgn1.data() : asgn2.data());
        }

        // Insert into flat inv_lists
        idx_t base_id = ntotal;
        for (idx_t i = 0; i < n; ++i) {
            size_t bucket = static_cast<size_t>(asgn1[i]) * ncentroids_half
                    + asgn2[i];
            inv_lists[s][bucket].push_back(base_id + i);
        }
    }

    ntotal += n;
}

// ============================================================================
// Dynamic Activation (Algorithm 3 from the paper)
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
        int32_t idx2_ptr;
    };

    const auto& lists = inv_lists[subspace_idx];
    std::vector<ActivatedEntry> active;
    active.reserve(ncentroids_half);

    // Activate the first row of idx1
    active.push_back({dists1[idx1[0]] + dists2[idx2[0]], 0});

    idx_t retrieved_num = 0;
    int   exhausted_count = 0; // number of entries with combined_dist == FLT_MAX

    while (exhausted_count < static_cast<int>(active.size())) {
        // Find activated entry with minimum combined distance (linear scan)
        int   pos  = 0;
        float best = active[0].combined_dist;
        for (int z = 1; z < static_cast<int>(active.size()); ++z) {
            if (active[z].combined_dist < best) {
                best = active[z].combined_dist;
                pos  = z;
            }
        }

        int32_t c1 = idx1[pos];
        int32_t c2 = idx2[active[pos].idx2_ptr];

        // Retrieve the flat inv_list bucket and increment sc_scores
        size_t bucket = static_cast<size_t>(c1) * ncentroids_half + c2;
        const auto& lst = lists[bucket];
        for (idx_t vid : lst) {
            sc_scores[vid]++;
        }
        retrieved_num += static_cast<idx_t>(lst.size());
        if (retrieved_num >= collision_num) {
            break;
        }

        // Activate next row of idx1 if this is the first time processing pos
        if (active[pos].idx2_ptr == 0
                && pos + 1 < ncentroids_half
                && pos + 1 == static_cast<int>(active.size())) {
            active.push_back(
                    {dists1[idx1[pos + 1]] + dists2[idx2[0]], 0});
        }

        // Advance this row's idx2 pointer, or mark exhausted
        if (active[pos].idx2_ptr < ncentroids_half - 1) {
            active[pos].idx2_ptr++;
            active[pos].combined_dist =
                    dists1[idx1[pos]] + dists2[idx2[active[pos].idx2_ptr]];
        } else {
            active[pos].combined_dist = FLT_MAX;
            exhausted_count++;
        }
    }
}

// ============================================================================
// search_one  (single-query search, called from search)
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

    // -------------------------------------------------------------------------
    // 1. Collision counting: run Dynamic Activation per subspace and
    //    increment SC-scores directly.
    //    Use the pre-allocated scratch_buf to eliminate per-query heap alloc.
    // -------------------------------------------------------------------------
    std::fill(scratch_buf, scratch_buf + ntotal, uint8_t(0));
    uint8_t* sc_scores = scratch_buf;

    std::vector<float>   dists1(ncentroids_half), dists2(ncentroids_half);
    std::vector<int32_t> idx1(ncentroids_half),   idx2(ncentroids_half);

    for (int s = 0; s < nsubspaces; ++s) {
        int col_start = s * subspace_dim;

        // Batch all centroid distances in one fvec_L2sqr_ny call (vectorised)
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

    // -------------------------------------------------------------------------
    // 2. SC-score selection: find threshold such that >= candidate_num points
    //    are included in the candidate set.
    // -------------------------------------------------------------------------
    std::vector<idx_t> score_hist(nsubspaces + 1, 0);
    for (idx_t i = 0; i < ntotal; ++i) {
        score_hist[sc_scores[i]]++;
    }

    int threshold_score = 0;
    idx_t cumulative = 0;
    for (int sc = nsubspaces; sc >= 0; --sc) {
        if (cumulative + score_hist[sc] <= candidate_num) {
            cumulative += score_hist[sc];
        } else {
            threshold_score = sc;
            break;
        }
    }

    std::vector<idx_t> candidates;
    candidates.reserve(candidate_num + score_hist[threshold_score]);
    for (idx_t i = 0; i < ntotal; ++i) {
        if (sc_scores[i] >= static_cast<uint8_t>(threshold_score)) {
            candidates.push_back(i);
        }
    }

    // -------------------------------------------------------------------------
    // 3. Re-ranking: gather candidates into a contiguous buffer, then compute
    //    all L2 distances in one vectorised pairwise_L2sqr call.
    // -------------------------------------------------------------------------
    const idx_t nc = static_cast<idx_t>(candidates.size());

    // Gathering eliminates the random-access cache misses that occur when
    // computing distances directly against scattered positions in xb.
    std::vector<float> cand_buf(static_cast<size_t>(nc) * d);
    for (idx_t j = 0; j < nc; ++j) {
        std::memcpy(
                cand_buf.data() + static_cast<size_t>(j) * d,
                xb.data() + static_cast<size_t>(candidates[j]) * d,
                d * sizeof(float));
    }

    std::vector<float> cand_dists(nc);
    pairwise_L2sqr(d, 1, xq, nc, cand_buf.data(), cand_dists.data());

    idx_t result_k = std::min(k, nc);
    std::vector<idx_t> order(nc);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
            order.begin(),
            order.begin() + result_k,
            order.end(),
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
// search  (batch entry point)
// ============================================================================

void IndexSuCo::search(
        idx_t                  n,
        const float*           x,
        idx_t                  k,
        float*                 distances,
        idx_t*                 labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "IndexSuCo: must call train() before search()");
    FAISS_THROW_IF_NOT_MSG(k > 0, "IndexSuCo: k must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            k <= ntotal,
            "IndexSuCo: k cannot exceed the number of indexed vectors");

    // Extract per-search ratio overrides from SearchParametersSuCo, if given.
    float cr  = collision_ratio;
    float cdr = candidate_ratio;
    if (params) {
        const auto* sp = dynamic_cast<const SearchParametersSuCo*>(params);
        FAISS_THROW_IF_NOT_MSG(
                sp,
                "IndexSuCo::search: params must be of type SearchParametersSuCo");
        if (sp->collision_ratio > 0.0f) cr  = sp->collision_ratio;
        if (sp->candidate_ratio > 0.0f) cdr = sp->candidate_ratio;
    }

    // Pre-allocate one sc_scores scratch buffer per OpenMP thread to avoid
    // a per-query heap allocation of O(ntotal) bytes.
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    std::vector<std::vector<uint8_t>> scratch_bufs(
            nthreads, std::vector<uint8_t>(ntotal, 0));

#pragma omp parallel for if (n > 1) schedule(dynamic, 1)
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
                cr,
                cdr);
    }
}

// ============================================================================
// write_index / read_index
// ============================================================================

void IndexSuCo::write_index(const char* fname) const {
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "IndexSuCo: cannot write untrained index");

    FILE* fp = fopen(fname, "wb");
    FAISS_THROW_IF_NOT_FMT(fp, "IndexSuCo::write_index: cannot open '%s'", fname);

    // ---- header ----
    const uint32_t magic   = 0x5375436F; // 'SuCo'
    const uint32_t version = 3;          // v3: uint64_t size fields
    fwrite(&magic,   sizeof(uint32_t), 1, fp);
    fwrite(&version, sizeof(uint32_t), 1, fp);

    int64_t dd = d;
    fwrite(&dd,               sizeof(int64_t), 1, fp);
    fwrite(&nsubspaces,       sizeof(int),     1, fp);
    fwrite(&ncentroids_half,  sizeof(int),     1, fp);
    fwrite(&collision_ratio,  sizeof(float),   1, fp);
    fwrite(&candidate_ratio,  sizeof(float),   1, fp);
    fwrite(&niter,            sizeof(int),     1, fp);
    int64_t ntotal_w = ntotal;
    fwrite(&ntotal_w,         sizeof(int64_t), 1, fp);

    // ---- centroids ----
    size_t ncents = (size_t)nsubspaces * 2 * ncentroids_half * half_dim;
    fwrite(centroids.data(), sizeof(float), ncents, fp);

    // ---- flat inv_lists ----
    // For each subspace: ncentroids_half^2 buckets, each preceded by its size.
    size_t nbuckets = (size_t)ncentroids_half * ncentroids_half;
    for (int s = 0; s < nsubspaces; ++s) {
        FAISS_THROW_IF_NOT_MSG(
                inv_lists[s].size() == nbuckets,
                "IndexSuCo::write_index: inv_lists size mismatch");
        for (size_t b = 0; b < nbuckets; ++b) {
            uint64_t nids = static_cast<uint64_t>(inv_lists[s][b].size());
            fwrite(&nids, sizeof(uint64_t), 1, fp);
            if (nids > 0) {
                fwrite(inv_lists[s][b].data(), sizeof(idx_t), nids, fp);
            }
        }
    }

    // ---- raw vectors ----
    uint64_t nxb = static_cast<uint64_t>(xb.size());
    fwrite(&nxb, sizeof(uint64_t), 1, fp);
    if (nxb > 0) {
        fwrite(xb.data(), sizeof(float), nxb, fp);
    }

    fclose(fp);
    if (verbose) {
        printf("IndexSuCo::write_index: saved to '%s'\n", fname);
    }
}

void IndexSuCo::read_index(const char* fname) {
    FILE* fp = fopen(fname, "rb");
    FAISS_THROW_IF_NOT_FMT(fp, "IndexSuCo::read_index: cannot open '%s'", fname);

    uint32_t magic, version;
    FAISS_THROW_IF_NOT_MSG(
            fread(&magic,   sizeof(uint32_t), 1, fp) == 1,
            "IndexSuCo::read_index: unexpected EOF reading magic");
    FAISS_THROW_IF_NOT_MSG(
            fread(&version, sizeof(uint32_t), 1, fp) == 1,
            "IndexSuCo::read_index: unexpected EOF reading version");
    FAISS_THROW_IF_NOT_MSG(
            magic == 0x5375436F,
            "IndexSuCo::read_index: bad magic, not a SuCo index file");
    FAISS_THROW_IF_NOT_MSG(
            version == 3,
            "IndexSuCo::read_index: unsupported version (expected 3; "
            "please rebuild the index)");

    int64_t dd;
    FAISS_THROW_IF_NOT_MSG(fread(&dd,              sizeof(int64_t), 1, fp) == 1, "IndexSuCo::read_index: unexpected EOF");
    FAISS_THROW_IF_NOT_MSG(fread(&nsubspaces,      sizeof(int),     1, fp) == 1, "IndexSuCo::read_index: unexpected EOF");
    FAISS_THROW_IF_NOT_MSG(fread(&ncentroids_half, sizeof(int),     1, fp) == 1, "IndexSuCo::read_index: unexpected EOF");
    FAISS_THROW_IF_NOT_MSG(fread(&collision_ratio, sizeof(float),   1, fp) == 1, "IndexSuCo::read_index: unexpected EOF");
    FAISS_THROW_IF_NOT_MSG(fread(&candidate_ratio, sizeof(float),   1, fp) == 1, "IndexSuCo::read_index: unexpected EOF");
    FAISS_THROW_IF_NOT_MSG(fread(&niter,           sizeof(int),     1, fp) == 1, "IndexSuCo::read_index: unexpected EOF");
    int64_t ntotal_r;
    FAISS_THROW_IF_NOT_MSG(fread(&ntotal_r, sizeof(int64_t), 1, fp) == 1, "IndexSuCo::read_index: unexpected EOF");

    d            = static_cast<idx_t>(dd);
    ntotal       = static_cast<idx_t>(ntotal_r);
    subspace_dim = static_cast<int>(d) / nsubspaces;
    half_dim     = subspace_dim / 2;

    // centroids
    size_t ncents = (size_t)nsubspaces * 2 * ncentroids_half * half_dim;
    centroids.resize(ncents);
    FAISS_THROW_IF_NOT_MSG(
            fread(centroids.data(), sizeof(float), ncents, fp) == ncents,
            "IndexSuCo::read_index: unexpected EOF reading centroids");

    // flat inv_lists
    size_t nbuckets = (size_t)ncentroids_half * ncentroids_half;
    inv_lists.resize(nsubspaces);
    for (int s = 0; s < nsubspaces; ++s) {
        inv_lists[s].resize(nbuckets);
        for (size_t b = 0; b < nbuckets; ++b) {
            uint64_t nids;
            FAISS_THROW_IF_NOT_MSG(
                    fread(&nids, sizeof(uint64_t), 1, fp) == 1,
                    "IndexSuCo::read_index: unexpected EOF reading bucket size");
            inv_lists[s][b].resize(static_cast<size_t>(nids));
            if (nids > 0) {
                FAISS_THROW_IF_NOT_MSG(
                        fread(inv_lists[s][b].data(), sizeof(idx_t), nids, fp) == nids,
                        "IndexSuCo::read_index: unexpected EOF reading bucket ids");
            }
        }
    }

    // raw vectors
    uint64_t nxb;
    FAISS_THROW_IF_NOT_MSG(
            fread(&nxb, sizeof(uint64_t), 1, fp) == 1,
            "IndexSuCo::read_index: unexpected EOF reading vector count");
    xb.resize(static_cast<size_t>(nxb));
    if (nxb > 0) {
        FAISS_THROW_IF_NOT_MSG(
                fread(xb.data(), sizeof(float), nxb, fp) == nxb,
                "IndexSuCo::read_index: unexpected EOF reading vectors");
    }

    fclose(fp);
    is_trained = true;

    if (verbose) {
        printf("IndexSuCo::read_index: loaded from '%s'  "
               "(ntotal=%ld  d=%ld  nsubspaces=%d)\n",
               fname, (long)ntotal, (long)d, nsubspaces);
    }
}

} // namespace faiss