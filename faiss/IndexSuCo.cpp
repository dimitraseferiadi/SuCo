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
    imi.clear();
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
    // Use an IndexFlatL2 over the centroids to find nearest centroid per vector.
    // This reuses FAISS SIMD-accelerated distance computation.
    IndexFlatL2 idx(dim);
    idx.add(ncentroids, cents);

    std::vector<float> dummy_dists(n);
    std::vector<idx_t> assign_l(n);
    idx.search(n, vecs, 1, dummy_dists.data(), assign_l.data());

    for (idx_t i = 0; i < n; ++i) {
        out_assign[i] = static_cast<int32_t>(assign_l[i]);
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

    // If this is the first add, initialise the IMI vector
    if (imi.empty()) {
        imi.resize(nsubspaces);
    }

    // Temporary buffer for one half-subspace's vectors
    std::vector<float>   half_buf(static_cast<size_t>(n) * half_dim);
    std::vector<int32_t> assign(n);

    for (int s = 0; s < nsubspaces; ++s) {
        int32_t* assign_first  = nullptr;
        int32_t* assign_second = nullptr;

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

        // Insert into IMI
        idx_t base_id = ntotal; // global ID of the first vector being added
        for (idx_t i = 0; i < n; ++i) {
            imi[s][{asgn1[i], asgn2[i]}].push_back(base_id + i);
        }
    }

    ntotal += n;
}

// ============================================================================
// Dynamic Activation (Algorithm 3 from the paper)
// ============================================================================

void IndexSuCo::dynamic_activate(
        int                                       subspace_idx,
        const std::vector<float>&                 dists1,
        const std::vector<int32_t>&               idx1,
        const std::vector<float>&                 dists2,
        const std::vector<int32_t>&               idx2,
        idx_t                                     collision_num,
        std::vector<std::pair<int32_t,int32_t>>&  out_cells) const {
    // activated_cell[pos] = {combined_distance, current_idx2_pointer}
    // We represent the activation list as a vector; for moderate ncentroids_half
    // (default 50) a linear scan for the minimum is faster than a heap because
    // cache effects dominate.
    struct ActivatedEntry {
        float    combined_dist;
        int32_t  idx2_ptr; // current pointer into idx2 for this activation row
    };

    const SuCoIMI& cur_imi = imi[subspace_idx];
    std::vector<ActivatedEntry> active;
    active.reserve(ncentroids_half);

    // Initialise: activate the first entry of idx1
    active.push_back({dists1[idx1[0]] + dists2[idx2[0]], 0});

    idx_t retrieved_num = 0;

    while (true) {
        // Find activated entry with minimum combined distance (linear scan)
        int pos = 0;
        float best = active[0].combined_dist;
        for (int z = 1; z < static_cast<int>(active.size()); ++z) {
            if (active[z].combined_dist < best) {
                best = active[z].combined_dist;
                pos  = z;
            }
        }

        int32_t c1 = idx1[pos];
        int32_t c2 = idx2[active[pos].idx2_ptr];

        // Retrieve the IMI cell
        auto it = cur_imi.find({c1, c2});
        if (it != cur_imi.end() && active[pos].combined_dist < FLT_MAX) {
            out_cells.emplace_back(c1, c2);
            retrieved_num += static_cast<idx_t>(it->second.size());
            if (retrieved_num >= collision_num) {
                break;
            }
        }

        // Activate the next row of idx1 if this entry just became active
        // (i.e. its idx2 pointer was 0) and there is a next row
        if (active[pos].idx2_ptr == 0 && pos < ncentroids_half - 1) {
            int next_pos = static_cast<int>(active.size());
            // Only activate if we haven't already
            if (next_pos <= pos + 1) {
                active.push_back(
                        {dists1[idx1[pos + 1]] + dists2[idx2[0]], 0});
            }
        }

        // Advance this row's idx2 pointer, or mark exhausted
        if (active[pos].idx2_ptr < ncentroids_half - 1) {
            active[pos].idx2_ptr++;
            active[pos].combined_dist =
                    dists1[idx1[pos]] + dists2[idx2[active[pos].idx2_ptr]];
        } else {
            active[pos].combined_dist = FLT_MAX; // mark exhausted
        }

        // Safety: if all entries are exhausted, stop
        bool all_exhausted = true;
        for (const auto& e : active) {
            if (e.combined_dist < FLT_MAX) {
                all_exhausted = false;
                break;
            }
        }
        if (all_exhausted) {
            break;
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
        idx_t*       out_labels) const {
    FAISS_THROW_IF_NOT_MSG(ntotal > 0, "IndexSuCo: index is empty");

    // collision_num = alpha * ntotal (number of points to retrieve per subspace)
    const idx_t collision_num = static_cast<idx_t>(
            std::max(1.0f, collision_ratio * static_cast<float>(ntotal)));

    // candidate_num = beta * ntotal (size of re-rank pool)
    const idx_t candidate_num = static_cast<idx_t>(
            std::max(static_cast<float>(k),
                     candidate_ratio * static_cast<float>(ntotal)));

    // -------------------------------------------------------------------------
    // 1. Collision counting: for each subspace, run Dynamic Activation
    //    and increment SC-scores for every retrieved point.
    // -------------------------------------------------------------------------
    std::vector<uint8_t> sc_scores(ntotal, 0); // SC-score per data point

    std::vector<float>   dists1(ncentroids_half), dists2(ncentroids_half);
    std::vector<int32_t> idx1(ncentroids_half),   idx2(ncentroids_half);

    for (int s = 0; s < nsubspaces; ++s) {
        int col_start = s * subspace_dim;

        // Distance from query to each first-half centroid
        const float* cents1 = centroids.data() +
                static_cast<size_t>(s * 2) * ncentroids_half * half_dim;
        for (int c = 0; c < ncentroids_half; ++c) {
            dists1[c] = fvec_L2sqr(
                    xq + col_start,
                    cents1 + static_cast<size_t>(c) * half_dim,
                    half_dim);
        }

        // Distance from query to each second-half centroid
        const float* cents2 = centroids.data() +
                static_cast<size_t>(s * 2 + 1) * ncentroids_half * half_dim;
        for (int c = 0; c < ncentroids_half; ++c) {
            dists2[c] = fvec_L2sqr(
                    xq + col_start + half_dim,
                    cents2 + static_cast<size_t>(c) * half_dim,
                    half_dim);
        }

        // Argsort dists1 and dists2 ascending
        std::iota(idx1.begin(), idx1.end(), 0);
        std::sort(idx1.begin(), idx1.end(), [&](int a, int b) {
            return dists1[a] < dists1[b];
        });
        std::iota(idx2.begin(), idx2.end(), 0);
        std::sort(idx2.begin(), idx2.end(), [&](int a, int b) {
            return dists2[a] < dists2[b];
        });

        // Dynamic Activation: collect IMI cells
        std::vector<std::pair<int32_t,int32_t>> cells;
        dynamic_activate(s, dists1, idx1, dists2, idx2, collision_num, cells);

        // Increment SC-scores for all retrieved points
        for (const auto& cell : cells) {
            auto it = imi[s].find(cell);
            if (it != imi[s].end()) {
                for (idx_t vid : it->second) {
                    if (sc_scores[vid] < 255u) {
                        sc_scores[vid]++;
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. SC-score selection: choose the top-candidate_num points by SC-score.
    //    (mirrors the "release" logic in the original query.cpp)
    //    We find the minimum SC-score threshold such that at least
    //    candidate_num points are selected.
    // -------------------------------------------------------------------------

    // Count how many points have each SC-score value (0 .. nsubspaces)
    std::vector<idx_t> score_hist(nsubspaces + 1, 0);
    for (idx_t i = 0; i < ntotal; ++i) {
        score_hist[sc_scores[i]]++;
    }

    // Walk from high to low score to find the cutoff threshold
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

    // Collect candidate indices (all points with sc_scores >= threshold_score)
    std::vector<idx_t> candidates;
    candidates.reserve(candidate_num + score_hist[threshold_score]);
    for (idx_t i = 0; i < ntotal; ++i) {
        if (sc_scores[i] >= static_cast<uint8_t>(threshold_score)) {
            candidates.push_back(i);
        }
    }

    // -------------------------------------------------------------------------
    // 3. Re-ranking: compute exact L2 distances for all candidates, return top-k
    // -------------------------------------------------------------------------
    const idx_t nc = static_cast<idx_t>(candidates.size());
    std::vector<float> cand_dists(nc);

    for (idx_t j = 0; j < nc; ++j) {
        idx_t vid = candidates[j];
        cand_dists[j] = fvec_L2sqr(
                xq,
                xb.data() + static_cast<size_t>(vid) * d,
                d);
    }

    // Partial sort to get top-k
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
    // Pad with -1 / +inf if fewer candidates than k
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
        const SearchParameters*) const {
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "IndexSuCo: must call train() before search()");
    FAISS_THROW_IF_NOT_MSG(k > 0, "IndexSuCo: k must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            k <= ntotal,
            "IndexSuCo: k cannot exceed the number of indexed vectors");

    for (idx_t i = 0; i < n; ++i) {
        search_one(
                x + i * d,
                k,
                distances + i * k,
                labels    + i * k);
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
    // magic + version
    const uint32_t magic   = 0x5375436F; // 'SuCo'
    const uint32_t version = 1;
    fwrite(&magic,   sizeof(uint32_t), 1, fp);
    fwrite(&version, sizeof(uint32_t), 1, fp);

    // scalar fields
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

    // ---- IMI ----
    for (int s = 0; s < nsubspaces; ++s) {
        size_t nbuckets = imi[s].size();
        fwrite(&nbuckets, sizeof(size_t), 1, fp);
        for (const auto& kv : imi[s]) {
            int32_t c1 = kv.first.first;
            int32_t c2 = kv.first.second;
            fwrite(&c1, sizeof(int32_t), 1, fp);
            fwrite(&c2, sizeof(int32_t), 1, fp);
            size_t nids = kv.second.size();
            fwrite(&nids, sizeof(size_t), 1, fp);
            fwrite(kv.second.data(), sizeof(idx_t), nids, fp);
        }
    }

    // ---- raw vectors ----
    size_t nxb = xb.size();
    fwrite(&nxb, sizeof(size_t), 1, fp);
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
    fread(&magic,   sizeof(uint32_t), 1, fp);
    fread(&version, sizeof(uint32_t), 1, fp);
    FAISS_THROW_IF_NOT_MSG(
            magic == 0x5375436F,
            "IndexSuCo::read_index: bad magic, not a SuCo index file");
    FAISS_THROW_IF_NOT_MSG(
            version == 1,
            "IndexSuCo::read_index: unsupported version");

    int64_t dd;
    fread(&dd,               sizeof(int64_t), 1, fp);
    fread(&nsubspaces,       sizeof(int),     1, fp);
    fread(&ncentroids_half,  sizeof(int),     1, fp);
    fread(&collision_ratio,  sizeof(float),   1, fp);
    fread(&candidate_ratio,  sizeof(float),   1, fp);
    fread(&niter,            sizeof(int),     1, fp);
    int64_t ntotal_r;
    fread(&ntotal_r, sizeof(int64_t), 1, fp);

    d           = static_cast<idx_t>(dd);
    ntotal      = static_cast<idx_t>(ntotal_r);
    subspace_dim = static_cast<int>(d) / nsubspaces;
    half_dim     = subspace_dim / 2;

    // centroids
    size_t ncents = (size_t)nsubspaces * 2 * ncentroids_half * half_dim;
    centroids.resize(ncents);
    fread(centroids.data(), sizeof(float), ncents, fp);

    // IMI
    imi.resize(nsubspaces);
    for (int s = 0; s < nsubspaces; ++s) {
        imi[s].clear();
        size_t nbuckets;
        fread(&nbuckets, sizeof(size_t), 1, fp);
        imi[s].reserve(nbuckets);
        for (size_t b = 0; b < nbuckets; ++b) {
            int32_t c1, c2;
            fread(&c1, sizeof(int32_t), 1, fp);
            fread(&c2, sizeof(int32_t), 1, fp);
            size_t nids;
            fread(&nids, sizeof(size_t), 1, fp);
            std::vector<idx_t> ids(nids);
            fread(ids.data(), sizeof(idx_t), nids, fp);
            imi[s][{c1, c2}] = std::move(ids);
        }
    }

    // raw vectors
    size_t nxb;
    fread(&nxb, sizeof(size_t), 1, fp);
    xb.resize(nxb);
    if (nxb > 0) {
        fread(xb.data(), sizeof(float), nxb, fp);
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