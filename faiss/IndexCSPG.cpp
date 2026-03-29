/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

/**
 * Implementation of IndexCSPG (CSPG algorithm from NeurIPS 2024).
 *
 * Key design notes
 * ----------------
 * Construction (Algorithm 2 in the paper):
 *   1. Sample num_routing = floor(n * lambda) routing vectors (first λn by ID).
 *   2. Routing vectors are replicated in ALL m partitions (local IDs 0..R-1).
 *   3. Remaining vectors are randomly assigned to exactly one partition.
 *   4. Build an independent HNSW graph per partition.
 *
 * Search (Algorithm 1 in the paper):
 *   Stage 1 – Fast approaching:
 *     Navigate partition 0's HNSW from entry_point through upper levels via
 *     greedy descent.  Then beam-search at level 0 with ef = ef1.
 *   Stage 2 – Cross-partition expansion:
 *     Starting from the closest point found in stage 1, perform beam search
 *     across ALL partitions with candidate set size ef2.  When a routing
 *     vector is encountered, it is simultaneously inserted into the candidate
 *     sets of all partitions, enabling the search to "cross" between graphs.
 *
 * Distance computation uses fvec_L2sqr (SIMD-optimized squared L2).
 * Graph neighbor access uses HNSW::neighbor_range at level 0.
 */

#include <faiss/IndexCSPG.h>
#include <faiss/IndexFlat.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <queue>
#include <random>
#include <tuple>
#include <vector>

namespace faiss {

// =========================================================================
// Constructor / Destructor
// =========================================================================

IndexCSPG::IndexCSPG(
        int d,
        int M,
        int num_partitions,
        float lambda,
        MetricType metric)
        : Index(d, metric),
          M(M),
          num_partitions(num_partitions),
          lambda(lambda) {
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2, "IndexCSPG only supports L2 distance");
    FAISS_THROW_IF_NOT_MSG(num_partitions >= 1, "num_partitions must be >= 1");
    FAISS_THROW_IF_NOT_MSG(
            lambda > 0.0f && lambda < 1.0f, "lambda must be in (0, 1)");
    is_trained = true;
}

IndexCSPG::~IndexCSPG() {
    if (own_fields) {
        for (auto* p : partitions)
            delete p;
    }
}

// =========================================================================
// train / reset / reconstruct
// =========================================================================

void IndexCSPG::train(idx_t, const float*) {
    is_trained = true;
}

void IndexCSPG::reset() {
    if (own_fields) {
        for (auto* p : partitions)
            delete p;
    }
    partitions.clear();
    all_vectors.clear();
    refunction.clear();
    num_routing = 0;
    ntotal = 0;
}

void IndexCSPG::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    memcpy(recons, all_vectors.data() + key * d, sizeof(float) * d);
}

// =========================================================================
// add  –  Algorithm 2 from the paper
// =========================================================================

void IndexCSPG::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            ntotal == 0,
            "IndexCSPG: must add all vectors in a single call "
            "(incremental add is not supported)");
    FAISS_THROW_IF_NOT(n > 0);

    ntotal = n;

    // ---- Store all vectors ----
    all_vectors.resize(n * d);
    memcpy(all_vectors.data(), x, n * d * sizeof(float));

    // ---- Compute number of routing vectors ----
    num_routing = static_cast<idx_t>(n * lambda);
    if (num_routing < 1)
        num_routing = 1;
    if (num_routing >= n)
        num_routing = n - 1;

    // ---- Build refunction mapping (Algorithm 2 lines 1-4) ----
    refunction.resize(num_partitions);
    for (int p = 0; p < num_partitions; p++) {
        refunction[p].clear();
    }

    // Routing vectors (global IDs 0..num_routing-1) go to ALL partitions.
    // They get local IDs 0..num_routing-1 in every partition.
    for (idx_t id = 0; id < num_routing; id++) {
        for (int p = 0; p < num_partitions; p++) {
            refunction[p].push_back(id);
        }
    }

    // Non-routing vectors are randomly assigned to exactly one partition.
    // Matches the original CSPG code: rand() % num_partition_.
    std::mt19937 rng(1234); // fixed seed for reproducibility
    for (idx_t id = num_routing; id < n; id++) {
        int p = static_cast<int>(rng() % num_partitions);
        refunction[p].push_back(id);
    }

    // ---- Clear old partitions ----
    if (own_fields) {
        for (auto* p : partitions)
            delete p;
    }
    partitions.clear();
    partitions.resize(num_partitions, nullptr);

    // ---- Build one HNSW index per partition (Algorithm 2 line 5) ----
    for (int p = 0; p < num_partitions; p++) {
        idx_t np = static_cast<idx_t>(refunction[p].size());

        // Gather partition data (copies, since HNSW stores its own vectors)
        std::vector<float> part_data(np * d);
        for (idx_t j = 0; j < np; j++) {
            memcpy(part_data.data() + j * d,
                   all_vectors.data() + refunction[p][j] * d,
                   d * sizeof(float));
        }

        // Build HNSW
        auto* part = new IndexHNSWFlat(d, M);
        part->hnsw.efConstruction = efConstruction;
        part->add(np, part_data.data());

        partitions[p] = part;
    }
}

// =========================================================================
// search  –  Algorithm 1 from the paper
// =========================================================================

void IndexCSPG::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT_MSG(ntotal > 0, "IndexCSPG::search on empty index");
    FAISS_THROW_IF_NOT(k > 0);

    // ---- Parse parameters ----
    int cur_ef1 = ef1;
    int cur_ef2 = efSearch;
    if (params_in) {
        const auto* p = dynamic_cast<const SearchParametersCSPG*>(params_in);
        if (p) {
            cur_ef1 = p->ef1;
            cur_ef2 = p->efSearch;
        }
    }
    if (cur_ef2 < (int)k)
        cur_ef2 = (int)k;

    // Candidate tuple: (distance, partition_id, local_id)
    using Candidate = std::tuple<float, int, int>;

#pragma omp parallel
    {
        // Per-thread visited arrays (allocated once, reset per query)
        std::vector<std::vector<char>> visited(num_partitions);
        for (int p = 0; p < num_partitions; p++) {
            visited[p].resize(partitions[p]->ntotal, 0);
        }

#pragma omp for schedule(dynamic)
        for (idx_t q = 0; q < n; q++) {
            const float* query = x + q * d;
            float* res_dis = distances + q * k;
            idx_t* res_lab = labels + q * k;

            // Initialize results to empty
            for (idx_t j = 0; j < k; j++) {
                res_dis[j] = std::numeric_limits<float>::max();
                res_lab[j] = -1;
            }

            // ==============================================================
            // Stage 1: Fast approaching – find entry point in partition 0
            // ==============================================================

            const HNSW& hnsw0 = partitions[0]->hnsw;
            HNSW::storage_idx_t nearest = hnsw0.entry_point;
            if (nearest < 0)
                continue; // empty partition (shouldn't happen)

            float d_nearest = fvec_L2sqr(query, get_vec(0, nearest), d);

            // Greedy descent through upper HNSW levels in partition 0.
            // For ef1 <= 1, also descend through level 0 (= GetClosestPoint).
            // For ef1 > 1, stop at level 1 (level 0 handled by beam search).
            int stop_level = (cur_ef1 <= 1) ? 0 : 1;

            for (int level = hnsw0.max_level; level >= stop_level; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    size_t begin, end;
                    hnsw0.neighbor_range(nearest, level, &begin, &end);
                    for (size_t j = begin; j < end; j++) {
                        HNSW::storage_idx_t v = hnsw0.neighbors[j];
                        if (v < 0)
                            continue;
                        float dv = fvec_L2sqr(query, get_vec(0, v), d);
                        if (dv < d_nearest) {
                            nearest = v;
                            d_nearest = dv;
                            changed = true;
                        }
                    }
                }
            }

            // If ef1 > 1: beam search at level 0 in partition 0.
            // This matches the original CSPG's hnsw_s_[0]->Search(q, 1, ef1).
            if (cur_ef1 > 1) {
                memset(visited[0].data(), 0, visited[0].size());
                visited[0][nearest] = 1;

                std::priority_queue<Candidate> s1_max; // candidate set L
                std::priority_queue<Candidate> s1_min; // exploration queue
                s1_max.emplace(d_nearest, 0, static_cast<int>(nearest));
                s1_min.emplace(-d_nearest, 0, static_cast<int>(nearest));
                float s1_bound = d_nearest;

                while (!s1_min.empty()) {
                    auto [neg_d, gid, vid] = s1_min.top();
                    s1_min.pop();

                    if (-neg_d > s1_bound &&
                        static_cast<int>(s1_max.size()) >= cur_ef1)
                        break;

                    size_t begin, end;
                    hnsw0.neighbor_range(vid, 0, &begin, &end);
                    for (size_t j = begin; j < end; j++) {
                        HNSW::storage_idx_t nid = hnsw0.neighbors[j];
                        if (nid < 0 || visited[0][nid])
                            continue;
                        visited[0][nid] = 1;

                        float dn = fvec_L2sqr(query, get_vec(0, nid), d);
                        if (s1_max.empty() ||
                            std::get<0>(s1_max.top()) > dn ||
                            static_cast<int>(s1_max.size()) < cur_ef1) {
                            s1_max.emplace(dn, 0, static_cast<int>(nid));
                            s1_min.emplace(-dn, 0, static_cast<int>(nid));
                        }
                        while (static_cast<int>(s1_max.size()) > cur_ef1)
                            s1_max.pop();
                        if (!s1_max.empty())
                            s1_bound = std::get<0>(s1_max.top());
                    }
                }

                // Paper line 7: p ← closest vector in visited
                while (s1_max.size() > 1)
                    s1_max.pop();
                if (!s1_max.empty()) {
                    nearest = std::get<2>(s1_max.top());
                    d_nearest = std::get<0>(s1_max.top());
                }
            }

            // ==============================================================
            // Stage 2: Cross-partition expansion (Algorithm 1 lines 8-14)
            // ==============================================================

            // Paper line 8: L ← {(p,1)}, visited ← {p}
            // Reset visited for all partitions
            for (int p = 0; p < num_partitions; p++) {
                memset(visited[p].data(), 0, visited[p].size());
            }

            visited[0][nearest] = 1;

            // maxheap = candidate set L (bounded by ef2, max-heap by distance)
            // minheap = exploration queue (min-heap via negated distances)
            std::priority_queue<Candidate> maxheap;
            std::priority_queue<Candidate> minheap;
            maxheap.emplace(d_nearest, 0, static_cast<int>(nearest));
            minheap.emplace(-d_nearest, 0, static_cast<int>(nearest));
            float bound = d_nearest;

            // Paper lines 9-14: main search loop
            while (!minheap.empty()) {
                // Line 10: (r, h) ← closest vector w.r.t. q in L
                auto [neg_d, gid, vid] = minheap.top();
                minheap.pop();

                // Termination: if the closest candidate's distance exceeds
                // the bound and we have enough candidates, stop.
                if (-neg_d > bound &&
                    static_cast<int>(maxheap.size()) >= cur_ef2)
                    break;

                // Line 11: for all unvisited neighbor u of r in G_h
                const HNSW& hnsw = partitions[gid]->hnsw;
                size_t begin, end;
                hnsw.neighbor_range(vid, 0, &begin, &end);

                for (size_t j = begin; j < end; j++) {
                    HNSW::storage_idx_t nid = hnsw.neighbors[j];
                    if (nid < 0 || visited[gid][nid])
                        continue;

                    // Line 12: L ← L ∪ {(u, h)}, visited ← visited ∪ {u}
                    visited[gid][nid] = 1;
                    float dn = fvec_L2sqr(query, get_vec(gid, nid), d);

                    if (maxheap.empty() ||
                        std::get<0>(maxheap.top()) > dn ||
                        static_cast<int>(maxheap.size()) < cur_ef2) {
                        maxheap.emplace(dn, gid, static_cast<int>(nid));
                        minheap.emplace(-dn, gid, static_cast<int>(nid));

                        // Line 13: if u is a routing vector then
                        //   L ← L ∪ {(u,i) | i ∈ {1,...,m} ∧ i ≠ h}
                        if (nid < static_cast<HNSW::storage_idx_t>(
                                          num_routing)) {
                            for (int p = 0; p < num_partitions; p++) {
                                if (p == gid)
                                    continue;
                                visited[p][nid] = 1;
                                maxheap.emplace(
                                        dn, p, static_cast<int>(nid));
                                minheap.emplace(
                                        -dn, p, static_cast<int>(nid));
                            }
                        }
                    }

                    // Line 14: keep |L| = ef2
                    while (static_cast<int>(maxheap.size()) > cur_ef2)
                        maxheap.pop();
                    if (!maxheap.empty())
                        bound = std::get<0>(maxheap.top());
                }
            }

            // ==============================================================
            // Line 15: return top-k closest vectors in L
            // ==============================================================
            while (static_cast<int>(maxheap.size()) > static_cast<int>(k))
                maxheap.pop();

            // maxheap pops in descending distance order → fill from back
            int ri = static_cast<int>(maxheap.size()) - 1;
            while (!maxheap.empty()) {
                auto [dist, gid, vid] = maxheap.top();
                maxheap.pop();
                if (ri >= 0 && ri < static_cast<int>(k)) {
                    res_dis[ri] = dist;
                    res_lab[ri] = refunction[gid][vid];
                }
                ri--;
            }
        }
    }
}

} // namespace faiss
