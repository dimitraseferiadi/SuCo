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
 *     greedy descent (using FAISS's greedy_update_nearest).  If ef1 > 1,
 *     also perform beam search at level 0 using search_from_candidate_unbounded.
 *   Stage 2 – Cross-partition expansion:
 *     Starting from the closest point found in stage 1, perform beam search
 *     across ALL partitions with candidate set size ef2.  When a routing
 *     vector is encountered, it is simultaneously inserted into the candidate
 *     sets of all partitions, enabling the search to "cross" between graphs.
 *
 * All vectors are stored once in a shared IndexFlat (shared_flat).
 * Per-partition flat storage is freed after construction; search uses
 * RemappedDistanceComputer to map local IDs → global positions in
 * shared_flat, eliminating routing-vector duplication and cache misses.
 */

#include <faiss/IndexCSPG.h>
#include <faiss/IndexFlat.h>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/prefetch.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace faiss {

// =========================================================================
// RemappedDistanceComputer
// =========================================================================
//
// Wraps a DistanceComputer backed by shared_flat (all n vectors stored once)
// and remaps local partition IDs → global IDs via a switchable mapping.
// This lets us use a single contiguous vector store for all partitions,
// eliminating routing-vector duplication and cross-partition cache misses.

struct RemappedDistanceComputer : DistanceComputer {
    DistanceComputer* base; // DC from shared_flat (NOT owned)
    const idx_t* mapping;   // refunction[p].data() for current partition

    RemappedDistanceComputer(DistanceComputer* base, const idx_t* mapping)
            : base(base), mapping(mapping) {}

    void set_mapping(const idx_t* m) {
        mapping = m;
    }

    void set_query(const float* x) override {
        base->set_query(x);
    }

    float operator()(idx_t i) override {
        return (*base)(mapping[i]);
    }

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        base->distances_batch_4(
                mapping[idx0],
                mapping[idx1],
                mapping[idx2],
                mapping[idx3],
                dis0,
                dis1,
                dis2,
                dis3);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return base->symmetric_dis(mapping[i], mapping[j]);
    }

    ~RemappedDistanceComputer() override {} // base not owned
};

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
    delete shared_flat;
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
    delete shared_flat;
    shared_flat = nullptr;
    refunction.clear();
    global_to_local.clear();
    num_routing = 0;
    ntotal = 0;
}

void IndexCSPG::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    memcpy(recons, shared_flat->get_xb() + key * d, sizeof(float) * d);
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

    // ---- Compute number of routing vectors ----
    num_routing = static_cast<idx_t>(n * lambda);
    if (num_routing < 1)
        num_routing = 1;
    if (num_routing >= n)
        num_routing = n - 1;

    // ---- Build refunction mapping (Algorithm 2 lines 1-4) ----
    // Paper Definition 5 / Algorithm 2: routing vectors are RANDOMLY
    // sampled from D (not a deterministic prefix by ID), so that no
    // structural ordering of the input biases the routing set.
    refunction.assign(num_partitions, {});

    std::mt19937 rng(1234); // fixed seed for reproducibility

    // Sample num_routing IDs uniformly at random without replacement.
    std::vector<idx_t> shuffled(n);
    std::iota(shuffled.begin(), shuffled.end(), 0);
    std::shuffle(shuffled.begin(), shuffled.end(), rng);

    std::vector<idx_t> routing_ids(
            shuffled.begin(), shuffled.begin() + num_routing);
    // Sort to preserve sequential access patterns when gathering vector data.
    std::sort(routing_ids.begin(), routing_ids.end());

    // Routing vectors: local IDs 0..num_routing-1 in every partition,
    // holding the SAME sequence of sorted global IDs so that
    // refunction[p1][lid] == refunction[p2][lid] for lid < num_routing.
    for (int p = 0; p < num_partitions; p++) {
        refunction[p] = routing_ids;
        refunction[p].reserve(
                num_routing + (n - num_routing) / num_partitions + 16);
    }

    // Mark routing vectors so we can skip them when assigning the rest.
    std::vector<char> is_routing(n, 0);
    for (idx_t id : routing_ids) {
        is_routing[id] = 1;
    }

    // Non-routing vectors: randomly assign each to exactly one partition.
    for (idx_t id = 0; id < n; id++) {
        if (is_routing[id])
            continue;
        int p = static_cast<int>(rng() % num_partitions);
        refunction[p].push_back(id);
    }

    // ---- Build reverse mapping global_id -> (partition, local_id) ----
    global_to_local.resize(n, {-1, -1});
    for (int p = 0; p < num_partitions; p++) {
        for (idx_t lid = 0; lid < static_cast<idx_t>(refunction[p].size());
             lid++) {
            idx_t gid = refunction[p][lid];
            // For routing vectors, store partition 0 (any would work).
            if (global_to_local[gid].first < 0) {
                global_to_local[gid] = {p, lid};
            }
        }
    }

    // ---- Create shared flat storage (all n vectors stored once) ----
    delete shared_flat;
    shared_flat = new IndexFlat(d, METRIC_L2);
    shared_flat->add(n, x);

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

        // Gather partition data for HNSW construction.
        std::vector<float> part_data(np * d);
        for (idx_t j = 0; j < np; j++) {
            memcpy(part_data.data() + j * d,
                   x + refunction[p][j] * d,
                   d * sizeof(float));
        }

        // Build HNSW
        auto* part = new IndexHNSWFlat(d, M);
        part->hnsw.efConstruction = efConstruction;
        part->add(np, part_data.data());

        // Free per-partition vector storage – we only need the HNSW graph.
        // Search uses shared_flat via RemappedDistanceComputer.
        static_cast<IndexFlat*>(part->storage)->reset();

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

    // ---- Precompute encoding stride for MinimaxHeap ----
    // Encode (partition_id, local_id) into a single storage_idx_t:
    //   encoded = partition_id * stride + local_id
    // This lets us use FAISS's SIMD-optimized MinimaxHeap (single heap)
    // instead of dual std::priority_queue (halves heap operations).
    HNSW::storage_idx_t stride = 0;
    for (int p = 0; p < num_partitions; p++) {
        stride = std::max(
                stride,
                static_cast<HNSW::storage_idx_t>(partitions[p]->ntotal));
    }

#pragma omp parallel
    {
        // Per-thread VisitedTables (O(1) reset via generation counter)
        std::vector<VisitedTable> visited;
        visited.reserve(num_partitions);
        for (int p = 0; p < num_partitions; p++) {
            visited.emplace_back(partitions[p]->ntotal);
        }

        // Single shared DistanceComputer from shared_flat per thread,
        // wrapped in RemappedDistanceComputer with switchable mapping.
        std::unique_ptr<DistanceComputer> base_dc(
                shared_flat->get_distance_computer());
        RemappedDistanceComputer rdc(base_dc.get(), refunction[0].data());

#pragma omp for schedule(dynamic, 64)
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
                continue; // empty partition

            rdc.set_query(query);
            rdc.set_mapping(refunction[0].data());

            float d_nearest = rdc(nearest);

            for (int level = hnsw0.max_level; level >= 1; level--) {
                greedy_update_nearest(hnsw0, rdc, level, nearest, d_nearest);
            }

            // If ef1 > 1: beam search at level 0 in partition 0.
            if (cur_ef1 > 1) {
                visited[0].advance();
                HNSWStats stats_s1;
                auto top_cands = search_from_candidate_unbounded(
                        hnsw0,
                        HNSW::Node(d_nearest, nearest),
                        rdc,
                        cur_ef1,
                        &visited[0],
                        stats_s1);

                while (top_cands.size() > 1)
                    top_cands.pop();
                if (!top_cands.empty()) {
                    d_nearest = top_cands.top().first;
                    nearest = top_cands.top().second;
                }
            }

            // ==============================================================
            // Stage 2: Cross-partition expansion (Algorithm 1 lines 8-14)
            // Uses direct fvec_L2sqr_batch_4 calls (no virtual dispatch)
            // with vector-data prefetching for cache-friendly random access.
            //
            // Two heaps are maintained:
            //   * candidates (MinimaxHeap, cap = ef2 * m): the BFS frontier,
            //     oversized so that cross-partition expansion (up to m copies
            //     of each routing vector) does not crowd out good candidates.
            //   * top_results (std::vector as a binary max-heap, cap = ef2):
            //     the ef2 best (distance, partition, local_id) triples seen.
            //     Its max supplies the early-termination threshold AND
            //     provides the final result set (per paper line 15:
            //     top-k comes from the "visited" pool, not the candidate queue).
            // ==============================================================

            const float* xb = shared_flat->get_xb();

            for (int p = 0; p < num_partitions; p++) {
                visited[p].advance();
            }
            visited[0].set(nearest);

            // Candidates heap — oversized for cross-partition expansion.
            HNSW::MinimaxHeap candidates(cur_ef2 * num_partitions);

            // top-ef2 results: max-heap on distance.
            struct TopResult {
                float dist;
                int32_t gid;
                int32_t lid;
                bool operator<(const TopResult& o) const {
                    return dist < o.dist; // max-heap on dist
                }
            };
            std::vector<TopResult> top_results;
            top_results.reserve(cur_ef2 + 1);

            auto try_add_result = [&](float dist, int gid, int lid) {
                if (static_cast<int>(top_results.size()) < cur_ef2) {
                    top_results.push_back({dist, gid, lid});
                    std::push_heap(top_results.begin(), top_results.end());
                } else if (dist < top_results.front().dist) {
                    std::pop_heap(top_results.begin(), top_results.end());
                    top_results.back() = {dist, gid, lid};
                    std::push_heap(top_results.begin(), top_results.end());
                }
            };

            const int num_routing_int = static_cast<int>(num_routing);

            // Seed.
            try_add_result(d_nearest, 0, static_cast<int>(nearest));
            {
                HNSW::storage_idx_t enc_seed =
                        0 * stride +
                        static_cast<HNSW::storage_idx_t>(nearest);
                candidates.push(enc_seed, d_nearest);

                // Fix 4: if the Stage 1 result is itself a routing vector,
                // broadcast the seed to every partition so Stage 2 can
                // enter each partition's graph directly (no need to walk
                // to a gateway via partition 0 first).
                if (static_cast<int>(nearest) < num_routing_int) {
                    for (int p = 1; p < num_partitions; p++) {
                        visited[p].set(static_cast<int>(nearest));
                        HNSW::storage_idx_t enc_p =
                                static_cast<HNSW::storage_idx_t>(p) * stride +
                                static_cast<HNSW::storage_idx_t>(nearest);
                        candidates.push(enc_p, d_nearest);
                    }
                }
            }

            while (candidates.size() > 0) {
                float cur_d;
                HNSW::storage_idx_t enc = candidates.pop_min(&cur_d);
                if (enc < 0)
                    break;

                // Fix 2: early termination. Once top_results is saturated,
                // the candidate we are about to expand cannot improve the
                // top-ef2 set if its own distance already exceeds the
                // worst kept result — standard HNSW termination, with the
                // threshold provided by an explicit top-ef2 accumulator
                // (candidates.max() is no longer a faithful proxy now
                //  that candidates is oversized by a factor of m).
                if (static_cast<int>(top_results.size()) >= cur_ef2 &&
                    cur_d > top_results.front().dist) {
                    break;
                }

                // Decode (partition_id, local_id)
                int cur_gid = static_cast<int>(enc / stride);
                int vid = static_cast<int>(enc % stride);

                const HNSW& hnsw = partitions[cur_gid]->hnsw;
                const idx_t* map = refunction[cur_gid].data();
                size_t begin, end;
                hnsw.neighbor_range(vid, 0, &begin, &end);

                // Acceptance predicate: push into candidates if the
                // neighbor could still be useful to expand. We gate on
                // top_results.max() (the true top-ef2 threshold) rather
                // than candidates.max().
                auto process_neighbor = [&](int nid, float dn) {
                    bool tr_full =
                            static_cast<int>(top_results.size()) >= cur_ef2;
                    if (tr_full && dn >= top_results.front().dist) {
                        // Neither a candidate worth expanding nor a
                        // result worth keeping.
                        return;
                    }

                    // Add to top-ef2 results (single add per neighbor —
                    // same routing vector reached via a different gid
                    // in a later iteration may add again; deduped at
                    // final extraction).
                    try_add_result(dn, cur_gid, nid);

                    HNSW::storage_idx_t enc_n =
                            static_cast<HNSW::storage_idx_t>(cur_gid) *
                                    stride +
                            nid;
                    candidates.push(enc_n, dn);

                    // Cross-partition expansion for routing vectors.
                    if (nid < num_routing_int) {
                        for (int p = 0; p < num_partitions; p++) {
                            if (p == cur_gid)
                                continue;
                            visited[p].set(nid);
                            HNSW::storage_idx_t enc_p =
                                    static_cast<HNSW::storage_idx_t>(p) *
                                            stride +
                                    nid;
                            candidates.push(enc_p, dn);
                        }
                    }
                };

                // Pass 1: prefetch visited table entries for valid neighbors.
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    visited[cur_gid].prefetch(v1);
                    jmax++;
                }

                // Pass 2: collect unvisited neighbors + prefetch their
                // vector data from shared_flat (hide DRAM latency).
                int n_unvis = 0;
                int unvis[128]; // max 2*M neighbors at level 0 (M <= 64)
                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (visited[cur_gid].set(v1)) {
                        unvis[n_unvis++] = v1;
                        prefetch_L2(xb + map[v1] * d);
                    }
                }

                // Pass 3: batch-4 distance computation.
                int i4;
                for (i4 = 0; i4 + 4 <= n_unvis; i4 += 4) {
                    float dis[4];
                    fvec_L2sqr_batch_4(
                            query,
                            xb + map[unvis[i4 + 0]] * d,
                            xb + map[unvis[i4 + 1]] * d,
                            xb + map[unvis[i4 + 2]] * d,
                            xb + map[unvis[i4 + 3]] * d,
                            d,
                            dis[0],
                            dis[1],
                            dis[2],
                            dis[3]);
                    for (int id4 = 0; id4 < 4; id4++) {
                        process_neighbor(unvis[i4 + id4], dis[id4]);
                    }
                }
                // Handle remaining neighbors (< 4).
                for (; i4 < n_unvis; i4++) {
                    float dis = fvec_L2sqr(
                            query, xb + map[unvis[i4]] * d, d);
                    process_neighbor(unvis[i4], dis);
                }
            }

            // ==============================================================
            // Line 15 (paper): return top-k from the pool of visited vectors,
            // not from the candidate queue. We sort top_results ascending
            // and dedup by global_id (routing vectors may appear under
            // multiple (gid, lid) entries with the same distance).
            // ==============================================================
            std::sort(top_results.begin(),
                      top_results.end(),
                      [](const TopResult& a, const TopResult& b) {
                          return a.dist < b.dist;
                      });
            {
                int ri = 0;
                for (const auto& r : top_results) {
                    if (ri >= static_cast<int>(k))
                        break;
                    idx_t global_id = refunction[r.gid][r.lid];
                    bool dup = false;
                    for (int s = 0; s < ri; s++) {
                        if (res_lab[s] == global_id) {
                            dup = true;
                            break;
                        }
                    }
                    if (!dup) {
                        res_dis[ri] = r.dist;
                        res_lab[ri] = global_id;
                        ri++;
                    }
                }
            }
        }
    }
}

} // namespace faiss
