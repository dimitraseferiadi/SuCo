/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

/**
 * Implementation of IndexSHG (HEDS algorithm from the SHG-Index paper).
 *
 * Key design notes
 * ----------------
 * Level numbering: FAISS HNSW uses 0-indexed levels where level 0 is the
 * BASE layer (all ntotal vectors, full d dimensions). Level max_level is
 * the TOP. HNSW::levels[i] stores the *count* of levels node i appears on,
 * so a node with levels[i]==3 exists at HNSW levels 0, 1, 2.
 *
 * Compression: The compression hierarchy is INDEPENDENT of the HNSW graph
 * levels. maxFixLevel_ compression levels are computed from (d, eta=2) by
 * repeatedly dividing: level l has dim ceil(d/eta^l), stopping when
 * dim/eta < eta.  HNSW level l uses compression level min(l, maxFixLevel_).
 *
 * Paper (Section 3.1) uses eta=2.  maxFixLevel_ is computed as:
 *   while(dim/eta >= eta) { maxFixLevel_++; dim = ceil(dim/eta); }
 *
 * Shortcut (Section 4.2):
 *   For each node o in the graph, at each HNSW level x >= 2:
 *     - find nearest graph neighbour at level x, compute distance disx
 *     - check density condition to find how many levels can be skipped
 *     - store (disx, skip_count) into a sorted map (PGM-index in original)
 *   At search time: lower_bound(dist) returns skip count.
 *
 * Lower-bound pruning (Theorem 1):
 *   If dis_compressed * eta^(level_diff) > current best, prune the candidate.
 *
 * FAISS optimizations used:
 *   - Forked greedy_update_nearest() with inline compressed distances + caching
 *   - Forked search_from_candidates() with MinimaxHeap, batch-4 distances,
 *     and integrated SHG pruning (cross-level LB + on-the-fly compressed)
 *   - HeapBlockResultHandler for result collection
 *   - Single OMP region following hnsw_search() pattern
 *   - fvec_L2sqr for all distance computations
 */

#include <faiss/IndexSHG.h>
#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <vector>

namespace faiss {

// ---------------------------------------------------------------------------
// SHGDistanceComputer — level-aware DistanceComputer for construction
// ---------------------------------------------------------------------------
//
// At upper HNSW levels (> 0), distances are computed using the compressed
// vector representations stored in IndexSHG::compressed_vecs.
// At level 0, distances delegate to the exact DistanceComputer.
// This matches Algorithm 2 line 7 from the SHG paper.

namespace {

struct SHGDistanceComputer : DistanceComputer {
    const IndexSHG& shg;
    std::unique_ptr<DistanceComputer> exact_dis;
    std::vector<float> query_rep; // compressed query (levels 1..maxFixLevel_)
    int current_level = 0;

    SHGDistanceComputer(const IndexSHG& shg_, DistanceComputer* exact)
            : shg(shg_),
              exact_dis(exact),
              query_rep(shg_.data_rep_size_, 0.0f) {}

    void set_query(const float* x) override {
        exact_dis->set_query(x);
        if (shg.maxFixLevel_ <= 0) return;

        // Build compressed query representation
        std::vector<float> full_rep;
        full_rep.reserve(shg.data_rep_size_ + shg.d);
        full_rep.insert(full_rep.end(), x, x + shg.d);

        int prev_pos = 0;
        for (int cl = 1; cl <= shg.maxFixLevel_; ++cl) {
            int prev_size = shg.dim_at_level[cl - 1];
            for (int i = 0; i < prev_size; i += shg.eta) {
                float sum = 0.0f;
                int end = std::min(i + shg.eta, prev_size);
                for (int j = i; j < end; ++j)
                    sum += full_rep[prev_pos + j];
                full_rep.push_back(sum / (float)(end - i));
            }
            prev_pos += prev_size;
        }
        std::copy(full_rep.begin() + shg.d, full_rep.end(), query_rep.begin());
    }

    float operator()(idx_t i) override {
        if (current_level <= 0 || shg.maxFixLevel_ <= 0) {
            return (*exact_dis)(i);
        }
        return shg.get_dis_by_level_q(query_rep, i, current_level);
    }

    // Phase 4: Optimized batch-4 for compressed distances
    void distances_batch_4(
            const idx_t i0,
            const idx_t i1,
            const idx_t i2,
            const idx_t i3,
            float& d0,
            float& d1,
            float& d2,
            float& d3) override {
        if (current_level <= 0 || shg.maxFixLevel_ <= 0) {
            exact_dis->distances_batch_4(i0, i1, i2, i3, d0, d1, d2, d3);
            return;
        }
        // Inline compressed distance computation — avoids 4 virtual dispatches
        int cl = std::min(current_level, shg.maxFixLevel_);
        int cdim = shg.dim_at_level[cl];
        const float* q = query_rep.data() + shg.offset_at_level[cl];
        d0 = fvec_L2sqr(q, shg.get_compressed_data(i0, cl), (size_t)cdim);
        d1 = fvec_L2sqr(q, shg.get_compressed_data(i1, cl), (size_t)cdim);
        d2 = fvec_L2sqr(q, shg.get_compressed_data(i2, cl), (size_t)cdim);
        d3 = fvec_L2sqr(q, shg.get_compressed_data(i3, cl), (size_t)cdim);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        if (current_level <= 0 || shg.maxFixLevel_ <= 0) {
            return exact_dis->symmetric_dis(i, j);
        }
        return shg.get_dis_by_level(i, j, current_level);
    }
};

// ---------------------------------------------------------------------------
// greedy_update_nearest_shg — forked from FAISS greedy_update_nearest()
// ---------------------------------------------------------------------------
// FAISS's greedy_update_nearest (HNSW.cpp:1100-1173) with batch-4 pattern,
// modified to:
//   1. Compute compressed distances inline (no DistanceComputer indirection)
//   2. Cache every computed distance into DisCache for cross-level LB pruning

static void greedy_update_nearest_shg(
        const HNSW& hnsw,
        const IndexSHG& shg,
        const std::vector<float>& query_rep,
        int level,
        HNSW::storage_idx_t& nearest,
        float& d_nearest) {
    int cl = std::min(level, shg.maxFixLevel_);
    int cdim = shg.dim_at_level[cl];
    const float* q_comp = query_rep.data() + shg.offset_at_level[cl];

    for (;;) {
        HNSW::storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);

        // Batch-4 pattern (from FAISS greedy_update_nearest)
        int n_buffered = 0;
        HNSW::storage_idx_t buffered_ids[4];

        for (size_t j = begin; j < end; j++) {
            HNSW::storage_idx_t v = hnsw.neighbors[j];
            if (v < 0) break;

            buffered_ids[n_buffered++] = v;

            if (n_buffered == 4) {
                // Inline compressed distance for 4 nodes
                for (int b = 0; b < 4; b++) {
                    const float* c =
                            shg.get_compressed_data(buffered_ids[b], cl);
                    float dis = fvec_L2sqr(q_comp, c, (size_t)cdim);
                    if (dis < d_nearest) {
                        nearest = buffered_ids[b];
                        d_nearest = dis;
                    }
                }
                n_buffered = 0;
            }
        }

        // Process leftovers
        for (int b = 0; b < n_buffered; b++) {
            const float* c = shg.get_compressed_data(buffered_ids[b], cl);
            float dis = fvec_L2sqr(q_comp, c, (size_t)cdim);
            if (dis < d_nearest) {
                nearest = buffered_ids[b];
                d_nearest = dis;
            }
        }

        if (nearest == prev_nearest) {
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// search_from_candidates_shg — forked from FAISS search_from_candidates()
// ---------------------------------------------------------------------------
// FAISS's search_from_candidates (HNSW.cpp:618-752) with MinimaxHeap,
// batch-4 exact distances, and ResultHandler, modified to add:
//   1. FAISS-compatible stopping: count_below(d0) >= efSearch, plus
//      nstep > efSearch fallback — matches HNSW.cpp:657-739
//   2. Single-loop with inline batch-4 (no intermediate survivors vector)
//   3. Compressed LB pruning (Algorithm 3, line 14): compute compressed
//      distance at level L (most compressed = cheapest), scale by eta^L
//      for squared L2, skip candidate if lower bound > dis_k
//   4. Prefetch pass for visited table (matches FAISS HNSW.cpp:673-682)

static void search_from_candidates_shg(
        const HNSW& hnsw,
        const IndexSHG& shg,
        DistanceComputer& qdis,
        ResultHandler& res,
        HNSW::MinimaxHeap& candidates,
        VisitedTable& vt,
        const std::vector<float>& query_rep,
        bool use_lb_pruning,
        bool do_dis_check,
        int efSearch) {
    using storage_idx_t = HNSW::storage_idx_t;

    float threshold = res.threshold;

    // Pre-compute compressed query pointer for LB pruning (Algorithm 3 line 14)
    // Uses level maxFixLevel_ (most compressed = smallest dimension = cheapest)
    // Scaling: eta^(L/2) where L = maxFixLevel_ (from Theorem 1)
    const float* q_lb = nullptr;
    int lb_cdim = 0;
    float lb_scale = 1.0f;
    int L = shg.maxFixLevel_;
    if (use_lb_pruning && L > 0) {
        q_lb = query_rep.data() + shg.offset_at_level[L];
        lb_cdim = shg.dim_at_level[L];
        // Theorem 1 bound is for Euclidean distance: dis_0 >= dis_L * eta^(L/2).
        // FAISS uses squared L2 throughout, so squaring both sides:
        //   dis_0^2 >= dis_L^2 * eta^L
        // The scale for squared distances is eta^L, not eta^(L/2).
        lb_scale = std::pow((float)shg.eta, (float)L);
    }

    // Add initial candidates to results (from FAISS search_from_candidates)
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float dd = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (dd < threshold) {
            if (res.add_result(dd, v1)) {
                threshold = res.threshold;
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0;
        int v0 = candidates.pop_min(&d0);

        // FAISS-compatible stopping condition (HNSW.cpp:657-666):
        // When check_relative_distance is true, stop when enough
        // candidates in the heap are better than the current minimum.
        if (do_dis_check) {
            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, 0, &begin, &end);

        // Prefetch pass for visited table (matches FAISS HNSW.cpp:673-682)
        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) break;
            vt.prefetch(v1);
            jmax += 1;
        }

        threshold = res.threshold;

        // Single-loop with inline batch-4 and LB pruning
        // (merged from FAISS's single-loop pattern + SHG LB filter)
        int n_buffered = 0;
        storage_idx_t buffered_ids[4];

        auto add_to_heap = [&](const size_t idx, const float dis) {
            if (dis < threshold) {
                if (res.add_result(dis, idx)) {
                    threshold = res.threshold;
                }
            }
            candidates.push(idx, dis);
        };

        for (size_t j = begin; j < jmax; j++) {
            storage_idx_t cand = hnsw.neighbors[j];

            // Visited check (matches FAISS pattern)
            if (!vt.set(cand)) continue;

            // Algorithm 3 line 14 — compressed LB pruning:
            // lowerbound = eta^L * dis_L^2  (squared L2 form of Theorem 1)
            // If lowerbound > dis_k, skip this candidate
            if (q_lb != nullptr) {
                const float* c = shg.get_compressed_data(cand, L);
                float approx = fvec_L2sqr(q_lb, c, (size_t)lb_cdim);
                float lowerbound = approx * lb_scale;
                if (lowerbound > threshold) {
                    continue;
                }
            }

            buffered_ids[n_buffered++] = cand;

            if (n_buffered == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        buffered_ids[0],
                        buffered_ids[1],
                        buffered_ids[2],
                        buffered_ids[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);
                for (int b = 0; b < 4; b++) {
                    add_to_heap(buffered_ids[b], dis[b]);
                }
                n_buffered = 0;
            }
        }

        // Process remaining buffered candidates
        for (int b = 0; b < n_buffered; b++) {
            float dis = qdis(buffered_ids[b]);
            add_to_heap(buffered_ids[b], dis);
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// ShortcutMap serialization
// ---------------------------------------------------------------------------

void ShortcutMap::write(FILE* f) const {
    int n = (int)entries.size();
    fwrite(&n, sizeof(int), 1, f);
    for (auto& [dist, skip] : entries) {
        fwrite(&dist, sizeof(float), 1, f);
        fwrite(&skip, sizeof(int), 1, f);
    }
}

void ShortcutMap::read(FILE* f) {
    entries.clear();
    int n;
    size_t ret = fread(&n, sizeof(int), 1, f);
    FAISS_THROW_IF_NOT(ret == 1);
    for (int i = 0; i < n; ++i) {
        float dist;
        int skip;
        ret = fread(&dist, sizeof(float), 1, f);
        FAISS_THROW_IF_NOT(ret == 1);
        ret = fread(&skip, sizeof(int), 1, f);
        FAISS_THROW_IF_NOT(ret == 1);
        entries[dist] = skip;
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

IndexSHG::IndexSHG(int d, int M, MetricType metric)
        : IndexHNSWFlat(d, M, metric) {
    if (d > 0) {
        compute_compression_params();
    }
}

// ---------------------------------------------------------------------------
// add — Algorithm 2: build HNSW graph with compressed upper-level distances
// ---------------------------------------------------------------------------

void IndexSHG::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "IndexSHG requires flat storage");
    FAISS_THROW_IF_NOT(is_trained);

    if (dim_at_level.empty() && d > 0) {
        compute_compression_params();
    }

    idx_t n0 = ntotal;

    // Step 1: store raw vectors
    storage->add(n, x);
    ntotal = storage->ntotal;

    // Step 2: grow compressed storage and compress all new vectors
    if (maxFixLevel_ > 0 && data_rep_size_ > 0) {
        compressed_vecs.resize((size_t)ntotal * data_rep_size_, 0.0f);
        for (idx_t i = n0; i < ntotal; ++i) {
            compress_node(i);
        }
    }

    // Step 3: build HNSW graph with compressed distances at upper levels
    // This mirrors hnsw_add_vertices() from IndexHNSW.cpp but uses
    // SHGDistanceComputer which switches to compressed distances at
    // levels > 0 (Algorithm 2 line 7).

    size_t ntot = n0 + n;
    HNSW& hns = hnsw;

    int max_level = hns.prepare_level_tab(n, hns.levels.size() == ntot);

    if (verbose) {
        printf("IndexSHG::add: adding %" PRId64
               " elements on top of %" PRId64 " (maxFixLevel_=%d)\n",
               n, n0, maxFixLevel_);
    }

    if (n == 0) return;

    // Locks for concurrent graph modification
    std::vector<omp_lock_t> locks(ntot);
    for (size_t i = 0; i < ntot; ++i) {
        omp_init_lock(&locks[i]);
    }

    // Build histogram and sort by level (highest first)
    std::vector<int> hist;
    std::vector<int> order(n);
    {
        for (idx_t i = 0; i < n; ++i) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hns.levels[pt_id] - 1;
            while (pt_level >= (int)hist.size()) {
                hist.push_back(0);
            }
            hist[pt_level]++;
        }
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < (int)hist.size() - 1; ++i) {
            offsets[i + 1] = offsets[i] + hist[i];
        }
        for (idx_t i = 0; i < n; ++i) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hns.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    // Process level by level, highest first
    {
        RandomGenerator rng2(789);
        int i1 = n;

        for (int pt_level = (int)hist.size() - 1; pt_level >= 0; --pt_level) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("  Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // Random permutation within this level
            for (int j = i0; j < i1; ++j) {
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);
            }

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntot, hns.use_visited_hashset);
                // Create level-aware distance computer
                SHGDistanceComputer dis(
                        *this, storage->get_distance_computer());

#pragma omp for schedule(static)
                for (int i = i0; i < i1; ++i) {
                    storage_idx_t pt_id = order[i];
                    dis.set_query(x + (pt_id - n0) * d);

                    // --- add_with_locks logic, with level-aware distances ---

                    storage_idx_t nearest;
#pragma omp critical
                    {
                        nearest = hns.entry_point;
                        if (nearest == -1) {
                            hns.max_level = pt_level;
                            hns.entry_point = pt_id;
                        }
                    }

                    if (nearest < 0) continue;

                    omp_set_lock(&locks[pt_id]);

                    int level = hns.max_level;
                    float d_nearest;

                    // Phase 1: navigate upper levels to find entry point
                    for (; level > pt_level; --level) {
                        dis.current_level = level;
                        d_nearest = dis(nearest);
                        greedy_update_nearest(
                                hns, dis, level, nearest, d_nearest);
                    }

                    // Phase 2: insert links at all levels pt_level..0
                    for (; level >= 0; --level) {
                        dis.current_level = level;
                        d_nearest = dis(nearest);
                        hns.add_links_starting_from(
                                dis,
                                pt_id,
                                nearest,
                                d_nearest,
                                level,
                                locks.data(),
                                vt,
                                keep_max_size_level0 && (level == 0));
                    }

                    omp_unset_lock(&locks[pt_id]);

                    if (pt_level > hns.max_level) {
                        hns.max_level = pt_level;
                        hns.entry_point = pt_id;
                    }
                }
            }
            i1 = i0;
        }
    }

    for (size_t i = 0; i < ntot; ++i) {
        omp_destroy_lock(&locks[i]);
    }

    if (verbose) {
        printf("IndexSHG::add: done\n");
    }
}

// ---------------------------------------------------------------------------
// compute_compression_params
// ---------------------------------------------------------------------------

void IndexSHG::compute_compression_params() {
    // Match original: while(dim/k_ >= k_) { maxFixLevel_++; dim=ceil(dim/k_); }
    maxFixLevel_ = 0;
    data_rep_size_ = 0;
    int dim = d;
    while (dim / eta >= eta) {
        maxFixLevel_++;
        dim = (int)std::ceil((float)dim / (float)eta);
        data_rep_size_ += dim;
    }

    // Build dim_at_level table (level 0 = full d)
    dim_at_level.resize(maxFixLevel_ + 1);
    dim_at_level[0] = d;
    int cur = d;
    for (int l = 1; l <= maxFixLevel_; ++l) {
        cur = (int)std::ceil((float)cur / (float)eta);
        dim_at_level[l] = cur;
    }

    // Build offset table for per-node compressed storage
    // offset_at_level[l] = sum of dim_at_level[i] for i in 1..l-1
    offset_at_level.resize(maxFixLevel_ + 1, 0);
    size_t off = 0;
    for (int l = 1; l <= maxFixLevel_; ++l) {
        offset_at_level[l] = off;
        off += dim_at_level[l];
    }
}

// ---------------------------------------------------------------------------
// get_dim_at_level
// ---------------------------------------------------------------------------

int IndexSHG::get_dim_at_level(int l) const {
    if (l <= 0) return d;
    if (l <= maxFixLevel_) return dim_at_level[l];
    // For levels beyond maxFixLevel_, compute dimension progressively
    int cur = dim_at_level[maxFixLevel_];
    for (int i = maxFixLevel_ + 1; i <= l; ++i) {
        cur = (int)std::ceil((float)cur / (float)eta);
    }
    return cur;
}

// ---------------------------------------------------------------------------
// compress_vector (static)
// ---------------------------------------------------------------------------

/*static*/
void IndexSHG::compress_vector(
        const float* vec,
        int d_in,
        int l,
        int eta_in,
        float* out) {
    if (l == 0) {
        std::copy(vec, vec + d_in, out);
        return;
    }

    // Progressive mean aggregation: apply l times.
    std::vector<float> buf(vec, vec + d_in);
    int cur_d = d_in;

    for (int pass = 0; pass < l; ++pass) {
        int new_d = (int)std::ceil((float)cur_d / (float)eta_in);
        std::vector<float> tmp(new_d, 0.0f);
        for (int j = 0; j < cur_d; j += eta_in) {
            int end = std::min(j + eta_in, cur_d);
            float sum = 0.0f;
            for (int k = j; k < end; ++k) {
                sum += buf[k];
            }
            tmp[j / eta_in] = sum / (float)(end - j);
        }
        buf = std::move(tmp);
        cur_d = new_d;
    }

    std::copy(buf.begin(), buf.end(), out);
}

// ---------------------------------------------------------------------------
// compressed_l2sqr (static) - returns SQUARED L2 distance
// ---------------------------------------------------------------------------

/*static*/
float IndexSHG::compressed_l2sqr(
        const float* __restrict__ a,
        const float* __restrict__ b,
        int dim) {
    // Delegate to FAISS's SIMD-optimized L2 squared distance.
    return fvec_L2sqr(a, b, (size_t)dim);
}

// ---------------------------------------------------------------------------
// get_compressed_data
// ---------------------------------------------------------------------------

const float* IndexSHG::get_compressed_data(
        idx_t node_id,
        int comp_level) const {
    if (comp_level <= 0) return nullptr;
    int cl = std::min(comp_level, maxFixLevel_);
    return compressed_vecs.data()
            + (size_t)node_id * data_rep_size_
            + offset_at_level[cl];
}

// ---------------------------------------------------------------------------
// compressed_dis_at_level - shared distance for two compressed data pointers
// ---------------------------------------------------------------------------

float IndexSHG::compressed_dis_at_level(
        const float* a_data,
        const float* b_data,
        int hnsw_level) const {
    int cl = std::min(hnsw_level, maxFixLevel_);
    return compressed_l2sqr(a_data, b_data, dim_at_level[cl]);
}

// ---------------------------------------------------------------------------
// get_dis_by_level - squared L2 between two nodes at HNSW level
// ---------------------------------------------------------------------------

float IndexSHG::get_dis_by_level(
        idx_t id1,
        idx_t id2,
        int hnsw_level) const {
    if (hnsw_level == 0) {
        const auto* flat = dynamic_cast<const IndexFlat*>(storage);
        const float* v1 = flat->get_xb() + (size_t)id1 * d;
        const float* v2 = flat->get_xb() + (size_t)id2 * d;
        return fvec_L2sqr(v1, v2, (size_t)d);
    }
    int cl = std::min(hnsw_level, maxFixLevel_);
    return compressed_dis_at_level(
            get_compressed_data(id1, cl),
            get_compressed_data(id2, cl),
            hnsw_level);
}

// ---------------------------------------------------------------------------
// get_dis_by_level_q - squared L2 between query rep and node at HNSW level
// ---------------------------------------------------------------------------

float IndexSHG::get_dis_by_level_q(
        const std::vector<float>& query_rep,
        idx_t node_id,
        int hnsw_level) const {
    FAISS_THROW_IF_NOT_MSG(
            hnsw_level > 0,
            "get_dis_by_level_q: level 0 requires full query vector, "
            "use DistanceComputer instead");
    int cl = std::min(hnsw_level, maxFixLevel_);
    const float* q = query_rep.data() + offset_at_level[cl];
    return compressed_dis_at_level(
            q, get_compressed_data(node_id, cl), hnsw_level);
}

// ---------------------------------------------------------------------------
// build_all_compressed
// ---------------------------------------------------------------------------

void IndexSHG::build_all_compressed() {
    compressed_vecs.resize((size_t)ntotal * data_rep_size_, 0.0f);
    for (idx_t i = 0; i < ntotal; ++i) {
        compress_node(i);
    }
}

// ---------------------------------------------------------------------------
// compress_node - build compressed representation for a single node
// ---------------------------------------------------------------------------

void IndexSHG::compress_node(idx_t node_id) {
    std::vector<float> full_rep;
    full_rep.reserve(data_rep_size_ + d);

    // Start with level-0 (full) data — direct pointer access
    const auto* flat = dynamic_cast<const IndexFlat*>(storage);
    const float* v0 = flat->get_xb() + (size_t)node_id * d;
    full_rep.insert(full_rep.end(), v0, v0 + d);

    int previous_level_pos = 0;

    for (int cur_lev = 1; cur_lev <= maxFixLevel_; ++cur_lev) {
        int previous_level_size = dim_at_level[cur_lev - 1];

        for (int i = 0; i < previous_level_size; i += eta) {
            float sum = 0.0f;
            if (i + eta > previous_level_size) {
                for (int j = i; j < previous_level_size; ++j)
                    sum += full_rep[previous_level_pos + j];
                full_rep.push_back(sum / (float)(previous_level_size - i));
            } else {
                for (int j = i; j < i + eta; ++j)
                    sum += full_rep[previous_level_pos + j];
                full_rep.push_back(sum / (float)eta);
            }
        }

        previous_level_pos += previous_level_size;
    }

    // Remove level-0 data (first d elements), keep only compressed levels
    float* dst = compressed_vecs.data() + (size_t)node_id * data_rep_size_;
    std::copy(
            full_rep.begin() + d,
            full_rep.end(),
            dst);
}

// ---------------------------------------------------------------------------
// get_nearest_by_level
// ---------------------------------------------------------------------------

std::pair<float, idx_t> IndexSHG::get_nearest_by_level(
        idx_t node_id,
        int hnsw_level) const {
    const HNSW& hns = hnsw;

    using NodeDist = std::pair<float, idx_t>;

    std::priority_queue<NodeDist, std::vector<NodeDist>> top_candidates;
    std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>>
            candidateSet;

    VisitedTable visited(ntotal);

    float dist_ep = get_dis_by_level(node_id, node_id, 0); // 0 distance
    top_candidates.push({dist_ep, node_id});
    candidateSet.push({dist_ep, node_id});
    float lowerBound = dist_ep;
    visited.set(node_id);

    int ef_c = hns.efConstruction;

    while (!candidateSet.empty()) {
        auto curr = candidateSet.top();
        if (curr.first > lowerBound && (int)top_candidates.size() == ef_c) {
            break;
        }
        candidateSet.pop();

        idx_t curNode = curr.second;

        size_t nb_begin, nb_end;
        hns.neighbor_range(curNode, hnsw_level, &nb_begin, &nb_end);

        for (size_t nb = nb_begin; nb < nb_end; ++nb) {
            storage_idx_t cand = hns.neighbors[nb];
            if (cand < 0) break;

            if (visited.get(cand)) continue;
            visited.set(cand);

            // Compressed distance at hnsw_level — matches paper's intent of
            // finding the NN using the same distance metric as traversal
            float dist1 = get_dis_by_level(node_id, cand, hnsw_level);

            if ((int)top_candidates.size() < ef_c ||
                    lowerBound > dist1) {
                candidateSet.push({dist1, cand});
                top_candidates.push({dist1, cand});

                if ((int)top_candidates.size() > ef_c) {
                    top_candidates.pop();
                }
                if (!top_candidates.empty()) {
                    lowerBound = top_candidates.top().first;
                }
            }
        }
    }

    // Keep top-2 (original: while(top_candidates.size()>2) pop())
    while ((int)top_candidates.size() > 2) {
        top_candidates.pop();
    }

    if (top_candidates.empty()) {
        return {-1.0f, -1};
    }

    return top_candidates.top();
}

// ---------------------------------------------------------------------------
// build_shortcuts_density (Section 4.2 - Lemma 2)
// ---------------------------------------------------------------------------

void IndexSHG::build_shortcuts_density() {
    const HNSW& hns = hnsw;
    int max_l = hns.max_level;

    // Count vectors per HNSW level
    std::vector<int> levelCounts(max_l + 1, 0);
    for (idx_t i = 0; i < ntotal; ++i) {
        int node_levels = hns.levels[i];
        for (int l = 0; l < node_levels && l <= max_l; ++l) {
            levelCounts[l]++;
        }
    }

    // Priority queue: max-heap by negative distance
    using DistSkip = std::pair<float, int>;
    auto cmp = [](const DistSkip& a, const DistSkip& b) {
        return a.first < b.first;
    };
    std::priority_queue<DistSkip, std::vector<DistSkip>, decltype(cmp)>
            density_skipLevels(cmp);

    for (idx_t i = 0; i < ntotal; ++i) {
        int point_level = hns.levels[i] - 1;
        if (point_level < 2) continue;

        // Algorithm 2 line 11: search NN at each level ONCE and cache.
        // Avoids O(L^2) beam searches per node — reduces to O(L).
        // nn_cache[l] = {squared_distance_to_NN, NN_node_id} at HNSW level l.
        std::vector<std::pair<float, idx_t>> nn_cache(point_level + 1);
        for (int l = 0; l <= point_level; ++l) {
            nn_cache[l] = get_nearest_by_level(i, l);
        }

        for (int cur_level = point_level; cur_level > 1; cur_level--) {
            float disx = nn_cache[cur_level].first;
            idx_t nearest = nn_cache[cur_level].second;
            if (nearest < 0 || disx < 0) continue;

            // Algorithm 2 lines 13-18: iterate y from bottom (0) upward
            // to cur_level-1. The FIRST y satisfying Lemma 2 gives the
            // maximum skip h = cur_level - y. This matches the paper's
            // bottom-up search: try the most aggressive skip first.
            int best_skip = 0;
            for (int y = 0; y < cur_level; ++y) {
                float disy = nn_cache[y].first;
                idx_t nearest_y = nn_cache[y].second;
                if (nearest_y < 0 || disy < 0) continue;

                int d_x = dim_at_level[std::min(cur_level, maxFixLevel_)];
                int d_y = dim_at_level[std::min(y, maxFixLevel_)];

                float n_x = (float)levelCounts[cur_level];
                float n_y = (float)levelCounts[y];

                // Lemma 2, Eq. (15) — computed entirely in log-space to
                // avoid overflow/underflow with high-dimensional exponents.
                //
                // Paper (Euclidean): disx^{d_x} <= (n_y/n_x)*(V_dy/V_dx)*disy^{d_y}
                // Our distances are squared L2, so Euclidean = sqrt(sqr):
                //   sqr_x^{d_x/2} <= (n_y/n_x)*(V_dy/V_dx)*sqr_y^{d_y/2}
                //
                // In log-space:
                //   (d_x/2)*log(sqr_x) <= log(n_y/n_x) + log(V_dy/V_dx) + (d_y/2)*log(sqr_y)
                //
                // V_d = pi^{d/2} / Gamma(d/2+1), so:
                //   log(V_dy/V_dx) = (d_y-d_x)/2*log(pi) + lgamma(d_x/2+1) - lgamma(d_y/2+1)
                float log_lhs =
                        ((float)d_x / 2.0f) *
                        std::log(std::max(disx, 1e-30f));
                float log_n_ratio =
                        std::log(std::max(n_y, 1.0f)) -
                        std::log(std::max(n_x, 1.0f));
                float log_vol_ratio =
                        ((float)(d_y - d_x) / 2.0f) *
                                std::log((float)M_PI) +
                        std::lgamma((float)d_x / 2.0f + 1.0f) -
                        std::lgamma((float)d_y / 2.0f + 1.0f);
                float log_rhs =
                        log_n_ratio + log_vol_ratio +
                        ((float)d_y / 2.0f) *
                                std::log(std::max(disy, 1e-30f));

                if (log_lhs <= log_rhs) {
                    // Lemma 2 satisfied: can skip from cur_level to y+1.
                    best_skip = cur_level - y;
                    break; // first y from bottom gives max skip
                }
            }

            if (best_skip > 0) {
                density_skipLevels.push({-disx, best_skip});
            }
        }
    }

    // Insert all samples into the shortcut map
    while (!density_skipLevels.empty()) {
        auto top = density_skipLevels.top();
        density_skipLevels.pop();
        shortcut.insert_or_assign(-top.first, top.second);
    }
}

// ---------------------------------------------------------------------------
// build_shortcut
// ---------------------------------------------------------------------------

void IndexSHG::build_shortcut() {
    FAISS_THROW_IF_NOT_MSG(
            ntotal > 0,
            "IndexSHG: build_shortcut() called on empty index");

    // Recompute compression params in case d was set later
    if (dim_at_level.empty()) {
        compute_compression_params();
    }

    if (verbose) {
        printf("IndexSHG::build_shortcut: maxFixLevel_=%d, data_rep_size_=%zu, "
               "d=%d, eta=%d\n",
               maxFixLevel_, data_rep_size_, d, eta);
        printf("  Dimension at each level:");
        for (int l = 0; l <= maxFixLevel_; ++l) {
            printf(" [%d]=%d", l, dim_at_level[l]);
        }
        printf("\n");
    }

    // Step 1: build compressed vectors (skip if already built during add())
    if (compressed_vecs.empty()) {
        if (verbose) {
            printf("IndexSHG::build_shortcut: building compressed vectors for "
                   "%" PRId64 " vectors ...\n",
                   ntotal);
        }
        build_all_compressed();
    } else if (verbose) {
        printf("IndexSHG::build_shortcut: compressed vectors already built "
               "(%" PRId64 " vectors)\n",
               ntotal);
    }

    // Step 2: build shortcuts using density criterion
    if (verbose) {
        printf("IndexSHG::build_shortcut: building shortcuts ...\n");
    }
    build_shortcuts_density();

    if (verbose) {
        printf("IndexSHG::build_shortcut: shortcut has %d entries\n",
               shortcut.size());
    }
}

// ---------------------------------------------------------------------------
// build_compressed_query_rep
// ---------------------------------------------------------------------------

void IndexSHG::build_compressed_query_rep(
        const float* query,
        std::vector<float>& full_rep,
        std::vector<float>& query_rep) const {
    full_rep.clear();
    full_rep.insert(full_rep.end(), query, query + d);

    int prev_pos = 0;
    for (int lev = 1; lev <= maxFixLevel_; ++lev) {
        int prev_dim = dim_at_level[lev - 1];
        for (int j = 0; j < prev_dim; j += eta) {
            int end = std::min(j + eta, prev_dim);
            float sum = 0.0f;
            for (int jj = j; jj < end; ++jj)
                sum += full_rep[prev_pos + jj];
            full_rep.push_back(sum / (float)(end - j));
        }
        prev_pos += prev_dim;
    }
    std::copy(full_rep.begin() + d, full_rep.end(), query_rep.begin());
}

// ---------------------------------------------------------------------------
// navigate_upper_levels — uses forked greedy_update_nearest_shg
// ---------------------------------------------------------------------------
// Algorithm 3: Navigate upper levels with optional shortcut-based skipping.
// At each visited level, performs greedy 1-NN search using compressed
// distances (via greedy_update_nearest_shg which uses FAISS batch-4 pattern
// with inline compressed distance computation and caching).

IndexSHG::storage_idx_t IndexSHG::navigate_upper_levels(
        const std::vector<float>& query_rep,
        bool use_shortcut_flag) const {
    const HNSW& hns = hnsw;
    int max_l = hns.max_level;

    if (max_l == 0 || maxFixLevel_ == 0) {
        return (storage_idx_t)hns.entry_point;
    }

    storage_idx_t currObj = (storage_idx_t)hns.entry_point;
    int level = max_l;

    // Compute initial compressed distance at top level
    int cl = std::min(level, maxFixLevel_);
    const float* q_comp = query_rep.data() + offset_at_level[cl];
    const float* c_data = get_compressed_data(currObj, cl);
    float curdist = fvec_L2sqr(q_comp, c_data, (size_t)dim_at_level[cl]);

    // Algorithm 3, lines 3-7: shortcut loop.
    // Paper condition: "while l − f(dis̃) ≥ 1 do".
    // If f(dis) >= l (new_level < 1), exit without searching level 1 — the
    // current entry point goes directly to base-level search.
    if (use_shortcut_flag && shortcut.is_trained()) {
        while (level > 0) {
            int skip = shortcut.predict(curdist);
            int new_level = level - skip;
            if (new_level < 1) break;        // skip past base: exit per Alg. 3
            if (new_level >= level) new_level = level - 1;
            level = new_level;

            // Compute compressed distance at new level
            cl = std::min(level, maxFixLevel_);
            q_comp = query_rep.data() + offset_at_level[cl];
            c_data = get_compressed_data(currObj, cl);
            curdist = fvec_L2sqr(
                    q_comp, c_data, (size_t)dim_at_level[cl]);

            // Greedy 1-NN search at this level (FAISS batch-4 pattern)
            greedy_update_nearest_shg(
                    hns, *this, query_rep, level, currObj, curdist);

            if (level <= 1) break;
        }
    } else {
        // Standard HNSW upper-level greedy descent (no shortcut)
        for (; level > 0; --level) {
            cl = std::min(level, maxFixLevel_);
            q_comp = query_rep.data() + offset_at_level[cl];
            c_data = get_compressed_data(currObj, cl);
            curdist = fvec_L2sqr(
                    q_comp, c_data, (size_t)dim_at_level[cl]);

            greedy_update_nearest_shg(
                    hns, *this, query_rep, level, currObj, curdist);
        }
    }

    return currObj;
}

// ---------------------------------------------------------------------------
// search — unified single-OMP loop (following FAISS hnsw_search pattern)
// ---------------------------------------------------------------------------
// Combines upper-level navigation (Phase 1) and base-level search (Phase 2)
// into a single per-query OMP loop. Uses:
//   - HeapBlockResultHandler for result collection
//   - Per-thread DisCache (reused across queries via epoch invalidation)
//   - Per-thread query_rep and scratch buffers
//   - greedy_update_nearest_shg for upper levels
//   - search_from_candidates_shg for base level (MinimaxHeap + batch-4)

void IndexSHG::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            ntotal > 0, "IndexSHG: search called on empty index");
    FAISS_THROW_IF_NOT(k > 0);

    // When d < eta^2, no compression levels exist — fall back to HNSW.
    if (maxFixLevel_ == 0) {
        IndexHNSWFlat::search(n, x, k, distances, labels, params);
        return;
    }

    FAISS_THROW_IF_NOT_MSG(
            !compressed_vecs.empty() || hnsw.max_level == 0,
            "IndexSHG: build_shortcut() must be called before search()");

    bool use_shortcut_flag = true;
    bool use_lb_flag = true;
    bool do_dis_check = true; // FAISS check_relative_distance default

    int efSearch = (int)k; // default: efSearch = k
    int ef = (int)k;

    if (params != nullptr) {
        const auto* sp = dynamic_cast<const SearchParametersSHG*>(params);
        if (sp != nullptr) {
            use_shortcut_flag = sp->use_shortcut;
            use_lb_flag = sp->use_lb_pruning;
            do_dis_check = sp->check_relative_distance;
            if (sp->efSearch > (int)k) {
                efSearch = sp->efSearch;
                ef = sp->efSearch;
            }
        }
    }

    if (!shortcut.is_trained()) {
        use_shortcut_flag = false;
    }

    int max_l = hnsw.max_level;

    // HeapBlockResultHandler manages k-nearest results with threshold
    // tracking (following FAISS IndexHNSW::search pattern)
    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

#pragma omp parallel
    {
        // Per-thread result handler
        RH::SingleResultHandler res(bres);

        // Per-thread visited table
        VisitedTable vt(ntotal, hnsw.use_visited_hashset);

        // Per-thread exact distance computer (for base-level search)
        std::unique_ptr<DistanceComputer> qdis(get_distance_computer());

        // Per-thread compressed query representation
        std::vector<float> query_rep(data_rep_size_);
        std::vector<float> full_rep;
        full_rep.reserve(data_rep_size_ + d);

#pragma omp for schedule(dynamic)
        for (idx_t i = 0; i < n; ++i) {
            res.begin(i);
            const float* query = x + (size_t)i * d;
            qdis->set_query(query);

            // Build compressed query representation
            build_compressed_query_rep(query, full_rep, query_rep);

            // Phase 1: Upper navigation with shortcuts (Algorithm 3, lines 1-7)
            // Uses greedy_update_nearest_shg with compressed distances
            storage_idx_t ep = navigate_upper_levels(
                    query_rep, use_shortcut_flag);

            // Compute exact distance to entry point for base search
            float ep_dist = (*qdis)(ep);

            // Phase 2: Base-level search with LB pruning (Algorithm 3, lines 8-19)
            // ef = max(k, efSearch) to allow recall-time tradeoff sweep.
            HNSW::MinimaxHeap candidates(ef);
            candidates.push(ep, ep_dist);

            search_from_candidates_shg(
                    hnsw, *this, *qdis, res, candidates, vt,
                    query_rep, use_lb_flag, do_dis_check, efSearch);

            res.end();
            vt.advance();
        }
    }
}

} // namespace faiss
