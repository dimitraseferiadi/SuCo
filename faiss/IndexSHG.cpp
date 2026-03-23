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
 * levels. maxFixLevel_ compression levels are computed from (d, eta=4) by
 * repeatedly dividing: level l has dim ceil(d/eta^l), stopping when
 * dim/eta < eta.  HNSW level l uses compression level min(l, maxFixLevel_).
 *
 * This matches the original code where k_=4 and maxFixLevel_ is computed as:
 *   while(dim/k_ >= k_) { maxFixLevel_++; dim = ceil(dim/k_); }
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
 */

#include <faiss/IndexSHG.h>
#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/HNSW.h>
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
// ShortcutMap serialization
// ---------------------------------------------------------------------------

void ShortcutMap::write(FILE* f) const {
    int n = (int)entries.size();
    fwrite(&n, sizeof(int), 1, f);
    for (const auto& kv : entries) {
        fwrite(&kv.first, sizeof(float), 1, f);
        fwrite(&kv.second, sizeof(int), 1, f);
    }
}

void ShortcutMap::read(FILE* f) {
    int n = 0;
    [[maybe_unused]] size_t r;
    r = fread(&n, sizeof(int), 1, f);
    entries.clear();
    for (int i = 0; i < n; ++i) {
        float dist;
        int skip;
        r = fread(&dist, sizeof(float), 1, f);
        r = fread(&skip, sizeof(int), 1, f);
        entries[dist] = skip;
    }
}

// ---------------------------------------------------------------------------
// IndexSHG constructor
// ---------------------------------------------------------------------------

IndexSHG::IndexSHG(int d, int M, MetricType metric)
        : IndexHNSWFlat(d, M, metric) {
    if (d > 0) {
        compute_compression_params();
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
// get_dis_by_level - squared L2 between two nodes at HNSW level
// ---------------------------------------------------------------------------

float IndexSHG::get_dis_by_level(
        idx_t id1,
        idx_t id2,
        int hnsw_level) const {
    if (hnsw_level == 0) {
        // Direct pointer access to flat storage (avoids reconstruct copy).
        const auto* flat = dynamic_cast<const IndexFlat*>(storage);
        const float* v1 = flat->get_xb() + (size_t)id1 * d;
        const float* v2 = flat->get_xb() + (size_t)id2 * d;
        return fvec_L2sqr(v1, v2, (size_t)d);
    }
    if (hnsw_level >= maxFixLevel_) {
        // Original: at maxFixLevel_ and above, only compare the first
        // element of the maxFixLevel_ compressed representation.
        const float* a = get_compressed_data(id1, maxFixLevel_);
        const float* b = get_compressed_data(id2, maxFixLevel_);
        float t = a[0] - b[0];
        return t * t;
    }
    int cl = hnsw_level;
    int cdim = dim_at_level[cl];
    const float* a = get_compressed_data(id1, cl);
    const float* b = get_compressed_data(id2, cl);
    return compressed_l2sqr(a, b, cdim);
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
    if (hnsw_level >= maxFixLevel_) {
        // Original: at maxFixLevel_ and above, only compare first element
        const float* q = query_rep.data() + offset_at_level[maxFixLevel_];
        const float* n = get_compressed_data(node_id, maxFixLevel_);
        float t = q[0] - n[0];
        return t * t;
    }
    int cl = hnsw_level;
    int cdim = dim_at_level[cl];
    const float* q = query_rep.data() + offset_at_level[cl];
    const float* n = get_compressed_data(node_id, cl);
    return compressed_l2sqr(q, n, cdim);
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
    // Match original addDataPoint compression:
    // Build progressive mean aggregation from level 0 data through
    // all compression levels.
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

    // Match original: searchBaseLayer(curData, data_point, label, level)
    // The original does a beam search at the given HNSW level using
    // ef_construction_ as beam width, computing level-0 (full) distances.
    // Then keeps top-2 and returns the best.

    using NodeDist = std::pair<float, idx_t>;

    std::priority_queue<NodeDist, std::vector<NodeDist>> top_candidates;
    std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>>
            candidateSet;

    VisitedTable visited(ntotal);

    // Start from the node itself as entry point
    float dist_ep = get_dis_by_level(node_id, node_id, 0); // 0 distance
    // Actually the original passes ep_id=curData, which is the same as the
    // query node. So initial distance is 0. We need to start the beam
    // search from this node's neighbours.
    // However the original's searchBaseLayer starts from ep_id which gets
    // distance to itself (0), and then explores from there.

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

            // Level-0 (full) distance, matching original
            float dist1 = get_dis_by_level(node_id, cand, 0);

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
        int node_levels = hns.levels[i]; // number of levels this node is on
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
        int point_level = hns.levels[i] - 1; // highest HNSW level for this node

        int cur_level = point_level;
        while (cur_level > 1) {
            int skipLevel = 1;
            int skip_to_level = cur_level - 2;

            // Find nearest graph neighbour at current level
            auto nearest_result = get_nearest_by_level(i, cur_level);
            float disx_nn_dist = nearest_result.first;
            idx_t nearest = nearest_result.second;
            if (nearest < 0 || disx_nn_dist < 0) {
                cur_level--;
                continue;
            }

            // Distance at current compressed level
            float disx = get_dis_by_level(i, nearest, cur_level);

            // Check density condition for lower levels
            while (skip_to_level >= 0) {
                auto skip_result = get_nearest_by_level(i, skip_to_level);
                float disy_nn_dist = skip_result.first;
                idx_t nearest_y = skip_result.second;
                if (nearest_y < 0 || disy_nn_dist < 0) {
                    break;
                }

                float disy = get_dis_by_level(i, nearest_y, skip_to_level);

                // Cross-level distance check (original condition 1):
                // distance between node i and its NN from cur_level,
                // computed at skip_to_level
                float dis_cross = get_dis_by_level(i, nearest, skip_to_level);

                int d_x = get_dim_at_level(cur_level);
                int d_y = get_dim_at_level(skip_to_level);

                // Density-based condition matching original buildShortcuts:
                float n_x = (float)levelCounts[cur_level];
                float n_y = (float)levelCounts[skip_to_level];

                bool can_skip = false;

                // Condition 1: cross-level distance is small enough
                if (dis_cross <= disy * 2.0f) {
                    can_skip = true;
                } else {
                    // Condition 2: complex formula from original
                    float c1 = std::pow(n_x / std::max(n_y, 1.0f), 2.0f);
                    float c2_1 = std::pow(
                            (float)M_PI,
                            2.0f / std::pow((float)eta, (float)d_x));
                    float c2 = std::pow(
                            c2_1 / (float)eta,
                            (float)d_x *
                                    (float)(cur_level - skip_to_level));
                    float c3 = 1.0f;
                    for (int val = d_x; val <= d_y + 1; ++val) {
                        // Original uses integer division: 1/value
                        // This gives 0 for val >= 2, making c3 = 0
                        c3 = c3 * (float)(1 / val);
                    }
                    if (std::pow(disx, (float)d_x) <=
                            c1 * c2 * c3 *
                                    std::pow(disy, (float)d_y)) {
                        can_skip = true;
                    }
                }

                if (can_skip) {
                    skipLevel++;
                } else {
                    break;
                }

                skip_to_level--;
            }

            if (skipLevel > 1 ||
                    density_skipLevels.size() < (size_t)hns.efSearch) {
                density_skipLevels.push({-disx, skipLevel});
            }

            cur_level--;
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

    // Step 1: build compressed vectors for all nodes
    if (verbose) {
        printf("IndexSHG::build_shortcut: building compressed vectors for "
               "%" PRId64 " vectors ...\n",
               ntotal);
    }
    build_all_compressed();

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
// navigate_upper_levels
// ---------------------------------------------------------------------------

IndexSHG::storage_idx_t IndexSHG::navigate_upper_levels(
        const std::vector<float>& query_rep,
        bool use_shortcut_flag,
        dis_cache_t& dis_cache,
        int max_level_cache) const {
    const HNSW& hns = hnsw;
    int max_l = hns.max_level;

    if (max_l == 0) {
        return (storage_idx_t)hns.entry_point;
    }

    storage_idx_t currObj = (storage_idx_t)hns.entry_point;

    for (int level = max_l; level > 0;) {
        // Compute compressed distance at this level
        float curdist = get_dis_by_level_q(query_rep, currObj, level);
        // Cache it
        uint64_t cache_key_curr =
                (uint64_t)currObj * (max_level_cache + 1) + level;
        dis_cache[cache_key_curr] = curdist;

        // Greedy search at this level
        bool changed = true;
        while (changed) {
            changed = false;
            size_t nb_begin, nb_end;
            hns.neighbor_range(currObj, level, &nb_begin, &nb_end);

            for (size_t nb = nb_begin; nb < nb_end; ++nb) {
                storage_idx_t cand = hns.neighbors[nb];
                if (cand < 0) break;

                // Upper-level pruning (pruneDisCompute):
                // Check if any cached distance at a higher level can prune
                bool pruned = false;
                for (int cl = max_l; cl > level; --cl) {
                    uint64_t key =
                            (uint64_t)cand * (max_level_cache + 1) + cl;
                    auto it = dis_cache.find(key);
                    if (it != dis_cache.end()) {
                        // Original: pow(k_, l - currLevel + 1)
                        float infer = it->second *
                                std::pow((float)eta,
                                         (float)(cl - level + 1));
                        if (infer > curdist) {
                            pruned = true;
                            break;
                        }
                    }
                }
                if (pruned) continue;

                float d_cand = get_dis_by_level_q(query_rep, cand, level);
                // Cache the distance
                uint64_t cache_key_cand =
                        (uint64_t)cand * (max_level_cache + 1) + level;
                dis_cache[cache_key_cand] = d_cand;

                if (d_cand < curdist) {
                    curdist = d_cand;
                    currObj = cand;
                    changed = true;
                }
            }
        }

        // Try shortcut — matches original logic exactly
        if (use_shortcut_flag && shortcut.size() >= 100) {
            int skip = shortcut.predict(curdist);
            int skip_to = level - skip;

            if (skip_to == level - 1) {
                level--;
            } else {
                level = (skip_to > 0) ? skip_to : 0;
            }
            if (skip_to == level) {
                level--;
            }
        } else {
            level--;
        }
    }

    return currObj;
}

// ---------------------------------------------------------------------------
// prune_by_lb - lower-bound pruning using upper-level compressed distances
// ---------------------------------------------------------------------------

bool IndexSHG::prune_by_lb(
        float current_bound,
        int comp_level,
        const std::vector<float>& query_rep,
        idx_t candidate) const {
    if (comp_level <= 0 || maxFixLevel_ <= 0) return false;

    int cl = std::min(comp_level, maxFixLevel_);
    int cdim = dim_at_level[cl];
    const float* q = query_rep.data() + offset_at_level[cl];
    const float* c = get_compressed_data(candidate, cl);

    float approx_dis = compressed_l2sqr(q, c, cdim);
    // Original: inferDis = approDis * pow(k_, 3) with level=2, i.e. pow(k_, cl+1)
    float infer_dis = approx_dis * std::pow((float)eta, (float)(cl + 1));
    return infer_dis > current_bound;
}

// ---------------------------------------------------------------------------
// prune_by_cache - pruning based on cached higher-level distances
// ---------------------------------------------------------------------------

bool IndexSHG::prune_by_cache(
        float current_bound,
        int cur_level,
        idx_t candidate,
        const dis_cache_t& dis_cache,
        int max_level_cache) const {
    for (int cl = max_level_cache; cl > cur_level; --cl) {
        uint64_t key = (uint64_t)candidate * (max_level_cache + 1) + cl;
        auto it = dis_cache.find(key);
        if (it != dis_cache.end()) {
            // Original: pow(k_, l - currLevel + 1)
            float infer = it->second *
                    std::pow((float)eta, (float)(cl - cur_level + 1));
            if (infer > current_bound) return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// search_base_level
// ---------------------------------------------------------------------------

void IndexSHG::search_base_level(
        const float* query,
        idx_t k,
        storage_idx_t entry_point,
        float* out_distances,
        idx_t* out_labels,
        const std::vector<float>& query_rep,
        bool use_lb_pruning,
        dis_cache_t& dis_cache,
        int max_level_cache) const {
    const HNSW& hns = hnsw;

    std::unique_ptr<DistanceComputer> qdis(get_distance_computer());
    qdis->set_query(query);

    using NodeDist = std::pair<float, storage_idx_t>;
    std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>>
            candidates; // min-heap
    std::priority_queue<NodeDist> results; // max-heap

    VisitedTable visited(ntotal);

    float d_ep = (*qdis)(entry_point);
    candidates.push({d_ep, entry_point});
    results.push({d_ep, entry_point});
    visited.set(entry_point);

    // Original SHG uses ef=k for base search (not efSearch).
    // The shortcut + compressed navigation finds a good entry point,
    // so less exploration is needed at the base level.
    int ef = (int)k;
    int hops = 0;

    // Compression level for LB pruning (original uses level 2)
    int lb_comp_level = std::min(2, maxFixLevel_);

    while (!candidates.empty()) {
        float d_cand = candidates.top().first;
        storage_idx_t cand = candidates.top().second;

        // Original: candidate_dist > lowerBound (bare_bone_search path)
        if (d_cand > results.top().first) {
            break;
        }
        candidates.pop();
        hops++;

        size_t nb_begin, nb_end;
        hns.neighbor_range(cand, 0, &nb_begin, &nb_end);

        for (size_t nb = nb_begin; nb < nb_end; ++nb) {
            storage_idx_t u = hns.neighbors[nb];
            if (u < 0) break;

            // Original order: pruning checks BEFORE visited check.
            // This allows pruning unvisited nodes without computing
            // their full distance.

            // Cache-based pruning (pruneDisCompute at base level)
            if (use_lb_pruning &&
                    prune_by_cache(
                            results.top().first,
                            0, u, dis_cache, max_level_cache)) {
                continue;
            }

            // LB pruning after initial hops (hops > 20).
            // Original uses break: if one neighbor fails the LB test,
            // skip all remaining neighbors of this candidate.
            if (use_lb_pruning && hops > 20 &&
                    results.size() >= (size_t)ef &&
                    lb_comp_level > 0) {
                if (prune_by_lb(
                            results.top().first,
                            lb_comp_level,
                            query_rep,
                            u)) {
                    break;
                }
            }

            if (visited.get(u)) continue;
            visited.set(u);

            float d_u = (*qdis)(u);

            if (results.size() < (size_t)ef || d_u < results.top().first) {
                candidates.push({d_u, u});
                results.push({d_u, u});
                if ((int)results.size() > ef) {
                    results.pop();
                }
            }
        }
    }

    // Trim to k best
    while ((int)results.size() > k) {
        results.pop();
    }

    int n_res = (int)results.size();
    for (int i = n_res - 1; i >= 0; --i) {
        out_distances[i] = results.top().first;
        out_labels[i] = (idx_t)results.top().second;
        results.pop();
    }
    for (int i = n_res; i < (int)k; ++i) {
        out_distances[i] = std::numeric_limits<float>::max();
        out_labels[i] = -1;
    }
}

// ---------------------------------------------------------------------------
// search_one
// ---------------------------------------------------------------------------

void IndexSHG::search_one(
        const float* query,
        idx_t k,
        float* distances,
        idx_t* labels,
        bool use_shortcut_flag,
        bool use_lb_pruning) const {
    const HNSW& hns = hnsw;
    int max_l = hns.max_level;

    // Build compressed query representation for all levels
    std::vector<float> query_rep(data_rep_size_, 0.0f);

    // Same compression as compress_node, but for the query vector
    std::vector<float> full_rep;
    full_rep.reserve(data_rep_size_ + d);

    for (int i = 0; i < d; ++i) {
        full_rep.push_back(query[i]);
    }

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

    // Copy compressed levels (skip level-0 data)
    std::copy(full_rep.begin() + d, full_rep.end(), query_rep.begin());

    // Sparse distance cache: only stores entries for nodes actually visited.
    // The original uses a dense O(ntotal) array, but upper-level navigation
    // visits O(log n) nodes, so a hash map is far more memory-efficient.
    dis_cache_t dis_cache;

    // Navigate upper levels
    storage_idx_t ep = navigate_upper_levels(
            query_rep, use_shortcut_flag, dis_cache, max_l);

    // Base-level search
    search_base_level(
            query, k, ep, distances, labels, query_rep, use_lb_pruning,
            dis_cache, max_l);
}

// ---------------------------------------------------------------------------
// search (batch entry point)
// ---------------------------------------------------------------------------

void IndexSHG::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            ntotal > 0, "IndexSHG: search called on empty index");
    FAISS_THROW_IF_NOT_MSG(
            !compressed_vecs.empty() || hnsw.max_level == 0,
            "IndexSHG: build_shortcut() must be called before search()");

    bool use_shortcut_flag = true;
    bool use_lb = true;

    if (params != nullptr) {
        const auto* sp = dynamic_cast<const SearchParametersSHG*>(params);
        if (sp != nullptr) {
            use_shortcut_flag = sp->use_shortcut;
            use_lb = sp->use_lb_pruning;
        }
        const auto* hsp = dynamic_cast<const SearchParametersHNSW*>(params);
        if (hsp != nullptr) {
            const_cast<HNSW&>(hnsw).efSearch = hsp->efSearch;
        }
    }

    if (!shortcut.is_trained()) {
        use_shortcut_flag = false;
    }
    if (compressed_vecs.empty() || dim_at_level.empty()) {
        use_lb = false;
    }

#pragma omp parallel for schedule(dynamic)
    for (idx_t i = 0; i < n; ++i) {
        search_one(
                x + (size_t)i * d,
                k,
                distances + (size_t)i * k,
                labels + (size_t)i * k,
                use_shortcut_flag,
                use_lb);
    }
}

} // namespace faiss
