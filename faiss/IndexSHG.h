/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-
#pragma once

/**
 * IndexSHG: Shortcut-enabled Hierarchical Graph index for FAISS.
 *
 * Implements the SHG/HEDS algorithm from:
 *   Gong, Zeng, Chen. "Accelerating Approximate Nearest Neighbor Search in
 *   Hierarchical Graphs: Efficient Level Navigation with Shortcuts."
 *   PVLDB 18(10): 3518-3530, 2025.
 *
 * Two core innovations over HNSW:
 *   1. Hierarchical vector compression (progressive mean aggregation, eta=4).
 *      Upper-level distances use compressed low-dimensional representations,
 *      cutting per-level computation cost.  Compression levels are computed
 *      independently of the HNSW graph levels: maxFixLevel_ is determined by
 *      repeatedly dividing d by eta until (dim/eta < eta), and each HNSW graph
 *      level l uses compressed level min(l, maxFixLevel_).
 *   2. Learned shortcut: a sorted map from (approximate distance -> skip count)
 *      built using the PGM-index / sorted-map pattern.  Given the approximate
 *      distance between the query and the current entry point, the shortcut
 *      returns the number of HNSW levels that can safely be skipped.
 *      Training uses kNN density estimation (Lemma 2 in the paper).
 *
 * Usage
 * -----
 *   // Build:
 *   IndexSHG idx(d, M);
 *   idx.add(n, data);        // standard HNSW graph construction
 *   idx.build_shortcut();    // one-time: compress vectors + train shortcut
 *
 *   // Search (uses shortcuts automatically):
 *   idx.search(nq, queries, k, distances, labels);
 */

#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

namespace faiss {

// ---------------------------------------------------------------------------
// ShortcutMap
// ---------------------------------------------------------------------------

/**
 * Sorted-map shortcut model mapping approximate-distance -> skip count.
 *
 * This mirrors the DynamicPGMIndex usage in the original code: a sorted
 * collection of (distance, skip_count) pairs.  At query time,
 * lower_bound(dist) returns the skip count for the nearest distance key.
 *
 * We use std::map for simplicity and correctness; the PGM-index is an
 * optimization that does not change the algorithm.
 */
struct ShortcutMap {
    std::map<float, int> entries; ///< distance -> skip_count

    /// Insert or update (distance -> skip) mapping.
    void insert_or_assign(float dist, int skip) {
        entries[dist] = skip;
    }

    /// Return predicted skip count via lower_bound lookup.
    /// Returns 1 if no entry is found.
    int predict(float dist) const {
        if (entries.empty()) return 1;
        auto it = entries.lower_bound(dist);
        if (it == entries.end()) return 1;
        return it->second;
    }

    bool is_trained() const {
        return !entries.empty();
    }

    int size() const {
        return (int)entries.size();
    }

    // ---- serialization helpers ----
    void write(FILE* f) const;
    void read(FILE* f);
};

// ---------------------------------------------------------------------------
// Search parameters
// ---------------------------------------------------------------------------

struct SearchParametersSHG : SearchParametersHNSW {
    /// When true, use the shortcut for level skipping.
    bool use_shortcut = true;

    /// When true, apply compressed-vector lower-bound pruning at base level.
    bool use_lb_pruning = true;
};

// ---------------------------------------------------------------------------
// IndexSHG
// ---------------------------------------------------------------------------

/**
 * HNSW-based index augmented with hierarchical vector compression and a
 * learned shortcut for level navigation.
 *
 * Inherits from IndexHNSWFlat: the flat vector storage is used for exact
 * distance computation at the base level.  All upper-level distances use
 * the compressed representations stored in compressed_vecs.
 *
 * The compression hierarchy is independent of the HNSW levels:
 *   - maxFixLevel_ compression levels are computed from d and eta (=4),
 *     where each level l has dimension ceil(d / eta^l).
 *   - HNSW level l uses compression level min(l, maxFixLevel_).
 */
struct IndexSHG : IndexHNSWFlat {
    using storage_idx_t = HNSW::storage_idx_t;

    // --- compression ---

    /// Compression branching factor. Original SHG-Index uses k_=4.
    int eta = 4;

    /// Maximum compression level (computed from d and eta).
    int maxFixLevel_ = 0;

    /**
     * Compressed representations for all ntotal vectors.
     * Layout: compressed_vecs[node_id * data_rep_size_ + offset]
     * where offset is the start of the compressed data for a given level.
     * Level 0 = full dims (stored in flat storage, NOT here).
     * Levels 1..maxFixLevel_ are stored here concatenated.
     */
    std::vector<float> compressed_vecs;

    /// Total compressed representation size per node
    /// (sum of dims for levels 1..maxFixLevel_).
    size_t data_rep_size_ = 0;

    /// Dimension at each compression level (0..maxFixLevel_).
    std::vector<int> dim_at_level;

    /// Cumulative offset into per-node compressed data for each level.
    /// offset_at_level[l] = sum of dim_at_level[i] for i in 1..l-1.
    std::vector<size_t> offset_at_level;

    // --- shortcut ---

    ShortcutMap shortcut;

    // --- lifecycle ---

    explicit IndexSHG(
            int d = 0,
            int M = 32,
            MetricType metric = METRIC_L2);

    /**
     * Build compressed vectors for all nodes and train the shortcut.
     * Must be called once after all vectors have been added via add().
     */
    void build_shortcut();

    // --- core interface ---

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    // --- public helpers ---

    /// Return dimension at compression level l.
    int get_dim_at_level(int l) const;

    /// Compress a d-dimensional vector to compression level l.
    static void compress_vector(
            const float* vec,
            int d,
            int l,
            int eta,
            float* out);

    /// Squared L2 distance between two compressed vectors of length dim.
    static float compressed_l2sqr(
            const float* a,
            const float* b,
            int dim);

    /// Get pointer to compressed data for a node at a given compression level.
    /// Level 0 returns nullptr (use flat storage for level 0).
    const float* get_compressed_data(idx_t node_id, int comp_level) const;

    /// Compute compressed squared L2 distance between two nodes
    /// at a given HNSW level (capped at maxFixLevel_).
    float get_dis_by_level(idx_t id1, idx_t id2, int hnsw_level) const;

    /// Compute compressed squared L2 distance between a pre-built
    /// query compressed representation and a node at a given HNSW level.
    float get_dis_by_level_q(
            const std::vector<float>& query_rep,
            idx_t node_id,
            int hnsw_level) const;

   private:
    void compute_compression_params();
    void build_all_compressed();
    void compress_node(idx_t node_id);

    std::pair<float, idx_t> get_nearest_by_level(
            idx_t node_id, int hnsw_level) const;

    void build_shortcuts_density();

    void search_one(
            const float* query,
            idx_t k,
            float* distances,
            idx_t* labels,
            bool use_shortcut,
            bool use_lb_pruning) const;

    /// Sparse distance cache: key = node_id * (max_level+1) + level.
    /// Only stores distances for nodes actually visited (O(log n) entries
    /// during upper-level navigation), avoiding the O(ntotal) dense array.
    using dis_cache_t = std::unordered_map<uint64_t, float>;

    storage_idx_t navigate_upper_levels(
            const std::vector<float>& query_rep,
            bool use_shortcut,
            dis_cache_t& dis_cache,
            int max_level_cache) const;

    void search_base_level(
            const float* query,
            idx_t k,
            storage_idx_t entry_point,
            float* distances,
            idx_t* labels,
            const std::vector<float>& query_rep,
            bool use_lb_pruning,
            dis_cache_t& dis_cache,
            int max_level_cache) const;

    bool prune_by_lb(
            float current_bound,
            int comp_level,
            const std::vector<float>& query_rep,
            idx_t candidate) const;

    bool prune_by_cache(
            float current_bound,
            int cur_level,
            idx_t candidate,
            const dis_cache_t& dis_cache,
            int max_level_cache) const;
};

} // namespace faiss
