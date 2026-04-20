/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-
#pragma once

/**
 * IndexCSPG: Crossing Sparse Proximity Graph index for FAISS.
 *
 * Implements the CSPG algorithm from:
 *   Ming Yang, Yuzheng Cai, Weiguo Zheng. "CSPG: Crossing Sparse Proximity
 *   Graphs for Approximate Nearest Neighbor Search." NeurIPS 2024.
 *
 * Core idea:
 *   Randomly partition the dataset into m subsets that share a common set of
 *   routing vectors (fraction λ of the data).  Build an independent HNSW
 *   graph per partition.  At query time, perform a two-stage search:
 *     1. Fast approaching – greedy/beam search on partition 0's HNSW to find
 *        a good entry point (controlled by ef1).
 *     2. Cross-partition expansion – beam search across all partitions with
 *        candidate set size ef2.  When a routing vector is encountered, it is
 *        simultaneously inserted into the candidate sets of ALL partitions,
 *        enabling the search to "cross" between graphs.
 *
 * Usage
 * -----
 *   IndexCSPG idx(d, M, num_partitions, lambda);
 *   idx.efConstruction = 128;
 *   idx.add(n, data);                  // builds m HNSW partitions
 *
 *   idx.efSearch = 128;                // default ef2
 *   idx.search(nq, queries, k, distances, labels);
 *
 *   // or with per-query parameters:
 *   SearchParametersCSPG params;
 *   params.ef1 = 1;
 *   params.efSearch = 200;
 *   idx.search(nq, queries, k, distances, labels, &params);
 */

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>

#include <vector>

namespace faiss {

// ---------------------------------------------------------------------------
// Search parameters
// ---------------------------------------------------------------------------

struct SearchParametersCSPG : SearchParameters {
    /// Second-stage candidate set size (ef2 in the paper).
    int efSearch = 128;

    /// First-stage expansion factor.  ef1=1 uses greedy descent (fastest).
    /// ef1>1 uses beam search at level 0 in partition 0.
    int ef1 = 1;

    ~SearchParametersCSPG() {}
};

// ---------------------------------------------------------------------------
// IndexCSPG
// ---------------------------------------------------------------------------

struct IndexCSPG : Index {
    // --- HNSW parameters for partition sub-graphs ---

    /// Number of bi-directional links per node (HNSW M parameter).
    int M = 32;

    /// Expansion factor during HNSW construction.
    int efConstruction = 40;

    // --- CSPG parameters ---

    /// Number of partitions (m in the paper).  Default 2.
    int num_partitions = 2;

    /// Routing vector sampling ratio (λ in the paper).
    /// λn vectors are replicated across all partitions as routing vectors.
    float lambda = 0.5f;

    // --- Search defaults ---

    /// Default first-stage expansion factor.
    int ef1 = 1;

    /// Default second-stage candidate set size (ef2).
    int efSearch = 128;

    // --- Shared vector storage ---

    /// Single contiguous storage for ALL vectors (no duplication).
    /// Routing vectors are stored once; partitions reference them by global ID.
    IndexFlat* shared_flat = nullptr;

    // --- Partition sub-graphs ---

    /// One HNSW-Flat index per partition.  Owned if own_fields is true.
    /// During construction the partitions contain their own vector copies;
    /// at search time we bypass them and use shared_flat via remapped DCs.
    std::vector<IndexHNSWFlat*> partitions;
    bool own_fields = true;

    // --- ID mapping ---

    /// refunction[partition_id][local_id] = original global ID.
    /// Routing vectors occupy local IDs 0..num_routing-1 in every partition,
    /// mapped to the same (sorted) sequence of global IDs.
    std::vector<std::vector<idx_t>> refunction;

    /// Reverse mapping: global_id -> (partition_id, local_id).
    /// For routing vectors (present in all partitions), stores the first
    /// partition encountered during construction.
    std::vector<std::pair<int, idx_t>> global_to_local;

    /// Number of routing vectors (= floor(ntotal * lambda)).
    idx_t num_routing = 0;

    // --- Lifecycle ---

    explicit IndexCSPG(
            int d = 0,
            int M = 32,
            int num_partitions = 2,
            float lambda = 0.5f,
            MetricType metric = METRIC_L2);

    ~IndexCSPG() override;

    /// Add all vectors at once and build the CSPG index.
    /// Must be called exactly once (incremental add not supported).
    void add(idx_t n, const float* x) override;

    /// Two-stage CSPG search (Algorithm 1 in the paper).
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reset() override;
    void train(idx_t n, const float* x) override;
    void reconstruct(idx_t key, float* recons) const override;

    // --- Helpers ---

    /// Pointer to vector data for a given (partition, local_id) pair.
    /// Uses shared_flat: maps local_id → flat position → shared storage.
    const float* get_vec(int pid, idx_t local_id) const {
        idx_t flat_pos = refunction[pid][local_id];
        return shared_flat->get_xb() + flat_pos * d;
    }
};

} // namespace faiss
