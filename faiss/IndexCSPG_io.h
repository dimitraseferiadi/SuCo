/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

/**
 * IndexCSPG I/O support.
 *
 * Provides write/read functions for the CSPG-specific data
 * (parameters, refunction mapping, global_to_local mapping).
 * The partition HNSW sub-indices are written/read using the standard
 * FAISS write_index / read_index.
 */

#pragma once

#include <faiss/IndexCSPG.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>
#include <faiss/index_io.h>

#include <cstdint>
#include <cstring>

namespace faiss {

// -------------------------------------------------------------------------
// Write
// -------------------------------------------------------------------------

inline void write_index_cspg_extra(const IndexCSPG* idx, IOWriter* f) {
    // CSPG parameters
    WRITE1(idx->M);
    WRITE1(idx->efConstruction);
    WRITE1(idx->num_partitions);
    WRITE1(idx->lambda);
    WRITE1(idx->ef1);
    WRITE1(idx->efSearch);

    // Routing vector count
    int64_t nr = static_cast<int64_t>(idx->num_routing);
    WRITE1(nr);

    // refunction mapping
    int32_t np = static_cast<int32_t>(idx->num_partitions);
    WRITE1(np);
    for (int p = 0; p < np; p++) {
        uint64_t rs = static_cast<uint64_t>(idx->refunction[p].size());
        WRITE1(rs);
        if (rs > 0) {
            f->operator()(idx->refunction[p].data(), sizeof(idx_t), rs);
        }
    }

    // global_to_local mapping
    uint64_t gl_size = static_cast<uint64_t>(idx->global_to_local.size());
    WRITE1(gl_size);
    for (uint64_t i = 0; i < gl_size; i++) {
        int32_t gid = static_cast<int32_t>(idx->global_to_local[i].first);
        int64_t lid = static_cast<int64_t>(idx->global_to_local[i].second);
        WRITE1(gid);
        WRITE1(lid);
    }

    // Shared flat storage (all vectors stored once)
    write_index(idx->shared_flat, f);

    // Partition sub-indices (HNSW graphs; flat storage is empty after build)
    for (int p = 0; p < np; p++) {
        write_index(idx->partitions[p], f);
    }
}

// -------------------------------------------------------------------------
// Read
// -------------------------------------------------------------------------

inline void read_index_cspg_extra(IndexCSPG* idx, IOReader* f) {
    // CSPG parameters
    READ1(idx->M);
    READ1(idx->efConstruction);
    READ1(idx->num_partitions);
    READ1(idx->lambda);
    READ1(idx->ef1);
    READ1(idx->efSearch);

    // Routing vector count
    int64_t nr = 0;
    READ1(nr);
    idx->num_routing = static_cast<idx_t>(nr);

    // refunction mapping
    int32_t np = 0;
    READ1(np);
    FAISS_THROW_IF_NOT(np == idx->num_partitions);
    idx->refunction.resize(np);
    for (int p = 0; p < np; p++) {
        uint64_t rs = 0;
        READ1(rs);
        idx->refunction[p].resize(rs);
        if (rs > 0) {
            f->operator()(idx->refunction[p].data(), sizeof(idx_t), rs);
        }
    }

    // global_to_local mapping
    uint64_t gl_size = 0;
    READ1(gl_size);
    idx->global_to_local.resize(gl_size);
    for (uint64_t i = 0; i < gl_size; i++) {
        int32_t gid = 0;
        int64_t lid = 0;
        READ1(gid);
        READ1(lid);
        idx->global_to_local[i] = {
                static_cast<int>(gid), static_cast<idx_t>(lid)};
    }

    // Shared flat storage
    delete idx->shared_flat;
    {
        Index* raw_flat = read_index(f);
        auto* flat = dynamic_cast<IndexFlat*>(raw_flat);
        FAISS_THROW_IF_NOT_MSG(
                flat != nullptr,
                "IndexCSPG I/O: shared_flat is not IndexFlat");
        idx->shared_flat = flat;
    }

    // Partition sub-indices
    if (idx->own_fields) {
        for (auto* part : idx->partitions)
            delete part;
    }
    idx->partitions.clear();
    idx->partitions.resize(np, nullptr);
    for (int p = 0; p < np; p++) {
        Index* raw = read_index(f);
        auto* part = dynamic_cast<IndexHNSWFlat*>(raw);
        FAISS_THROW_IF_NOT_MSG(
                part != nullptr,
                "IndexCSPG I/O: partition is not IndexHNSWFlat");
        idx->partitions[p] = part;
    }
    idx->own_fields = true;
}

} // namespace faiss
