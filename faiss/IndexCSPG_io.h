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
 * (parameters, refunction mapping, all_vectors).
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

    // all_vectors
    uint64_t av_size = static_cast<uint64_t>(idx->all_vectors.size());
    WRITE1(av_size);
    if (av_size > 0) {
        f->operator()(idx->all_vectors.data(), sizeof(float), av_size);
    }

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

    // Partition sub-indices (full FAISS serialization)
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

    // all_vectors
    uint64_t av_size = 0;
    READ1(av_size);
    idx->all_vectors.resize(av_size);
    if (av_size > 0) {
        f->operator()(idx->all_vectors.data(), sizeof(float), av_size);
    }

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

    // Partition sub-indices
    // Clean up any existing partitions
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
