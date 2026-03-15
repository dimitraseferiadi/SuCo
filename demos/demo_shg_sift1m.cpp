/**
 * demo_shg_sift1m.cpp
 *
 * Quick smoke-test for IndexSHG on the SIFT1M dataset.
 *
 * Usage
 * -----
 *   # Build (from the repo build dir):
 *   cmake --build . --target demo_shg_sift1m
 *
 *   # Run (needs ANN_SIFT1M unpacked in sift1M/):
 *   ./demos/demo_shg_sift1m /path/to/sift1M
 *
 * Dataset: http://corpus-texmex.irisa.fr/  (ANN_SIFT1M, ~161 MB)
 *   mkdir sift1M && cd sift1M
 *   wget http://corpus-texmex.irisa.fr/ANN_SIFT1M.tar.gz
 *   tar -xzf ANN_SIFT1M.tar.gz && mv sift/* . && rmdir sift && cd ..
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/time.h>

#include <faiss/IndexSHG.h>

// ---------------------------------------------------------------------------
// fvecs / ivecs helpers (same as demo_sift1M.cpp)
// ---------------------------------------------------------------------------

static float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert(d > 0 && d < 1000000);
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0);
    size_t n = sz / ((d + 1) * 4);
    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1));
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(float));
    fclose(f);
    return x;
}

static int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

static double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// ---------------------------------------------------------------------------
// Recall@1 helper
// ---------------------------------------------------------------------------
static float recall_at_1(
        const faiss::idx_t* I,
        const int* gt,
        size_t nq,
        size_t k) {
    int hits = 0;
    for (size_t i = 0; i < nq; i++) {
        int true_nn = gt[i * k];
        for (size_t j = 0; j < k; j++) {
            if (I[i * k + j] == true_nn) {
                hits++;
                break;
            }
        }
    }
    return (float)hits / (float)nq;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    const char* data_dir = (argc > 1) ? argv[1] : "sift1M";

    char path_base[512], path_query[512], path_gt[512];
    snprintf(path_base,  sizeof(path_base),  "%s/sift_base.fvecs",       data_dir);
    snprintf(path_query, sizeof(path_query), "%s/sift_query.fvecs",      data_dir);
    snprintf(path_gt,    sizeof(path_gt),    "%s/sift_groundtruth.ivecs", data_dir);

    // ---- load data ----
    size_t d, nbase, nq, dq, d_gt, ngt;
    printf("Loading base vectors ... "); fflush(stdout);
    double t0 = elapsed();
    float* xb = fvecs_read(path_base,  &d,  &nbase);
    float* xq = fvecs_read(path_query, &dq, &nq);
    int*   gt = ivecs_read(path_gt,    &d_gt, &ngt);
    assert(d == dq);
    printf("done in %.1f s  (nb=%zu, nq=%zu, d=%zu)\n",
           elapsed() - t0, nbase, nq, d);

    const int M = 32;
    const int k = 10;
    const int efSearch = 64;

    // ---- build index ----
    printf("\n--- Building IndexSHG (M=%d) ---\n", M);
    faiss::IndexSHG index((int)d, M);
    index.hnsw.efConstruction = 200;
    index.verbose = true;

    t0 = elapsed();
    index.add((faiss::idx_t)nbase, xb);
    printf("add() done in %.1f s\n", elapsed() - t0);

    t0 = elapsed();
    index.build_shortcut();
    printf("build_shortcut() done in %.1f s\n", elapsed() - t0);

    // ---- search with shortcuts ----
    printf("\n--- Search (efSearch=%d, k=%d) ---\n", efSearch, k);
    index.hnsw.efSearch = efSearch;

    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float>        D(nq * k);

    // warm-up
    index.search(1, xq, k, D.data(), I.data());

    t0 = elapsed();
    index.search((faiss::idx_t)nq, xq, k, D.data(), I.data());
    double t_shg = elapsed() - t0;

    float r1 = recall_at_1(I.data(), gt, nq, k);
    printf("SHG   : R@1=%.4f  QPS=%.0f  (%.3f s total)\n",
           r1, (double)nq / t_shg, t_shg);

    // ---- compare: plain HNSW (no shortcuts, no LB pruning) ----
    printf("\n--- Search without shortcuts (baseline) ---\n");
    faiss::SearchParametersSHG sp;
    sp.efSearch = efSearch;
    sp.use_shortcut  = false;
    sp.use_lb_pruning = false;

    t0 = elapsed();
    index.search((faiss::idx_t)nq, xq, k, D.data(), I.data(), &sp);
    double t_base = elapsed() - t0;

    r1 = recall_at_1(I.data(), gt, nq, k);
    printf("No-SC : R@1=%.4f  QPS=%.0f  (%.3f s total)\n",
           r1, (double)nq / t_base, t_base);

    printf("\nSpeedup from shortcuts + LB-pruning: %.2fx\n",
           t_base / t_shg);

    delete[] xb;
    delete[] xq;
    delete[] gt;
    return 0;
}
