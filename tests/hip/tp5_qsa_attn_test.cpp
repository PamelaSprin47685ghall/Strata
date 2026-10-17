// tests/hip/tp5_qsa_attn_test.cpp - 4.2.3's first gate: the G-templated batched attention.
//
// G = n_head / n_head_kv became a template value (12 default, 6/4 for TP5's per-rank
// single-KV-head shapes of the [6,6,4,4,4] plan).  A rank's {n_head=6, n_head_kv=1} call
// must produce, for its 6 query heads, BIT FOR BIT the same rows as the unsplit {24,2}
// call produces for those heads: the per-head arithmetic (score, softmax, merge) is the
// same expression over the same operands; only the head grouping changed.
//
//   usage: tp5_qsa_attn_test [--seed S]
//   needs one GPU; exit 0 = pass, 1 = mismatch
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels;

namespace {

constexpr int HD = 256;
constexpr int kPages = 40;      // pool pages
constexpr int kPageSize = 4;    // cells per page (one indexer block)
constexpr int kCells = 96;      // selected cells (cap)
constexpr int kNQ = 3;          // queries per batch call

struct Case { int n_head, n_head_kv, q_start; };   // q_start: the rank's first head in the unsplit layout

}  // namespace

int main(int argc, char** argv) {
    unsigned seed = 42;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = unsigned(std::atoi(argv[++i]));
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        std::printf("tp5_qsa_attn_test: no GPU - SKIP\n");
        return 77;
    }

    // the unsplit geometry and the TP5 rank shapes of the plan's [6,6,4,4,4] split
    const Case full{24, 2, 0};
    const Case ranks[5] = {{6, 1, 0}, {6, 1, 6}, {4, 1, 12}, {4, 1, 16}, {4, 1, 20}};

    // fp16 pools: [page][kv_head][page_size][head_dim]
    const int kv_rows = kPages * 2 * kPageSize;
    std::vector<uint16_t> k_pool(kv_rows * HD), v_pool(kv_rows * HD);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> fk(-1.f, 1.f);
    for (auto& h : k_pool) *reinterpret_cast<__half*>(&h) = __float2half(fk(rng));
    for (auto& h : v_pool) *reinterpret_cast<__half*>(&h) = __float2half(fk(rng) * 0.5f);

    // page table: identity (page i holds cells 4i..4i+3)
    std::vector<int32_t> page_table(kPages);
    for (int i = 0; i < kPages; ++i) page_table[i] = i;

    // per-query selections and steps
    std::vector<int32_t> ids(kNQ * kCells);
    std::vector<int32_t> steps(kNQ * kStepCount);
    for (int q = 0; q < kNQ; ++q) {
        std::uniform_int_distribution<int> fc(0, kPages * kPageSize - 1);
        for (int c = 0; c < kCells; ++c) ids[q * kCells + c] = fc(rng);
        steps[q * kStepCount + kStepWidth] = kCells;
    }

    // queries for the FULL layout: [n_q][24][HD]
    std::vector<float> q_full(kNQ * 24 * HD);
    std::uniform_real_distribution<float> fq(-2.f, 2.f);
    for (auto& v : q_full) v = fq(rng);

    // device buffers
    uint16_t *dk = nullptr, *dv = nullptr;
    int32_t *dids = nullptr, *dsteps = nullptr, *dpt = nullptr;
    float *dq_full = nullptr;
    cudaMalloc(&dk, k_pool.size() * 2);
    cudaMalloc(&dv, v_pool.size() * 2);
    cudaMalloc(&dids, ids.size() * 4);
    cudaMalloc(&dsteps, steps.size() * 4);
    cudaMalloc(&dpt, page_table.size() * 4);
    cudaMalloc(&dq_full, q_full.size() * 4);
    cudaMemcpy(dk, k_pool.data(), k_pool.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v_pool.data(), v_pool.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dsteps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dpt, page_table.data(), page_table.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dq_full, q_full.data(), q_full.size() * 4, cudaMemcpyHostToDevice);

    QsaAttnPools pools;
    pools.k_pool = dk;
    pools.v_pool = dv;
    pools.page_table = dpt;

    QsaShapes s_full;
    s_full.n_head = 24;
    s_full.n_head_kv = 2;
    s_full.head_dim = HD;
    s_full.page_size = kPageSize;

    cudaStream_t st;
    cudaStreamCreate(&st);

    // the FULL call: attn [kNQ][24][HD]
    const uint64_t sc_full = qsa_decode_attn_scratch_floats(kCells, s_full);
    float* scratch_full = nullptr;
    cudaMalloc(&scratch_full, sc_full * 4 * kNQ);
    std::vector<float> attn_full(size_t(kNQ) * 24 * HD);
    float* dattn_full = nullptr;
    cudaMalloc(&dattn_full, attn_full.size() * 4);
    qsa_decode_attn_batch(dq_full, pools, dids, dsteps, kCells, s_full, scratch_full, dattn_full, kNQ, st);
    cudaStreamSynchronize(st);
    cudaMemcpy(attn_full.data(), dattn_full, attn_full.size() * 4, cudaMemcpyDeviceToHost);
    {   // debug: the full call's output range (zero output = the fixture itself is broken)
        float mn = 1e30f, mx = -1e30f;
        int zeros = 0;
        for (float v : attn_full) { mn = std::fmin(mn, v); mx = std::fmax(mx, v); if (v == 0.f) ++zeros; }
        std::printf("full attn: min %.4g max %.4g zeros %d/%zu\n", mn, mx, zeros, attn_full.size());
    }

    // the rank calls: q points at the rank's head window of the SAME full query buffer
    // (q + q_start*HD; the kernel's own q stride is n_head*HD per query, so the windowed
    // pointer needs a strided copy - simplest: copy the rank's rows into a compact buffer)
    int fails = 0;
    float* dscratch_r = nullptr;
    float* dattn_r = nullptr;
    float* dq_r = nullptr;
    {   // the largest scratch of the rank shapes (6 heads) + margin
        QsaShapes s6; s6.n_head = 6; s6.n_head_kv = 1; s6.head_dim = HD; s6.page_size = kPageSize;
        const uint64_t sc6 = qsa_decode_attn_scratch_floats(kCells, s6);
        cudaMalloc(&dscratch_r, sc6 * 4 * kNQ);
    }
    cudaMalloc(&dattn_r, size_t(kNQ) * 6 * HD * 4);
    cudaMalloc(&dq_r, size_t(kNQ) * 6 * HD * 4);
    std::vector<float> attn_r(size_t(kNQ) * 6 * HD);
    std::vector<float> q_r(size_t(kNQ) * 6 * HD);

    for (const Case& c : ranks) {
        // compact the rank's query rows: [n_q][n_head][HD] from the full buffer's window
        for (int q = 0; q < kNQ; ++q)
            std::memcpy(q_r.data() + size_t(q) * c.n_head * HD,
                        q_full.data() + size_t(q) * 24 * HD + size_t(c.q_start) * HD,
                        size_t(c.n_head) * HD * 4);
        cudaMemcpy(dq_r, q_r.data(), size_t(kNQ) * c.n_head * HD * 4, cudaMemcpyHostToDevice);
        QsaShapes s_r;
        s_r.n_head = c.n_head;
        s_r.n_head_kv = c.n_head_kv;
        s_r.head_dim = HD;
        s_r.page_size = kPageSize;
        // the rank's pool: its ONE kv head's rows.  The full pool is [page][2][4][HD];
        // rank kv0 (ranks with q_start 0/6 -> kv 0) reads head 0, kv1 head 1.  The row
        // formula (page*n_kv_heads + kvh)*page_size with n_kv_heads=1 gives page*4 - the
        // compact single-head pool.  Build it by strided copy.
        const int kvh = c.q_start / 12;   // 24/2: heads 0-11 -> kv0, 12-23 -> kv1
        std::vector<uint16_t> k_one(size_t(kPages) * kPageSize * HD), v_one(k_one.size());
        for (int pg = 0; pg < kPages; ++pg)
            for (int sl = 0; sl < kPageSize; ++sl) {
                const long long src_row = (long long) (pg * 2 + kvh) * kPageSize + sl;
                std::memcpy(k_one.data() + size_t(pg) * kPageSize * HD + size_t(sl) * HD,
                            k_pool.data() + size_t(src_row) * HD, HD * 2);
                std::memcpy(v_one.data() + size_t(pg) * kPageSize * HD + size_t(sl) * HD,
                            v_pool.data() + size_t(src_row) * HD, HD * 2);
            }
        cudaMemcpy(dk, k_one.data(), k_one.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, v_one.data(), v_one.size() * 2, cudaMemcpyHostToDevice);

        qsa_decode_attn_batch(dq_r, pools, dids, dsteps, kCells, s_r, dscratch_r, dattn_r, kNQ, st);
        cudaStreamSynchronize(st);
        std::fill(attn_r.begin(), attn_r.end(), -1e30f);
        cudaMemcpy(attn_r.data(), dattn_r, size_t(kNQ) * c.n_head * HD * 4, cudaMemcpyDeviceToHost);

        // compare: the rank's head h (global q_start+h) must equal the full call's row
        for (int q = 0; q < kNQ; ++q)
            for (int h = 0; h < c.n_head; ++h)
                for (int d = 0; d < HD; ++d) {
                    const float got = attn_r[size_t(q) * c.n_head * HD + size_t(h) * HD + d];
                    const float want = attn_full[size_t(q) * 24 * HD + size_t(c.q_start + h) * HD + d];
                    // bit for bit: same expression, same operands, same order
                    if (std::memcmp(&got, &want, 4) != 0) {
                        if (fails < 8)
                            std::fprintf(stderr, "q %d head %d (g %d) d %d: got %.9g want %.9g\\n",
                                         q, h, c.q_start + h, d, got, want);
                        ++fails;
                    }
                }
        // restore the full pools for the next rank's source rows
        cudaMemcpy(dk, k_pool.data(), k_pool.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, v_pool.data(), v_pool.size() * 2, cudaMemcpyHostToDevice);
        std::printf("rank {n_head %d, kv 1, q_start %d}: %s\\n", c.n_head, c.q_start,
                    fails == 0 ? "ok" : "FAIL");
        if (fails > 0) break;
    }

    cudaFree(dk); cudaFree(dv); cudaFree(dids); cudaFree(dsteps); cudaFree(dpt);
    cudaFree(dq_full); cudaFree(scratch_full); cudaFree(dattn_full);
    cudaFree(dscratch_r); cudaFree(dattn_r); cudaFree(dq_r);
    cudaStreamDestroy(st);
    const bool ok = fails == 0;
    std::printf("tp5_qsa_attn_test: %s (%d mismatches)\\n", ok ? "PASS" : "FAIL", fails);
    return ok ? 0 : 1;
}
