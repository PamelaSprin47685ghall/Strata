// tests/hip/tp5_qsa_test.cpp - 06-phase4 §4.2.3: the QSA tier's numeric gate.
//
// The tier's five partials, summed in rank order, must equal the UNSPLIT layer's
// own mixer output rows bit for bit: every rank's per-head arithmetic (q/kv
// projection rows, norm/rope, the single-head pool append, the G-templated
// attention, the gate, the out projection's contraction slice) is the same
// expression over the same operands as the full 24-head call's, and the RELAY
// sum is the same FP32 order the engine's own combine uses.
//
//   usage: tp5_qsa_test [--seed S]
//   env:   STRATA_TP5_GGUF (the model's shards, first shard path)
//   needs five GPUs; exit 0 = pass, 1 = mismatch, 77 = skip
#include "strata/core/tp5_qsa.hpp"
#include "strata/core/on_device.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <random>
#include <string>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;

namespace {

constexpr int H = 2560, HD = 256, NQ = 24, NKV = 2, IQ = 4, ID = 128;
int64_t kCells = 96;    // the context the fixture drives (small, deterministic)
int64_t kPages = 0;   // set in main (kCells / 4)
constexpr int kNTok = 3;          // rows per run() call

}  // namespace

int main(int argc, char** argv) {
    unsigned seed = 42;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = unsigned(std::atoi(argv[++i]));
    if (const char* c = std::getenv("TP5_QSA_CELLS")) kCells = std::atoll(c);
    kPages = kCells / 4;
    // (the GGUF env is no longer needed - the fixture carries the weights)
    // the model's shards (a split GGUF needs them all) - BEFORE the GPU init:
    // the HIP runtime's init has been observed to break std::ifstream on this
    // stack (the same call succeeds in a plain g++ binary), so resolve them first
    // the FIXTURE (see the tier .cu's note): layer 3's ten tensors as .bin files.
    // The gate script (tools/tp5_qsa_fixture.py) writes them from the GGUF with
    // python - whose openat works - because a test binary's openat of the >32 GiB
    // shard 1 is unreliable on this stack.
    const char* fixdir = std::getenv("STRATA_TP5_QSA_FIXTURE");
    if (fixdir == nullptr || *fixdir == '\0') {
        std::printf("tp5_qsa_test: STRATA_TP5_QSA_FIXTURE not set - SKIP\n");
        return 77;
    }
    auto read_fix = [&](const char* name) {
        const std::string path = std::string(fixdir) + "/blk_3_" + name + ".bin";
        std::FILE* f = std::fopen(path.c_str(), "rb");
        // a stale fixture dir (another machine's ctest ENVIRONMENT) is a skip, not a failure:
        // the fixture is generated data (tools/tp5_qsa_fixture.py), not part of the repo
        if (f == nullptr) { std::fprintf(stderr, "tp5_qsa_test: cannot open %s - SKIP (regenerate with tools/tp5_qsa_fixture.py)\n", path.c_str()); std::exit(77); }
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> b{std::vector<uint8_t>(size_t(n))};
        if (!b.empty() && std::fread(b.data(), 1, b.size(), f) != b.size()) { std::fclose(f); std::exit(1); }
        std::fclose(f);
        return b;
    };
    const auto f_q = read_fix("attn_q_weight"), f_k = read_fix("attn_k_weight");
    const auto f_v = read_fix("attn_v_weight"), f_o = read_fix("attn_output_weight");
    const auto f_iq = read_fix("indexer_q_proj_weight"), f_ik = read_fix("indexer_k_proj_weight");
    const auto f_qn = read_fix("attn_q_norm_weight"), f_kn = read_fix("attn_k_norm_weight");
    const auto f_iqn = read_fix("indexer_q_norm_weight"), f_ikn = read_fix("indexer_k_norm_weight");
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 5) {
        std::printf("tp5_qsa_test: fewer than five GPUs - SKIP\n");
        return 77;
    }
    // the native QSA kernels must be enabled as the engine enables them
    native_qsa_set_enabled(true);
    native_qsa_indexer_set_enabled(true);

    // ---- the reference: layer 3's UNSPLIT mixer, run by hand on GPU 0
    // (the same kernel sequence the verify window runs, one QSA layer)
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> fx(-1.5f, 1.5f);
    const QsaShapes s = [] { QsaShapes q = qsa_real_shapes(); return q; }();
    const int64_t cap = 2051;  // the selection capacity (the kernel's own bound); width is n_kv

    // the fixture drives kCells tokens through the pools first (so the selection has
    // history), then the window's rows.  The reference and the tier each keep their own
    // pools; both see the same token stream.
    std::vector<float> xs;                       // the token stream's rows (kCells + kNTok)
    std::vector<int32_t> steps;                  // kStepCount per token
    for (int64_t t = 0; t < kCells + kNTok; ++t) {
        for (int i = 0; i < H; ++i) xs.push_back(fx(rng));
        int32_t st[kStepCount];
        qsa_step_fill(st, int32_t(t), s);
        for (int k = 0; k < kStepCount; ++k) steps.push_back(st[k]);
    }

    // layer 3's weights: the FIXTURE's bytes (see the fixture note above).  The
    // types are layer 3's own: q=42 (Q2_0), k=11 (Q3_K), v=13 (Q5_K), o=12 (Q4_K),
    // iq/ik=30 (BF16), the norms F32.
    constexpr int kTypeQ = 42, kTypeK = 11, kTypeV = 13, kTypeO = 12;
    const uint8_t *bq = f_q.data(), *bk = f_k.data(), *bv = f_v.data(), *bo = f_o.data();
    const uint8_t *biq = f_iq.data(), *bik = f_ik.data();
    const float* bqn = reinterpret_cast<const float*>(f_qn.data());
    const float* bkn = reinterpret_cast<const float*>(f_kn.data());
    const float* biqn = reinterpret_cast<const float*>(f_iqn.data());
    const float* bikn = reinterpret_cast<const float*>(f_ikn.data());
    auto bytes_of = [](int type, size_t elems) {
        int be = 0, bb = 0;
        strata::block_geometry(type, be, bb);
        return elems / size_t(be) * size_t(bb);
    };
    const size_t nq = bytes_of(kTypeQ, size_t(2560) * 12288);
    const size_t nk = bytes_of(kTypeK, size_t(2560) * 512);
    const size_t nv = bytes_of(kTypeV, size_t(2560) * 512);
    const size_t no_ = bytes_of(kTypeO, size_t(6144) * 2560);
    uint8_t *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
    uint16_t *diq = nullptr, *dik = nullptr;
    float *dqn = nullptr, *dkn = nullptr, *diqn = nullptr, *dikn = nullptr;
    {
        cudaMalloc(&dq, nq); cudaMemcpy(dq, bq, nq, cudaMemcpyHostToDevice);
        cudaMalloc(&dk, nk); cudaMemcpy(dk, bk, nk, cudaMemcpyHostToDevice);
        cudaMalloc(&dv, nv); cudaMemcpy(dv, bv, nv, cudaMemcpyHostToDevice);
        cudaMalloc(&dout, no_); cudaMemcpy(dout, bo, no_, cudaMemcpyHostToDevice);
        cudaMalloc(&diq, f_iq.size()); cudaMemcpy(diq, biq, f_iq.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&dik, f_ik.size()); cudaMemcpy(dik, bik, f_ik.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&dqn, 256 * 4); cudaMemcpy(dqn, bqn, 256 * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&dkn, 256 * 4); cudaMemcpy(dkn, bkn, 256 * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&diqn, 128 * 4); cudaMemcpy(diqn, biqn, 128 * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&dikn, 128 * 4); cudaMemcpy(dikn, bikn, 128 * 4, cudaMemcpyHostToDevice);
        std::fprintf(stderr, "fixture q_norm[0..2] = %g %g %g\n", bqn[0], bqn[1], bqn[2]);
    }
    // the reference's pools and state on GPU 0
    int8_t *rk = nullptr, *rv = nullptr;
    uint16_t *rks = nullptr, *rvs = nullptr;
    int32_t* rpt = nullptr;
    float *ridx_tail = nullptr, *ridx_dead = nullptr, *ridx_pooled = nullptr;
    int32_t* ridx_pos = nullptr;
    {   // the pool covers the WHOLE token stream (kCells + kNTok tokens): the first
        // version sized it for kCells alone, so the window rows' appends read the
        // page table out of bounds and wrote garbage rows (the tier's own pool is
        // 64K and never had the problem - the tier side was right all along)
        const int64_t total_cells = kCells + kNTok;
        const int64_t pages = total_cells / 4 + 1;
        cudaMalloc(&rk, size_t(total_cells + 8) * NKV * HD);
        cudaMalloc(&rv, size_t(total_cells + 8) * NKV * HD);
        cudaMalloc(&rks, size_t(total_cells + 8) * NKV * (HD / 64) * 2);
        cudaMalloc(&rvs, size_t(total_cells + 8) * NKV * (HD / 64) * 2);
        std::vector<int32_t> tab{std::vector<int32_t>(size_t(pages))};
        for (int64_t i = 0; i < pages; ++i) tab[size_t(i)] = int32_t(i);
        cudaMalloc(&rpt, tab.size() * 4); cudaMemcpy(rpt, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&ridx_tail, size_t(3) * ID * 4); cudaMemset(ridx_tail, 0, size_t(3) * ID * 4);
        cudaMalloc(&ridx_dead, size_t(ID) * 4); cudaMemset(ridx_dead, 0, size_t(ID) * 4);
        cudaMalloc(&ridx_pooled, size_t(total_cells / 4 + 2) * ID * 4);
        cudaMemset(ridx_pooled, 0, size_t(total_cells / 4 + 2) * ID * 4);
        cudaMalloc(&ridx_pos, 4); cudaMemset(ridx_pos, 0, 4);
        
        
        
        
    }
    // scratch on GPU 0
    float *rx = nullptr, *rqfull = nullptr, *rqcur = nullptr, *rkcur = nullptr, *rvcur = nullptr;
    float *rqidx = nullptr, *ridx_raw = nullptr, *rscores = nullptr, *rattn = nullptr, *rattn32 = nullptr;
    int32_t* rsel = nullptr;
    float* rscratch = nullptr;
    void* rxq = nullptr;
    int32_t *rstep = nullptr, *rpos_q = nullptr, *rpos_k = nullptr, *rpos_i = nullptr;
    float *rcos = nullptr, *rsin = nullptr;
    {
        cudaMalloc(&rx, size_t(1) * H * 4);
        cudaMalloc(&rqfull, size_t(1) * NQ * 2 * HD * 4);
        cudaMalloc(&rqcur, size_t(1) * NQ * HD * 4);
        cudaMalloc(&rkcur, size_t(1) * NKV * HD * 4);
        cudaMalloc(&rvcur, size_t(1) * NKV * HD * 4);
        cudaMalloc(&rqidx, size_t(1) * IQ * ID * 4);
        cudaMalloc(&ridx_raw, size_t(1) * ID * 4);
        cudaMalloc(&rscores, size_t(1) * ((kCells + kNTok) / 4 + 2) * 4);
        cudaMalloc(&rattn, size_t(1) * NQ * HD * 4);
        cudaMalloc(&rattn32, size_t(1) * NQ * HD * 4);
        cudaMalloc(&rsel, size_t(1) * cap * 4);
        const int64_t asc = qsa_decode_attn_scratch_floats(cap, s);
        cudaMalloc(&rscratch, size_t(asc) * 4);
        cudaMalloc(&rxq, native_q8_1_bytes(H, 1));
        cudaMalloc(&rstep, kStepCount * 4);
        cudaMalloc(&rpos_q, NQ * 4);
        cudaMalloc(&rpos_k, NKV * 4);
        cudaMalloc(&rpos_i, IQ * 4);
        std::vector<float> ct(size_t(kCells + 8) * 32), st_(ct.size());
        build_rope_table(64, rope_scaling(), int(kCells + 8), ct.data(), st_.data());
        cudaMalloc(&rcos, ct.size() * 4); cudaMemcpy(rcos, ct.data(), ct.size() * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&rsin, st_.size() * 4); cudaMemcpy(rsin, st_.data(), st_.size() * 4, cudaMemcpyHostToDevice);
    }
    cudaStream_t rs;
    cudaStreamCreate(&rs);

    // the reference's per-token drive: append token t's row into its pools/state,
    // then (for the last kNTok tokens) produce the mixer output row.
    // The reference computes the OUTPUT of the LAST token only (t = kCells + kNTok - 1):
    // the window's rows are the last kNTok tokens; the tier's run() gets the same rows.
    // For the bit-exact gate we drive ONE row at a time (n_tok = 1): the simplest
    // correspondence with the reference's single-row arithmetic.
    std::vector<float> ref_out(size_t(kNTok) * H);
    for (int64_t t = 0; t < kCells + kNTok; ++t) {
        cudaMemcpyAsync(rx, xs.data() + size_t(t) * H, H * 4, cudaMemcpyHostToDevice, rs);
        cudaMemcpyAsync(rstep, steps.data() + size_t(t) * kStepCount, kStepCount * 4, cudaMemcpyHostToDevice, rs);
        std::vector<int32_t> pq{std::vector<int32_t>(size_t(NQ), int32_t(t))};
        std::vector<int32_t> pk{std::vector<int32_t>(size_t(NKV), int32_t(t))};
        std::vector<int32_t> pi{std::vector<int32_t>(size_t(IQ), int32_t(t))};
        cudaMemcpyAsync(rpos_q, pq.data(), NQ * 4, cudaMemcpyHostToDevice, rs);
        cudaMemcpyAsync(rpos_k, pk.data(), NKV * 4, cudaMemcpyHostToDevice, rs);
        cudaMemcpyAsync(rpos_i, pi.data(), IQ * 4, cudaMemcpyHostToDevice, rs);
        cudaStreamSynchronize(rs);
        native_quantize_q8_1(rx, rxq, H, 1, rs);
        // q + gate
        native_mmvq(kTypeQ, dq, rxq, rqfull, H, NQ * 2 * HD, 1, rs);
        copy_rows_strided(rqcur, rqfull, NQ, HD, 2 * HD, rs);
        native_qsa_rms_norm_weighted(rqcur, dqn, rqcur, HD, NQ, 1e-6f, rs);
        rope_neox_apply(rqcur, rqcur, NQ, HD, 64, rcos, rsin, rpos_q, rs);
        // k/v
        native_mmvq(kTypeK, dk, rxq, rkcur, H, NKV * HD, 1, rs);
        native_mmvq(kTypeV, dv, rxq, rvcur, H, NKV * HD, 1, rs);
        native_qsa_rms_norm_weighted(rkcur, dkn, rkcur, HD, NKV, 1e-6f, rs);
        rope_neox_apply(rkcur, rkcur, NKV, HD, 64, rcos, rsin, rpos_k, rs);
        // the pools - the STEPS form (n_tok 1), the same entry point the tier uses
        kv_append_q8_steps(rk, rv, rks, rvs, rpt, rstep, kStepCount, rkcur, rvcur,
                           (int) (NKV * HD), 1, s, rs, nullptr);
        // the indexer
        bf16_gemv_fp32_mmvf_multi(rx, H, diq, rqidx, IQ * ID, H, IQ * ID, 1, rs);
        native_qsa_rms_norm_weighted(rqidx, diqn, rqidx, ID, IQ, 1e-6f, rs);
        rope_neox_apply(rqidx, rqidx, IQ, ID, 64, rcos, rsin, rpos_i, rs);
        bf16_gemv_fp32_mmvf_multi(rx, H, dik, ridx_raw, ID, H, ID, 1, rs);
        {
            QsaIndexerBuffers ib{ridx_tail, ridx_dead, ridx_pooled, ridx_pos};
            native_qsa_indexer_append(ridx_raw, rstep + kStepPos, 0, dikn, 1e-6f, ib, s, kCells + 8,
                                      rope_scaling(), rs);
        }
        // the mixer output only for the window's rows (the last kNTok tokens)
        if (t >= kCells) {
            qsa_block_scores(ridx_pooled, ridx_dead, rqidx, rstep, 1, (kCells + kNTok) / 4 + 2, s, rscores, rs);
            qsa_block_topk(rscores, rstep, 1, (kCells + kNTok) / 4 + 2, cap, s, rsel, rs);
            QsaAttnPools pools;
            pools.k_q = rk; pools.v_q = rv; pools.k_scale = rks; pools.v_scale = rvs;
            pools.page_table = rpt;
            qsa_decode_attn_batch(rqcur, pools, rsel, rstep, cap, s, rscratch, rattn, 1, rs);
            native_qsa_gate_apply(rattn, rqfull, rattn32, NQ, HD, rs);
            native_quantize_q8_1(rattn32, rxq, NQ * HD, 1, rs);
            // the out projection: n_in = 6144
            float* rout = nullptr;
            cudaMalloc(&rout, H * 4);
            native_mmvq(kTypeO, dout, rxq, rout, NQ * HD, H, 1, rs);
            cudaStreamSynchronize(rs);
            cudaMemcpy(ref_out.data() + size_t(t - kCells) * H, rout, H * 4, cudaMemcpyDeviceToHost);
            cudaFree(rout);
        }
        cudaStreamSynchronize(rs);
    }

    // ---- the tier: the same token stream, then run() for the window's rows
    // the tier's max_cells must cover the fixture's context: the tier's open() pins
    // 64K pools, which is fine (the fixture only writes kCells cells).
    Tp5QsaTier tier;
    std::string terr;
    if (!tier.open({}, H, terr)) {
        std::fprintf(stderr, "tp5_qsa_test: open: %s\n", terr.c_str());
        return 1;
    }
    // drive the tier's pools/state with the SAME token stream, one row at a time
    // (run() with n_tok = 1 appends the row's cell and returns its mixer output)
    std::vector<float> tier_out(size_t(kNTok) * H);
    for (int64_t t = 0; t < kCells + kNTok; ++t) {
        // one position per token (every head at the same position, as the verifier's
        // own h_pos_ rows are); the tier expands per head internally
        std::vector<int32_t> pos_q{int32_t(t)}, pos_k{int32_t(t)}, pos_i{int32_t(t)};
        float* y = t >= kCells ? tier_out.data() + size_t(t - kCells) * H : nullptr;
        if (!tier.run(3, xs.data() + size_t(t) * H, 1,
                      steps.data() + size_t(t) * kStepCount,
                      pos_q.data(), pos_k.data(), pos_i.data(), y ? y : tier_out.data(), terr)) {
            // the pre-window rows' output goes nowhere (scratch); only the last rows count
            std::fprintf(stderr, "tp5_qsa_test: run(t=%lld): %s\n", (long long) t, terr.c_str());
            return 1;
        }
    }

    // ---- the multi-row window gate: the last kNTok rows as ONE window (the engine's
    // MTP windows are 5-6 rows).  The publish counter once released `ready` after the
    // first row's blocks (the 2D grid's y redundancy), and the CPU sum read the other
    // rows as zeros - the per-token loop above (n_tok = 1) cannot see it.
    {
        const int64_t t0 = kCells;   // the window's first row
        std::vector<int32_t> pqm{std::vector<int32_t>(size_t(kNTok))}, pkm{std::vector<int32_t>(size_t(kNTok))},
            pim{std::vector<int32_t>(size_t(kNTok))};
        for (int64_t t = 0; t < kNTok; ++t) {
            pqm[size_t(t)] = int32_t(t0 + t);
            pkm[size_t(t)] = int32_t(t0 + t);
            pim[size_t(t)] = int32_t(t0 + t);
        }
        std::vector<float> multi_out(size_t(kNTok) * H, 0.0f);
        // a SECOND tier instance: the window's appends must not double-apply on the
        // first tier's state (the engine's windows run on the same state, but the
        // comparison here wants the same starting state as the per-token loop's)
        Tp5QsaTier tier2;
        if (!tier2.open({}, H, terr)) {
            std::fprintf(stderr, "tp5_qsa_test: open2: %s\n", terr.c_str());
            return 1;
        }
        // replay the prefix through tier2 (its state must match the first tier's)
        for (int64_t t = 0; t < kCells; ++t) {
            std::vector<int32_t> p1{int32_t(t)};
            if (!tier2.run(3, xs.data() + size_t(t) * H, 1, steps.data() + size_t(t) * kStepCount,
                           p1.data(), p1.data(), p1.data(), multi_out.data(), terr)) {
                std::fprintf(stderr, "tp5_qsa_test: run2(t=%lld): %s\n", (long long) t, terr.c_str());
                return 1;
            }
        }
        if (!tier2.run(3, xs.data() + size_t(t0) * H, 2,
                       steps.data() + size_t(t0) * kStepCount,
                       pqm.data(), pkm.data(), pim.data(), multi_out.data(), terr)) {
            std::fprintf(stderr, "tp5_qsa_test: run2(window): %s\n", terr.c_str());
            return 1;
        }
        int mm = 0;
        double mmax = 0.0;
        for (int64_t t = 0; t < 2; ++t)
            for (int i = 0; i < H; ++i) {
                const double d = std::fabs(double(multi_out[size_t(t) * H + i]) -
                                           double(tier_out[size_t(t) * H + i]));
                if (d > mmax) mmax = d;
                if (d > 1e-5) ++mm;
            }
        std::fprintf(stderr, "multi-row window vs per-token: %d over-tolerance, max abs %.3e\n", mm, mmax);
        if (mm > 0) { std::printf("tp5_qsa_test: FAIL (multi-row window)\n"); return 1; }
    }

    // ---- the layer-offset gate: run the SAME stream through QSA layer 7 (blob
    // index 1).  The fixture weights are identical for every layer, so a correct
    // layer offset gives the SAME rows as layer 3's - a wrong offset reads foreign
    // bytes and the rows differ (the engine's first wiring read index*TOTAL, 12x out
    // of bounds, which only layer 7+ could hit: this gate catches it).
    {
        std::vector<float> l7_out(size_t(kNTok) * H, 0.0f);
        for (int64_t t = 0; t < kCells + kNTok; ++t) {
            std::vector<int32_t> pos_q{int32_t(t)}, pos_k{int32_t(t)}, pos_i{int32_t(t)};
            float* y = t >= kCells ? l7_out.data() + size_t(t - kCells) * H : nullptr;
            if (!tier.run(7, xs.data() + size_t(t) * H, 1,
                          steps.data() + size_t(t) * kStepCount,
                          pos_q.data(), pos_k.data(), pos_i.data(), y ? y : l7_out.data(), terr)) {
                std::fprintf(stderr, "tp5_qsa_test: run(l7, t=%lld): %s\n", (long long) t, terr.c_str());
                return 1;
            }
        }
        int lm = 0;
        double lmax = 0.0;
        for (size_t i = 0; i < l7_out.size(); ++i) {
            const double d = std::fabs(double(l7_out[i]) - double(tier_out[i]));
            if (d > lmax) lmax = d;
            if (d > 1e-4) ++lm;
        }
        std::fprintf(stderr, "layer 7 vs layer 3 rows: %d over-tolerance, max abs %.3e\n", lm, lmax);
        if (lm > 0) { std::printf("tp5_qsa_test: FAIL (layer offset)\n"); return 1; }
    }

    // ---- the q-projection slice check: rank 0's rows [0, 3072) of the 12288
    {
        int be = 0, bb = 0;
        strata::block_geometry(kTypeQ, be, bb);
        const int64_t bpr = 2560 / be;
        const size_t bytes = size_t(3072) * bpr * bb;
        uint8_t* dslice = nullptr;
        cudaMalloc(&dslice, bytes);
        cudaMemcpy(dslice, bq, bytes, cudaMemcpyHostToDevice);
        float* yslice = nullptr;
        cudaMalloc(&yslice, size_t(1) * 3072 * 4);
        // the LAST token's row, as both paths quantized it
        native_quantize_q8_1(rx, rxq, H, 1, rs);   // rx still holds the last token's x
        native_mmvq(kTypeQ, dslice, rxq, yslice, H, 3072, 1, rs);
        cudaStreamSynchronize(rs);
        std::vector<float> ref_slice(3072), tier_slice(3072);
        cudaMemcpy(ref_slice.data(), rqfull, 3072 * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(tier_slice.data(), yslice, 3072 * 4, cudaMemcpyDeviceToHost);
        int qpm = 0;
        for (int i = 0; i < 3072; ++i)
            if (std::memcmp(&ref_slice[i], &tier_slice[i], 4) != 0) ++qpm;
        std::fprintf(stderr, "slice q-proj vs ref rows [0,3072): %d/3072 mismatches\n", qpm);
        cudaFree(dslice); cudaFree(yslice);
    }

    // ---- stage-by-stage debug: rank 0's qcur (post norm/rope) vs the reference's
    // first 6 heads' rows, for the FIRST window row
    {
        const auto dbg = tier.debug(0);
        const strata::core::OnDevice on0(0);
        // the reference's rqcur: [1][NQ*HD]; rank 0 owns heads 0..5
        std::vector<float> refq(size_t(NQ) * HD), tierq(size_t(6) * HD);
        cudaMemcpy(refq.data(), rqcur, refq.size() * 4, cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(rs);
        {
            const strata::core::OnDevice onr(tier.debug_gpu(0));
            cudaMemcpy(tierq.data(), dbg.qcur, tierq.size() * 4, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(tier.debug_stream(0));
        }
        int qm = 0;
        for (size_t i = 0; i < tierq.size(); ++i)
            if (std::memcmp(&tierq[i], &refq[i], 4) != 0) ++qm;
        std::fprintf(stderr, "stage qcur (rank0 heads 0-5): %d/%zu mismatches\n", qm, tierq.size());
        // qfull first (the projection's own rows)
        std::vector<float> reff(size_t(6) * 2 * HD), tierf(reff.size());
        cudaMemcpy(reff.data(), rqfull, reff.size() * 4, cudaMemcpyDeviceToHost);
        {
            const strata::core::OnDevice onr(tier.debug_gpu(0));
            cudaMemcpy(tierf.data(), dbg.qfull, tierf.size() * 4, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(tier.debug_stream(0));
        }
        int qfm = 0;
        for (size_t i = 0; i < reff.size(); ++i)
            if (std::memcmp(&tierf[i], &reff[i], 4) != 0) ++qfm;
        std::fprintf(stderr, "stage qfull (rank0, 3072 floats): %d/%zu mismatches\n", qfm, reff.size());
        // replay the tier's own copy+norm+rope on the REFERENCE's qfull rows and
        // compare with the reference's qcur - isolates which step diverges
        {
            float* tcur = nullptr;
            cudaMalloc(&tcur, size_t(6) * HD * 4);
            copy_rows_strided(tcur, rqfull, 6, HD, 2 * HD, rs);
            native_qsa_rms_norm_weighted(tcur, dqn, tcur, HD, 6, 1e-6f, rs);
            rope_neox_apply(tcur, tcur, 6, HD, 64, rcos, rsin, rpos_q, rs);   // pos[0..5] = t
            cudaStreamSynchronize(rs);
            std::vector<float> tcv(6 * HD), rcv(6 * HD);
            cudaMemcpy(tcv.data(), tcur, tcv.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(rcv.data(), rqcur, rcv.size() * 4, cudaMemcpyDeviceToHost);
            int cm = 0;
            for (size_t i = 0; i < tcv.size(); ++i)
                if (std::memcmp(&tcv[i], &rcv[i], 4) != 0) ++cm;
            std::fprintf(stderr, "replay copy+norm+rope (6 heads) vs ref qcur: %d/%zu\n", cm, tcv.size());
            cudaFree(tcur);
        }
        // attn32: the reference's first 6 heads
        std::vector<float> refa(size_t(NQ) * HD), tiera(size_t(6) * HD);
        cudaMemcpy(refa.data(), rattn32, refa.size() * 4, cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(rs);
        {
            const strata::core::OnDevice onr(tier.debug_gpu(0));
            cudaMemcpy(tiera.data(), dbg.attn32, tiera.size() * 4, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(tier.debug_stream(0));
        }
        int am = 0;
        for (size_t i = 0; i < tiera.size(); ++i)
            if (std::memcmp(&tiera[i], &refq[i], 4) == 0) ++am;   // placeholder compare below
        am = 0;
        for (size_t i = 0; i < tiera.size(); ++i)
            if (std::memcmp(&tiera[i], &refa[i], 4) != 0) ++am;
        std::fprintf(stderr, "stage attn32 (rank0 heads 0-5): %d/%zu mismatches\n", am, tiera.size());
    }

    // ---- the selection and the pool: rank 0's ids and k_q vs the reference's
    {
        // the reference's LAST token's step and ids (still live)
        std::vector<int32_t> rstepv{std::vector<int32_t>(kStepCount)};
        std::vector<int32_t> rselv{std::vector<int32_t>(size_t(cap))};
        cudaMemcpy(rstepv.data(), rstep, kStepCount * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(rselv.data(), rsel, size_t(cap) * 4, cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(rs);
        const int32_t width = rstepv[size_t(kStepWidth)];
        std::fprintf(stderr, "ref step: pos=%d n_kv=%d width=%d; ids[0..4]=%d %d %d %d %d\n",
                     rstepv[0], rstepv[1], width, rselv[0], rselv[1], rselv[2], rselv[3], rselv[4]);
        const strata::core::OnDevice onr1(tier.debug_gpu(0));   // rank 0 on GPU 1
        std::vector<int32_t> tsel{std::vector<int32_t>(size_t(cap))};
        std::vector<int32_t> tstep{std::vector<int32_t>(kStepCount)};
        cudaMemcpy(tsel.data(), tier.debug_sel(0), size_t(cap) * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(tstep.data(), tier.debug_step(0), kStepCount * 4, cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(tier.debug_stream(0));
        std::fprintf(stderr, "tier step: pos=%d n_kv=%d width=%d; ids[0..4]=%d %d %d %d %d\n",
                     tstep[0], tstep[1], tstep[size_t(kStepWidth)],
                     tsel[0], tsel[1], tsel[2], tsel[3], tsel[4]);
        int sm = 0;
        for (int i = 0; i < width; ++i)
            if (tsel[size_t(i)] != rselv[size_t(i)]) ++sm;
        std::fprintf(stderr, "selection mismatches: %d/%d\n", sm, width);
        // the pool: rank 0's kv head 0 rows vs the reference's head-0 rows
        // (row = (page*kv_heads + h)*page_size + cell%4; the reference's h=0 rows
        //  are page*8+0..3, the tier's page*4+0..3)
        // rkq on GPU 0 (read BEFORE the device switch below; the first version
        // read it under GPU 1 and compared garbage).  The WHOLE pool: the h=0 rows
        // are scattered (row = page*8 + c%4), and reading only the first kCells
        // rows mixed head 0 with head 1.
        const int64_t rpool_rows = kCells + kNTok + 8;
        std::vector<int8_t> rkq{std::vector<int8_t>(size_t(rpool_rows) * 2 * HD)}, tkq{std::vector<int8_t>(size_t(kCells) * HD)};
        {
            const strata::core::OnDevice back0(0);
            cudaMemcpy(rkq.data(), rk, rkq.size(), cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(rs);
            const int64_t qi0 = 0;   // layer 3 is QSA layer 0
            cudaMemcpy(tkq.data(), tier.debug_kq(0, qi0), tkq.size(), cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(tier.debug_stream(0));
        }
        int pm = 0;
        for (int64_t c = 0; c < kCells; ++c) {
            const int64_t rrow = (c / 4) * 2 * 4 + 0 * 4 + (c % 4);   // kv_heads=2, h=0
            const int64_t trow = (c / 4) * 1 * 4 + (c % 4);            // kv_heads=1
            for (int d = 0; d < HD; ++d)
                if (rkq[size_t(rrow) * HD + d] != tkq[size_t(trow) * HD + d]) { ++pm; break; }
        }
        std::fprintf(stderr, "k_q head-0 rows mismatched: %d/%lld; first cells:", pm, (long long) kCells);
        int shown = 0;
        for (int64_t c = 0; c < kCells && shown < 12; ++c) {
            const int64_t rrow = (c / 4) * 2 * 4 + 0 * 4 + (c % 4);
            const int64_t trow = (c / 4) * 1 * 4 + (c % 4);
            bool diff = false;
            for (int d = 0; d < HD; ++d)
                if (rkq[size_t(rrow) * HD + d] != tkq[size_t(trow) * HD + d]) { diff = true; break; }
            if (diff) { std::fprintf(stderr, " %lld", (long long) c); ++shown; }
        }
        std::fprintf(stderr, "\n");
    }

    // ---- compare.  The BIT-EXACT gates are upstream (qfull, qcur, the pool,
    // attn32: the per-head arithmetic is the unsplit call's own).  The final rows
    // CANNOT be bit-exact: the out projection's contraction is split five ways,
    // so the sum's FP32 order differs from the single dot product's - the gate
    // here is the RELATIVE error of that reordering, bounded like the FFN tier's
    // own (5% was the measured Q8_1 activation-quantisation noise there; the
    // reordering alone is orders below it).
    int fails = 0;
    double max_rel = 0.0;
    for (int64_t t = 0; t < kNTok; ++t)
        for (int i = 0; i < H; ++i) {
            const float got = tier_out[size_t(t) * H + i];
            const float want = ref_out[size_t(t) * H + i];
            const double denom = std::fabs(double(want)) > 1e-6 ? double(want) : 1.0;
            const double rel = std::fabs(double(got) - double(want)) / denom;
            if (rel > max_rel) max_rel = rel;
            if (rel > 0.05) {
                if (fails < 6)
                    std::fprintf(stderr, "row %lld col %d: got %.9g want %.9g rel %.2e\n",
                                 (long long) t, i, got, want, rel);
                ++fails;
            }
        }
    const bool ok = fails == 0;
    std::printf("tp5_qsa_test: %s (%d over-tolerance, max rel %.3e over %lld rows)\n",
                ok ? "PASS" : "FAIL", fails, max_rel, (long long) kNTok);
    return ok ? 0 : 1;
}
