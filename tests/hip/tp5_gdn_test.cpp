// tests/hip/tp5_gdn_test.cpp - 06-phase4 §4.2.3: the GDN tier's numeric gate.
//
// The tier's five partials, summed in rank order, must equal the UNSPLIT layer's
// The tier's five slices, gathered in rank order, must equal the UNSPLIT layer's
// own mixer output rows: the rank's channel-gathered qkv projection rows, the
// conv, a/b, the gate projection, the recurrence (h_k = v_count makes
// `head % h_k` the identity - the shared kernel unchanged) are the same
// expressions over the same operands as the full 10240-channel call's, and the
// RELAY gather is a disjoint copy, so every stage is bit for bit.  The OUT
// projection is NOT the tier's - it runs once on GPU0 over the gathered full y
// (the engine's own bit path); the tier never projects bo.
//
// The export gate (export_state_span): after a real window run and a real
// commit, the ranks' state must scatter back into the engine-shaped arena
// exactly as the UNSPLIT reference's own advanced state - the inverse of
// sync_state_span's gather, checked against an independently computed
// full-width reference, never a sync/export copy roundtrip.
//
//   usage: tp5_gdn_test [--seed S]
//   env:   STRATA_TP5_GDN_FIXTURE (the fixture dir)
//   needs five GPUs; exit 0 = pass, 1 = mismatch, 77 = skip
#include "strata/core/tp5_gdn.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;

namespace {

constexpr int H = 2560, S = 128, HK = 16, HV = 48, C = 2 * S * HK + S * HV;
constexpr int kVStart[5] = {0, 10, 20, 30, 40};
constexpr int kVCount[5] = {10, 10, 10, 10, 8};
int kNTok = 3;                // rows per run() call (TP5_GDN_NTOK=1 isolates the first step)
constexpr int kMaxT = 8;
constexpr float EPS = 1e-6f;

}  // namespace

int main(int argc, char** argv) {
    unsigned seed = 42;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = unsigned(std::atoi(argv[++i]));
    if (const char* v = std::getenv("TP5_GDN_NTOK")) kNTok = std::atoi(v);
    if (kNTok < 1 || kNTok > 8) { std::fprintf(stderr, "tp5_gdn_test: bad TP5_GDN_NTOK\n"); return 1; }
    const bool no_sync_gate = std::getenv("TP5_GDN_NOSYNC") != nullptr;
    const char* fixdir = std::getenv("STRATA_TP5_GDN_FIXTURE");
    const bool use_gguf = (fixdir == nullptr || *fixdir == '\0') || std::getenv("TP5_GDN_CMPFIXTURE") != nullptr;
    if (use_gguf && std::getenv("STRATA_TP5_GGUF") == nullptr) {
        std::printf("tp5_gdn_test: STRATA_TP5_GDN_FIXTURE (or STRATA_TP5_GGUF) not set - SKIP\n");
        return 77;
    }
    auto read_fix = [&](const char* name) {
        if (use_gguf && (fixdir == nullptr || *fixdir == '\0')) return std::vector<uint8_t>{};
        const std::string path = std::string(fixdir ? fixdir : "") + "/blk_1_" + name + ".bin";
        std::FILE* f = std::fopen(path.c_str(), "rb");
        // a stale fixture dir (another machine's ctest ENVIRONMENT) is a skip, not a failure:
        // the fixture is generated data (tools/tp5_gdn_fixture.py), not part of the repo
        if (f == nullptr) { std::fprintf(stderr, "tp5_gdn_test: cannot open %s - SKIP (regenerate with tools/tp5_gdn_fixture.py)\n", path.c_str()); std::exit(77); }
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> b{std::vector<uint8_t>(size_t(n < 0 ? 0 : n))};
        if (!b.empty() && std::fread(b.data(), 1, b.size(), f) != b.size()) { std::fclose(f); return std::vector<uint8_t>{}; }
        std::fclose(f);
        return b;
    };
    const auto f_qkv = read_fix("attn_qkv_weight"), f_g = read_fix("attn_gate_weight"),
               f_a = read_fix("ssm_alpha_weight"),
               f_b = read_fix("ssm_beta_weight"), f_c = read_fix("ssm_conv1d_weight"),
               f_nm = read_fix("ssm_norm_weight"), f_dt = read_fix("ssm_dt_bias"),
               f_sa = read_fix("ssm_a");
    if (!use_gguf &&
        (f_qkv.empty() || f_g.empty() || f_a.empty() || f_b.empty() ||
         f_c.empty() || f_nm.empty() || f_dt.empty() || f_sa.empty())) {
        std::fprintf(stderr, "tp5_gdn_test: the fixture is incomplete\n");
        return 1;
    }
    // the GGUF mode: layer 1's nine tensors straight from the model (the engine's
    // own source); the fixture bytes above are then the empty stand-ins
    std::vector<uint8_t> g_qkv, g_g, g_a, g_b, g_c, g_nm, g_dt, g_sa;
    int g_type_qkv = -1, g_type_g = -1;   // the GGUF's OWN layer-1 types (read below)
    if (use_gguf) {
        const std::string gg = std::getenv("STRATA_TP5_GGUF");
        strata::GgufModel model{strata::gguf_split_paths(gg)};
        size_t sh = 0;
        auto getb = [&](const char* name, std::vector<uint8_t>& out, int* type_out = nullptr) {
            const strata::TensorInfo* t = model.find("blk.1." + std::string(name), &sh);
            if (t == nullptr) { std::fprintf(stderr, "tp5_gdn_test: GGUF lacks %s\n", name); std::exit(1); }
            const uint8_t* p = model.shard(sh).tensor_data(*t);
            size_t nb = size_t(t->elements()) * 4;
            if (t->type != 0 && t->shape.size() >= 2) {
                int be = 0, bb = 0;
                strata::block_geometry(t->type, be, bb);
                if (be > 0 && t->shape[0] % uint64_t(be) == 0)
                    nb = size_t(t->shape[1]) * size_t(t->shape[0] / uint64_t(be)) * size_t(bb);
            }
            out.assign(p, p + nb);
            if (type_out != nullptr) *type_out = int(t->type);
        };
        getb("attn_qkv.weight", g_qkv, &g_type_qkv); getb("attn_gate.weight", g_g, &g_type_g);
        getb("ssm_alpha.weight", g_a);
        getb("ssm_beta.weight", g_b); getb("ssm_conv1d.weight", g_c);
        getb("ssm_norm.weight", g_nm); getb("ssm_dt.bias", g_dt);
        getb("ssm_a", g_sa);
    }
    const std::vector<uint8_t>& b_qkv = use_gguf ? g_qkv : f_qkv;
    const std::vector<uint8_t>& b_g = use_gguf ? g_g : f_g;
    const std::vector<uint8_t>& b_a = use_gguf ? g_a : f_a;
    const std::vector<uint8_t>& b_b = use_gguf ? g_b : f_b;
    const std::vector<uint8_t>& b_c = use_gguf ? g_c : f_c;
    const std::vector<uint8_t>& b_nm = use_gguf ? g_nm : f_nm;
    const std::vector<uint8_t>& b_dt = use_gguf ? g_dt : f_dt;
    const std::vector<uint8_t>& b_sa = use_gguf ? g_sa : f_sa;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 5) {
        std::printf("tp5_gdn_test: fewer than five GPUs - SKIP\n");
        return 77;
    }

    // the layer-1 types: qkv/gate IQ4_XS(23), out Q4_K(12) (the GGUF's own when
    // the layer-1 types: the FIXTURE's are IQ4_XS(23) by construction; the
    // GGUF mode uses the model's OWN blk.1 TensorInfo types (the real model
    // mixes Q3_K/Q4_K/IQ4_XS by layer - never assume 23 for actual GGUF runs)
    const int kTypeQKV = use_gguf ? g_type_qkv : 23;
    const int kTypeG = use_gguf ? g_type_g : 23;
    if (kTypeQKV < 0 || kTypeG < 0) {
        std::fprintf(stderr, "tp5_gdn_test: GGUF mode failed to read layer-1 tensor types\n");
        return 1;
    }
    std::printf("  layer-1 types: %d (qkv) %d (gate) - %s\n", kTypeQKV, kTypeG,
                use_gguf ? "the model's own" : "the fixture's");
    auto bytes_of = [](int type, size_t elems) {
        int be = 0, bb = 0;
        strata::block_geometry(type, be, bb);
        return elems / size_t(be) * size_t(bb);
    };
    const size_t nqkv = bytes_of(kTypeQKV, size_t(H) * C);
    const size_t ng = bytes_of(kTypeG, size_t(H) * HV * S);
    if (nqkv != b_qkv.size() || ng != b_g.size()) {
        std::fprintf(stderr, "tp5_gdn_test: fixture sizes wrong (qkv %zu/%zu g %zu/%zu)\n",
                     nqkv, b_qkv.size(), ng, b_g.size());
        return 1;
    }

    // the token stream: kNTok rows (the window), plus the state0 the gate injects
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> fx(-1.5f, 1.5f);
    std::vector<float> xs{std::vector<float>(size_t(kNTok) * H)};
    for (auto& v : xs) v = fx(rng);
    // the initial state: [48 heads x 128*128] + conv [10240 x 3]
    std::vector<float> state0{std::vector<float>(size_t(HV) * S * S + size_t(C) * 3)};
    for (auto& v : state0) v = fx(rng) * 0.25f;

    // ---- the reference: the UNSPLIT layer on GPU 0, the verify window's own
    // ---- the reference: the UNSPLIT layer on GPU 0, the verify window's own
    // kernel sequence (qkv -> conv -> ab -> gate -> recurrence)
    float *rx = nullptr, *rqkv = nullptr, *rhb = nullptr, *rgate = nullptr, *rbeta = nullptr,
          *rz = nullptr, *ry = nullptr, *rstate = nullptr, *rconv = nullptr;
    uint8_t *dwqkv = nullptr, *dwg = nullptr;
    uint16_t *dwa = nullptr, *dwb = nullptr;
    float *dwc = nullptr, *dwdt = nullptr, *dwsa = nullptr, *dwnm = nullptr;
    void* rxq = nullptr;
    {
        cudaMalloc(&dwqkv, nqkv); cudaMemcpy(dwqkv, b_qkv.data(), nqkv, cudaMemcpyHostToDevice);
        cudaMalloc(&dwg, ng); cudaMemcpy(dwg, b_g.data(), ng, cudaMemcpyHostToDevice);
        cudaMalloc(&dwa, b_a.size()); cudaMemcpy(dwa, b_a.data(), b_a.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&dwb, b_b.size()); cudaMemcpy(dwb, b_b.data(), b_b.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&dwc, b_c.size()); cudaMemcpy(dwc, b_c.data(), b_c.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&dwdt, b_dt.size()); cudaMemcpy(dwdt, b_dt.data(), b_dt.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&dwsa, b_sa.size()); cudaMemcpy(dwsa, b_sa.data(), b_sa.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&dwnm, b_nm.size()); cudaMemcpy(dwnm, b_nm.data(), b_nm.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&rx, size_t(kMaxT) * H * 4); cudaMemcpy(rx, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&rqkv, size_t(kMaxT) * C * 4);
        cudaMalloc(&rhb, size_t(kMaxT) * C * 4);
        cudaMalloc(&rgate, size_t(kMaxT) * HV * 4);
        cudaMalloc(&rbeta, size_t(kMaxT) * HV * 4);
        cudaMalloc(&rz, size_t(kMaxT) * HV * S * 4);
        cudaMalloc(&ry, size_t(kMaxT) * HV * S * 4);
        cudaMalloc(&rstate, size_t(HV) * S * S * 4);
        cudaMalloc(&rconv, size_t(C) * 3 * 4);
        cudaMalloc(&rxq, native_q8_1_bytes(H, kMaxT));
        // the state0 the reference AND the tier start from
        cudaMemcpy(rstate, state0.data(), size_t(HV) * S * S * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(rconv, state0.data() + size_t(HV) * S * S, size_t(C) * 3 * 4, cudaMemcpyHostToDevice);
    }
    // the reference sequence (the default stream, as the engine's own graph does)
    cudaStream_t rst = nullptr;
    cudaStreamCreateWithFlags(&rst, cudaStreamNonBlocking);
    native_quantize_q8_1(rx, rxq, H, kNTok, rst);
    native_mmvq(kTypeQKV, dwqkv, rxq, rqkv, H, C, kNTok, rst);
    std::fprintf(stderr, "conv args: rconv=%p rqkv=%p dwc=%p rhb=%p C=%d n=%d\n",
                 (void*)rconv, (void*)rqkv, (void*)dwc, (void*)rhb, C, kNTok);
    gdn_conv_l2_multi(rconv, rqkv, dwc, rhb, C, 2 * HK, EPS, kNTok, rst, 0, false);
    gdn_ab_multi(rx, (const uint16_t*) dwa, (const uint16_t*) dwb, dwdt, dwsa, rgate, rbeta, H, HV, kNTok, rst);
    native_mmvq(kTypeG, dwg, rxq, rz, H, HV * S, kNTok, rst);
    gdn_step_norm_multi(rstate, rhb, C, rgate, rbeta, rz, dwnm, EPS, ry, HK, HV, kNTok, nullptr, rst, 0, nullptr);
    cudaStreamSynchronize(rst);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::fprintf(stderr, "tp5_gdn_test: reference chain: %s\n", cudaGetErrorString(e));
        return 1;
    }

    // copy the reference's intermediates out
    std::vector<float> ref_qkv{std::vector<float>(size_t(kNTok) * C)};
    std::vector<float> ref_hb{std::vector<float>(size_t(kNTok) * C)};
    std::vector<float> ref_gate{std::vector<float>(size_t(kNTok) * HV)};
    std::vector<float> ref_beta{std::vector<float>(size_t(kNTok) * HV)};
    std::vector<float> ref_z{std::vector<float>(size_t(kNTok) * HV * S)};
    std::vector<float> ref_y{std::vector<float>(size_t(kNTok) * HV * S)};
    cudaMemcpy(ref_qkv.data(), rqkv, ref_qkv.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ref_hb.data(), rhb, ref_hb.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ref_gate.data(), rgate, ref_gate.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ref_beta.data(), rbeta, ref_beta.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ref_z.data(), rz, ref_z.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ref_y.data(), ry, ref_y.size() * 4, cudaMemcpyDeviceToHost);

    // ---- the tier (fixture mode), five partials.  The x rows go through a
    // MAPPED host buffer when TP5_GDN_MAPPEDX=1 (the engine's own doorbell path)
    float* mapped_x = nullptr; float* mapped_y = nullptr;
    if (std::getenv("TP5_GDN_MAPPEDX") != nullptr) {
        cudaHostAlloc((void**)&mapped_x, size_t(kNTok) * H * 4, cudaHostAllocMapped | cudaHostAllocPortable);
        cudaHostAlloc((void**)&mapped_y, size_t(kNTok) * HV * S * 4, cudaHostAllocMapped | cudaHostAllocPortable);
        std::memcpy(mapped_x, xs.data(), xs.size() * 4);
    }
    Tp5GdnTier tier;
    {
        std::string terr;
        std::vector<std::string> shards;
        if (use_gguf) {
            const char* g = std::getenv("STRATA_TP5_GGUF");
            if (g != nullptr) shards = strata::gguf_split_paths(std::string(g));
        }
        if (!tier.open(shards, H, terr)) {
            std::fprintf(stderr, "tp5_gdn_test: open: %s\n", terr.c_str());
            return 1;
        }
    }
    std::vector<float> y{std::vector<float>(size_t(kNTok) * HV * S)};   // the gathered y (full value dim)
    const float* xs_ptr = mapped_x != nullptr ? mapped_x : xs.data();
    float* y_ptr = mapped_x != nullptr ? mapped_y : y.data();
    {
        std::string terr;
        const bool no_state0 = std::getenv("TP5_GDN_NOSTATE0") != nullptr;
        if (!tier.run(1, xs_ptr, kNTok, y_ptr, terr, no_state0 ? nullptr : state0.data())) {
            std::fprintf(stderr, "tp5_gdn_test: run: %s\n", terr.c_str());
            return 1;
        }
    }
    int bad = 0;
    int nq = 0, nh = 0, ngt = 0, nbt = 0, nz = 0, ny = 0;
    // the intermediates, per rank, bit for bit (the channel map picks the rows)
    for (int r = 0; r < 5; ++r) {
        const Tp5GdnTier::Debug d = tier.debug(r);
        const int vc = d.v_count, cc = d.c_count;
        // qkv/hb: the rank's channels (the tier's chan map is private; recompute it)
        std::vector<int> chan{std::vector<int>(size_t(cc))};
        {
            int p = 0;
            for (int i = 0; i < vc; ++i) {
                const int qk = (kVStart[r] + i) % HK;
                for (int j = 0; j < S; ++j) chan[size_t(p++)] = qk * S + j;
            }
            for (int i = 0; i < vc; ++i) {
                const int qk = (kVStart[r] + i) % HK;
                for (int j = 0; j < S; ++j) chan[size_t(p++)] = S * HK + qk * S + j;
            }
            for (int i = 0; i < vc; ++i)
                for (int j = 0; j < S; ++j) chan[size_t(p++)] = 2 * S * HK + (kVStart[r] + i) * S + j;
        }
        std::vector<float> qkv{std::vector<float>(size_t(kNTok) * cc)}, hb{std::vector<float>(size_t(kNTok) * cc)};
        std::vector<float> gate{std::vector<float>(size_t(kNTok) * vc)}, beta{std::vector<float>(size_t(kNTok) * vc)};
        std::vector<float> z{std::vector<float>(size_t(kNTok) * vc * S)}, yy{std::vector<float>(size_t(kNTok) * vc * S)};
        {
            cudaSetDevice(tier.debug_gpu(r));
            cudaStreamSynchronize(tier.debug_stream(r));
            cudaMemcpy(qkv.data(), d.qkv, qkv.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hb.data(), d.hb, hb.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(gate.data(), d.gate, gate.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(beta.data(), d.beta, beta.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(z.data(), d.z, z.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(yy.data(), d.y, yy.size() * 4, cudaMemcpyDeviceToHost);
        }
        for (int t = 0; t < kNTok; ++t) {
            for (int c = 0; c < cc; ++c) {
                if (qkv[size_t(t) * cc + c] != ref_qkv[size_t(t) * C + chan[size_t(c)]]) { ++bad; ++nq; if (nq < 3) std::printf("  r%d t%d qkv c%d: ref=%g tier=%g\n", r, t, c, ref_qkv[size_t(t)*C+chan[size_t(c)]], qkv[size_t(t)*cc+c]); }
                if (hb[size_t(t) * cc + c] != ref_hb[size_t(t) * C + chan[size_t(c)]]) { ++bad; ++nh; if (nh < 3) std::printf("  r%d t%d hb c%d: ref=%g tier=%g\n", r, t, c, ref_hb[size_t(t)*C+chan[size_t(c)]], hb[size_t(t)*cc+c]); }
            }
            for (int i = 0; i < vc; ++i) {
                if (gate[size_t(t) * vc + i] != ref_gate[size_t(t) * HV + kVStart[r] + i]) { ++bad; ++ngt; if (ngt < 6) std::printf("  r%d t%d gate i%d: %g vs %g\n", r, t, i, ref_gate[size_t(t)*HV+kVStart[r]+i], gate[size_t(t)*vc+i]); }
                if (beta[size_t(t) * vc + i] != ref_beta[size_t(t) * HV + kVStart[r] + i]) { ++bad; ++nbt; if (nbt < 3) std::printf("  r%d t%d beta i%d: %g vs %g\n", r, t, i, ref_beta[size_t(t)*HV+kVStart[r]+i], beta[size_t(t)*vc+i]); }
            }
            for (int i = 0; i < vc * S; ++i) {
                if (z[size_t(t) * vc * S + i] != ref_z[size_t(t) * HV * S + (kVStart[r] + i / S) * S + i % S]) { ++bad; ++nz; if (nz < 3) std::printf("  r%d t%d z i%d: %g vs %g\n", r, t, i, ref_z[size_t(t)*HV*S+(kVStart[r]+i/S)*S+i%S], z[size_t(t)*vc*S+i]); }
                if (yy[size_t(t) * vc * S + i] != ref_y[size_t(t) * HV * S + (kVStart[r] + i / S) * S + i % S]) { ++bad; ++ny; if (ny < 3) std::printf("  r%d t%d y i%d: ref=%g tier=%g\n", r, t, i, ref_y[size_t(t)*HV*S+(kVStart[r]+i/S)*S+i%S], yy[size_t(t)*vc*S+i]); }
            }
        }
        std::printf("  rank %d (v %d, c %d): intermediates checked\n", r, vc, cc);
    }
    std::printf("  intermediate mismatches: %d (qkv %d hb %d gate %d beta %d z %d y %d)\n", bad, nq, nh, ngt, nbt, nz, ny);
    // the final bo: relative tolerance (the out projection's q8_1 requantisation)
    {   // the gathered y vs the reference's recurrence output: BIT-EXACT (the
        // gather is a disjoint copy; the quantize+project is the graph's)
        int off = 0;
        for (size_t i = 0; i < ref_y.size(); ++i)
            if (y_ptr[i] != ref_y[i]) { ++off; if (off < 3) std::printf("  y[%zu]: ref=%g tier=%g\n", i, ref_y[i], y_ptr[i]); }
        std::printf("  gathered y vs reference y: %d mismatches over %zu\n", off, ref_y.size());
        bad += off;
    }
    // the layer-offset gate (fixture mode only - the fixture's weights are
    // byte-identical across layers; the GGUF's layers differ): layer 2's output
    // must equal layer 1's - the QSA tier's own lesson: the first wiring read a
    // later layer's weights at a multiple of the single-layer span
    if (!use_gguf) {
        std::vector<float> y7{std::vector<float>(y.size())};
        std::string terr;
        const int64_t l7 = std::getenv("TP5_GDN_LAYER") != nullptr ? std::atoll(std::getenv("TP5_GDN_LAYER")) : 2;
        if (!tier.run(l7, xs_ptr, kNTok, y7.data(), terr, state0.data())) {
            std::fprintf(stderr, "tp5_gdn_test: run(7): %s\n", terr.c_str());
            return 1;
        }
        int off = 0;
        for (size_t i = 0; i < y.size(); ++i)
            if (y_ptr[i] != y7[i]) ++off;
        std::printf("  layer 2 vs layer 1 rows: %d mismatches over %zu\n", off, y.size());
        bad += off;
    }
    // the GRAPH-mode gate (STRATA_TP5_GDN_GRAPH=1 re-opens the tier with the
    // per-(layer, bucket) graphs captured): the graph path must produce the
    // SAME y as the eager path - the same kernels in the same stream order, so
    // bitwise.  2026-10-16: graph mode is the measured winner post-sync-fix
    // (-5.3% end-to-end); this pins its numerics at the fixture level.
    if (mapped_x == nullptr && !no_sync_gate) {
        std::string terr;
        Tp5GdnTier tier2;
        std::vector<std::string> shards;
        if (use_gguf) {
            const char* g = std::getenv("STRATA_TP5_GGUF");
            if (g != nullptr) shards = strata::gguf_split_paths(std::string(g));
        }
        // the graph path requires the engine's shape: state0 == nullptr and
        // n_tok <= 5 (the buckets).  The eager y above used state0 - recompute
        // the eager reference WITHOUT it (D2D from the tier's own state) so the
        // comparison isolates the graph path from the injection path.
        std::vector<float> ye{std::vector<float>(size_t(kNTok) * HV * S)};
        if (!tier.run(1, xs_ptr, kNTok, ye.data(), terr, nullptr)) {
            std::fprintf(stderr, "tp5_gdn_test: eager ref run: %s\n", terr.c_str());
            return 1;
        }
        // the graph tier's state starts at whatever the eager tier's is - the
        // graph gate only compares the PATH, so sync the graph tier to the same
        // state first (state0 injection is not the graph path's shape, but
        // sync_state is: the engine syncs, then runs graphs).
        setenv("STRATA_TP5_GDN_GRAPH", "1", 1);
        if (!tier2.open(shards, H, terr)) {
            std::fprintf(stderr, "tp5_gdn_test: graph open: %s\n", terr.c_str());
            return 1;
        }
        unsetenv("STRATA_TP5_GDN_GRAPH");
        // NO sync_state here: the eager reference above ran from the eager
        // tier's own state, which is still the OPEN-TIME ZERO (the state0
        // injections used the scratch, and the sync gate below has not run
        // yet).  tier2's own state is also the open-time zero - both runs
        // start from the SAME state, so the comparison isolates the graph path.
        std::vector<float> yg{std::vector<float>(ye.size())};
        if (!tier2.run(1, xs_ptr, kNTok, yg.data(), terr, nullptr)) {
            std::fprintf(stderr, "tp5_gdn_test: graph run: %s\n", terr.c_str());
            return 1;
        }
        int off = 0;
        for (size_t i = 0; i < ye.size(); ++i)
            if (yg[i] != ye[i]) { ++off; if (off < 3) std::printf("  yg[%zu]: eager=%g graph=%g\n", i, ye[i], yg[i]); }
        std::printf("  graph-mode y vs eager y: %d mismatches over %zu\n", off, ye.size());
        bad += off;
    }
    // the all-layers gate (TP5_GDN_ALLLAYERS=1): run every GDN layer in sequence
    // (0,1,2,4,...47) as the engine does - the per-layer spans and offsets must
    // hold for every layer, not just 1 and 2
    if (std::getenv("TP5_GDN_ALLLAYERS") != nullptr) {
        std::string terr;
        int fails = 0;
        int64_t lo = 0, hi = 48;
        if (const char* v = std::getenv("TP5_GDN_LAYER_RANGE"); v != nullptr) {
            const std::string r(v);
            const auto c = r.find(':');
            if (c != std::string::npos) { lo = std::atoll(r.substr(0, c).c_str()); hi = std::atoll(r.substr(c + 1).c_str()); }
        }
        for (int64_t l = lo; l < hi && fails < 3; ++l) {
            if (!tier.is_gdn_layer(l)) continue;
            if (!tier.run(l, xs_ptr, kNTok, y_ptr, terr, state0.data())) {
                std::fprintf(stderr, "tp5_gdn_test: layer %lld run: %s\n", (long long) l, terr.c_str());
                ++fails;
                break;
            }
            std::fprintf(stderr, "layer %lld ok\n", (long long) l);
        }
        std::printf("  all 36 layers: %s\n", fails == 0 ? "ran clean" : "FAILED");
        bad += fails;
    }
    // the sync_state gate: syncing the engine-shaped state (on GPU0) must equal
    // the direct state0 injection (the engine's post-prefill sync path)
    if (mapped_x == nullptr && !no_sync_gate) {
        // the engine-shaped state on GPU0: [48 heads x 128*128] + conv [10240 x 3]
        float* dstate = nullptr;
        {
            cudaSetDevice(0);
            cudaMalloc(&dstate, state0.size() * 4);
            cudaMemcpy(dstate, state0.data(), state0.size() * 4, cudaMemcpyHostToDevice);
        }
        std::string terr;
        if (!tier.sync_state(1, dstate, terr)) {
            std::fprintf(stderr, "tp5_gdn_test: sync_state: %s\n", terr.c_str());
            return 1;
        }
        std::vector<float> ys{std::vector<float>(y.size())};
        if (!tier.run(1, xs_ptr, kNTok, ys.data(), terr, nullptr)) {   // D2D from the SYNCED state
            std::fprintf(stderr, "tp5_gdn_test: run after sync: %s\n", terr.c_str());
            return 1;
        }
        // the direct-injection reference: y (already computed with state0) - but
        // y_ptr may be the mapped buffer; compare against a fresh state0 run
        std::vector<float> yref{std::vector<float>(y.size())};
        if (!tier.run(1, xs_ptr, kNTok, yref.data(), terr, state0.data())) {
            std::fprintf(stderr, "tp5_gdn_test: run state0: %s\n", terr.c_str());
            return 1;
        }
        int off = 0;
        for (size_t i = 0; i < ys.size(); ++i)
            if (ys[i] != yref[i]) ++off;
        std::printf("  sync_state vs direct injection: %d mismatches over %zu\n", off, ys.size());
        bad += off;
        cudaFree(dstate);
    }
    // the commit gate: commit 2 of 3 rows - the state MUST advance (the engine
    // wiring showed the state staying zero: the commit ran without error but the
    // recurrence's write-back never landed)
    {
        std::string terr;
        if (!tier.commit(1, 2, xs.data(), terr)) {
            std::fprintf(stderr, "tp5_gdn_test: commit: %s\n", terr.c_str());
            return 1;
        }
        cudaSetDevice(tier.debug_gpu(0));
        cudaStreamSynchronize(tier.debug_stream(0));
        std::vector<float> sv{std::vector<float>(4)};
        cudaMemcpy(sv.data(), tier.debug_state(0, 1), 4 * 4, cudaMemcpyDeviceToHost);
        const bool advanced = sv[0] != 0.0f || sv[1] != 0.0f || sv[2] != 0.0f || sv[3] != 0.0f;
        std::printf("  commit: %s (state[0..3] = %g %g %g %g)\n",
                    advanced ? "state ADVANCED" : "state STILL ZERO", sv[0], sv[1], sv[2], sv[3]);
        if (!advanced) ++bad;
    }
    // the commit_span gate: the ENGINE's rank-major batch must produce the
    // same state as the per-layer form for the same (layer, n_keep, x).  The
    // per-layer gate above already advanced layer 1's state with xs/2 - read
    // it as the reference, restore the state via sync_state, run the span
    // form over [1, 3), and require layer 1 bitwise equal.
    if (mapped_x == nullptr && !no_sync_gate) {
        std::vector<float> s1{std::vector<float>(4096)};
        cudaSetDevice(tier.debug_gpu(0));
        cudaMemcpy(s1.data(), tier.debug_state(0, 1), s1.size() * 4, cudaMemcpyDeviceToHost);
        float* dstate = nullptr;
        {
            cudaSetDevice(0);
            cudaMalloc(&dstate, state0.size() * 4);
            cudaMemcpy(dstate, state0.data(), state0.size() * 4, cudaMemcpyHostToDevice);
        }
        std::string terr;
        if (!tier.sync_state(1, dstate, terr)) {
            std::fprintf(stderr, "tp5_gdn_test: sync for span: %s\n", terr.c_str());
            return 1;
        }
        // the doorbell-snapshot layout: [layer][rows] with rows = n_tok * H.
        // The span reads x at layer stride `rows`; build a 2-layer buffer with
        // layer 1 = xs (the reference input) and layer 2 = xs again (its state
        // advance is not compared, only that the loop ran).
        const size_t rows = size_t(2) * 2560;
        std::vector<float> snap{std::vector<float>(2 * rows)};
        std::memcpy(snap.data() + 0 * rows, xs.data(), size_t(2) * 2560 * 4);   // layer 1's kept rows
        std::memcpy(snap.data() + 1 * rows, xs.data(), size_t(2) * 2560 * 4);   // layer 2's kept rows
        if (!tier.commit_span(1, 3, 2, snap.data(), rows, terr)) {
            std::fprintf(stderr, "tp5_gdn_test: commit_span: %s\n", terr.c_str());
            return 1;
        }
        cudaSetDevice(tier.debug_gpu(0));
        cudaStreamSynchronize(tier.debug_stream(0));
        std::vector<float> s2{std::vector<float>(4096)};
        cudaMemcpy(s2.data(), tier.debug_state(0, 1), s2.size() * 4, cudaMemcpyDeviceToHost);
        int off = 0;
        for (size_t i = 0; i < s1.size(); ++i)
            if (s1[i] != s2[i]) ++off;
        std::printf("  commit_span vs per-layer commit: %d mismatches over %zu\n", off, s1.size());
        bad += off;
        cudaFree(dstate);
    }
    // the EXPORT gate: after the REAL window runs and commits above, the ranks'
    // the EXPORT gate: after the REAL window runs and commits above, the ranks'
    // state must scatter back into an engine-shaped GPU0 arena EXACTLY as the
    // UNSPLIT reference's own advanced state - an independent full-width oracle,
    // not a sync/export copy roundtrip.  The tier's commit_span above advanced
    // layer 1 (lay_of(1) = 1 - layer 1 is a GDN layer, l%4 != 3) with the ACCEPTED
    // first 2 of 3 xs rows from the state0 sync; the reference below re-runs the engine's own
    // commit semantics over the same state0, the same 2 rows, the same kernel
    // order (conv_l2 from the PRE-commit history, conv_commit, the recurrence's
    // n_keep path) - a state the tier never saw.
    if (mapped_x == nullptr && !no_sync_gate) {
        // the reference's own commit: the same kernels the engine's commit graph
        // uses, over the first 2 rows, from the SAME state0 the tier was synced to
        // (the reference device pointers live on GPU0 - make sure we ARE on it)
        cudaSetDevice(0);
        cudaMemset(rstate, 0, size_t(HV) * S * S * 4);
        cudaMemset(rconv, 0, size_t(C) * 3 * 4);
        cudaMemcpy(rstate, state0.data(), size_t(HV) * S * S * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(rconv, state0.data() + size_t(HV) * S * S, size_t(C) * 3 * 4, cudaMemcpyHostToDevice);
        int nkeep = 2;
        int32_t* dkeep = nullptr;
        cudaMalloc(&dkeep, 4);
        cudaMemcpy(dkeep, &nkeep, 4, cudaMemcpyHostToDevice);
        gdn_conv_l2_multi(rconv, rqkv, dwc, rhb, C, 2 * HK, EPS, nkeep, rst, 0, false);
        gdn_conv_commit(rconv, rqkv, C, dkeep, rst);
        gdn_step_norm_multi(rstate, rhb, C, rgate, rbeta, rz, dwnm, EPS, ry, HK, HV,
                           nkeep, dkeep, rst, 0, nullptr);
        cudaStreamSynchronize(rst);
        cudaFree(dkeep);
        std::vector<float> ref_state{std::vector<float>(size_t(HV) * S * S + size_t(C) * 3)};
        cudaMemcpy(ref_state.data(), rstate, size_t(HV) * S * S * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(ref_state.data() + size_t(HV) * S * S, rconv, size_t(C) * 3 * 4, cudaMemcpyDeviceToHost);

        // the engine-shaped arena on GPU0: [36 GDN layer blocks] with a stride
        // CARRIING A TAIL (the canonical block + 17 padding floats) - the export
        // must fill only the canonical part of each written layer and never
        // touch the stride tail.  A marker fills everything so untouched layers
        // and tails must all survive.
        const uint64_t canonical = uint64_t(HV) * S * S + uint64_t(C) * 3;
        const uint64_t gdn_floats = canonical + 17;
        float* arena = nullptr;
        {
            cudaSetDevice(0);
            cudaMalloc(&arena, size_t(36) * gdn_floats * 4);
            std::vector<float> mark{std::vector<float>(size_t(36) * gdn_floats, -7.5f)};
            cudaMemcpy(arena, mark.data(), mark.size() * 4, cudaMemcpyHostToDevice);
        }
        std::string terr;
        // call from a NON-ZERO caller device: the export must preserve it
        cudaSetDevice(1);
        const bool ok = tier.export_state_span(1, 2, arena, gdn_floats, terr);
        int dev_after = -1;
        cudaGetDevice(&dev_after);
        const bool ctx_ok = dev_after == 1;
        std::printf("  export caller device preserved: %s (after=%d)\n", ctx_ok ? "yes" : "NO", dev_after);
        if (!ctx_ok) ++bad;
        if (!ok) {
            std::fprintf(stderr, "tp5_gdn_test: export_state_span: %s\n", terr.c_str());
            return 1;
        }
        std::vector<float> got{std::vector<float>(size_t(36) * gdn_floats)};
        cudaSetDevice(0);
        cudaMemcpy(got.data(), arena, got.size() * 4, cudaMemcpyDeviceToHost);
        // layer 1 (lay_of(1) = 1): the exported block must be the reference bitwise
        int off = 0;
        for (size_t i = 0; i < ref_state.size(); ++i)
            if (!std::isfinite(got[gdn_floats + i]) || !std::isfinite(ref_state[i]) ||
                std::memcmp(&got[gdn_floats + i], &ref_state[i], sizeof(float)) != 0) {
                ++off;
                if (off < 3) std::printf("  exp[%zu]: ref=%g tier=%g\n", i, ref_state[i], got[gdn_floats + i]);
            }
        std::printf("  exported layer 1 vs full-width reference commit: %d mismatches over %zu\n", off, ref_state.size());
        bad += off;
        // the SPAN contract: ONLY layer 1 is written ([1,2)); every other layer's
        // canonical block AND every layer's stride tail (including layer 1's)
        // must keep the marker - the export never writes past the canonical part
        int untouched = 0;
        for (int64_t gi = 0; gi < 36; ++gi)
            for (size_t i = gi == 1 ? size_t(canonical) : 0; i < size_t(gdn_floats); ++i)
                if (got[size_t(gi) * gdn_floats + i] != -7.5f) ++untouched;
        std::printf("  export span isolation: %d stray writes outside layer 1's canonical block\n", untouched);
        bad += untouched;
        cudaFree(arena);
    }
    // the CLOSE/REOPEN gate: the tier's close() must free every rank's
    // resources cleanly (no sticky error left behind - a failed free would
    // surface at the next unrelated launch) and a REOPENED tier must produce
    // the same y for the same layer/input from the fresh zero state.  The
    // reference is computed BEFORE the close (the full-width chain over the
    // same xs rows from the ZERO state - the reopened tier's own initial
    // state), so the comparison is against an independent oracle, never the
    // old tier's own output.
    if (mapped_x == nullptr && !no_sync_gate) {
        // the fresh-state reference: the same UNSPLIT chain from the ZERO state
        // (the reopened tier starts from the open-time zero, no state0 injection)
        cudaSetDevice(0);
        cudaMemset(rstate, 0, size_t(HV) * S * S * 4);
        cudaMemset(rconv, 0, size_t(C) * 3 * 4);
        native_quantize_q8_1(rx, rxq, H, kNTok, rst);
        native_mmvq(kTypeQKV, dwqkv, rxq, rqkv, H, C, kNTok, rst);
        gdn_conv_l2_multi(rconv, rqkv, dwc, rhb, C, 2 * HK, EPS, kNTok, rst, 0, false);
        gdn_ab_multi(rx, (const uint16_t*) dwa, (const uint16_t*) dwb, dwdt, dwsa, rgate, rbeta, H, HV, kNTok, rst);
        native_mmvq(kTypeG, dwg, rxq, rz, H, HV * S, kNTok, rst);
        gdn_step_norm_multi(rstate, rhb, C, rgate, rbeta, rz, dwnm, EPS, ry, HK, HV, kNTok, nullptr, rst, 0, nullptr);
        cudaStreamSynchronize(rst);
        if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
            std::fprintf(stderr, "tp5_gdn_test: fresh-state reference chain: %s\n", cudaGetErrorString(e));
            return 1;
        }
        std::vector<float> ref_fresh{std::vector<float>(size_t(kNTok) * HV * S)};
        cudaMemcpy(ref_fresh.data(), ry, ref_fresh.size() * 4, cudaMemcpyDeviceToHost);

        // close: a CHECKED drain of the error state - a failed free must FAIL
        // the gate, not be swallowed by a bare not-throw
        tier.close();
        const cudaError_t ce = cudaGetLastError();
        if (ce != cudaSuccess) {
            std::printf("  close left a sticky error: %s\n", cudaGetErrorString(ce));
            ++bad;
        } else {
            std::printf("  close: clean (no sticky error)\n");
        }

        // reopen and run the SAME layer from the fresh state - only when the
        // close was clean (a failed close leaves the reopen's results untrustworthy)
        if (ce == cudaSuccess) {
            std::string terr;
            std::vector<std::string> shards;
            if (use_gguf) {
                const char* g = std::getenv("STRATA_TP5_GGUF");
                if (g != nullptr) shards = strata::gguf_split_paths(std::string(g));
            }
            if (!tier.open(shards, H, terr)) {
                std::fprintf(stderr, "tp5_gdn_test: reopen: %s\n", terr.c_str());
                return 1;
            }
            std::vector<float> y2{std::vector<float>(size_t(kNTok) * HV * S)};
            if (!tier.run(1, xs_ptr, kNTok, y2.data(), terr, nullptr)) {   // the fresh zero state
                std::fprintf(stderr, "tp5_gdn_test: reopen run: %s\n", terr.c_str());
                return 1;
            }
            int off = 0;
            for (size_t i = 0; i < ref_fresh.size(); ++i)
                if (!std::isfinite(y2[i]) || !std::isfinite(ref_fresh[i]) ||
                    std::memcmp(&y2[i], &ref_fresh[i], sizeof(float)) != 0) {
                    ++off;
                    if (off < 3) std::printf("  reopen[%zu]: ref=%g tier=%g\n", i, ref_fresh[i], y2[i]);
                }
            std::printf("  reopened tier vs fresh-state full-width reference: %d mismatches over %zu\n", off, ref_fresh.size());
            bad += off;
        }
    }
    if (bad != 0) {
        std::printf("tp5_gdn_test: FAIL (%d mismatches)\n", bad);
        return 1;
    }
    std::printf("tp5_gdn_test: PASS\n");
    return 0;
}
