// src/core/tp5_gdn.cu - 06-phase4 §4.2.3: the 5-rank partial GDN tier (see the header).

#include "strata/core/tp5_gdn.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/bf16_gemv.hpp"

#include <immintrin.h>

#include "strata/kernels/tp5_relay.hpp"

#include <algorithm>
#include <memory>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace tp5 = strata::kernels::tp5;

namespace strata::core {

namespace {

using strata::kernels::tp5::kRanks;

constexpr int H = 2560;        // n_embd
constexpr int S = 128;         // GDN state size
constexpr int HK = 16;         // the model's qk heads
constexpr int HV = 48;         // the model's V heads
constexpr int C_ALL = 2 * S * HK + S * HV;   // 10240

/// Each rank inbox slot holds only a compact slice of up to 10 V heads (1280
/// floats/row), followed by its 128-byte-aligned ready flag. The CPU gathers
/// each rank's slice into the caller's full 6144-wide row at v_start_[r]*S.
constexpr int kMaxRankSlice = 10 * S;
inline __host__ __device__ uint64_t gdn_flag_off(int rows) {
    return ((uint64_t(rows) * kMaxRankSlice * 4 + 127) / 128) * 128;
}
inline __host__ __device__ uint64_t gdn_inbox_bytes(int rows) { return gdn_flag_off(rows) + 128; }

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("tp5 gdn: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

struct On {
    int prev = 0;
    explicit On(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
    ~On() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
};

/// The host clock in ns (06 §5.3: the fold anchor for prod_end_host_ns).
long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// The RELAY producer for the mixer partials: same contract as the QSA tier's
/// publish_partial_kernel (grid 2D: x covers the row elements, y one block per
/// row; the counter's target is the TOTAL block count).
// The publish carries the rank's RECURRENCE OUTPUT slice (vc*S floats per row,
// at the V-head offset vs*S) - not the out-projection partial: the out
// projection runs ONCE on the full y (GPU0, in the graph) so its q8_1
// quantisation and dot order are the engine's own, bitwise.  The split-sum
// version's ~1e-3 out-projection error accumulated in the recurrence state
// (s = g*s + sk*delta) over the windows and degenerated the text.
__global__ void publish_partial_kernel(const float* __restrict__ y_dev,
                                       uint8_t* __restrict__ inbox_base, uint64_t stride,
                                       int64_t epoch, uint32_t bank, int rank, int v_count,
                                       unsigned int* __restrict__ counter,
                                       const strata::kernels::tp5::StepState* __restrict__ step) {
    if (step != nullptr) {   // the graph path: epoch/bank ride the device StepState
        epoch = (int64_t) step->epoch;
        bank = step->bank;
    }
    uint8_t* ib = inbox_base + (uint64_t(bank) * kRanks + rank) * stride;
    float* payload = reinterpret_cast<float*>(ib);
    volatile uint64_t* ready = reinterpret_cast<volatile uint64_t*>(ib + gdn_flag_off(int(Tp5GdnTier::kWindowCap)));
    const int64_t token = blockIdx.y;
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + int(threadIdx.x);
    const int64_t vc_s = int64_t(v_count) * S;
    if (col < vc_s) {
        const int64_t idx = token * vc_s + col;
        payload[idx] = y_dev[idx];
    }
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        const unsigned arrived = atomicAdd(counter, 1u) + 1;
        if (arrived == gridDim.x * gridDim.y) {
            __threadfence_system();
            *ready = uint64_t(epoch);
        }
    }
}

}  // namespace

Tp5GdnTier::~Tp5GdnTier() { close(); }

void Tp5GdnTier::close() {
    bool owned = open_ || inbox_ || credits_ || g_abort_ || n_keep_h_ || exp_stage_ || exp_rank_;
    for (const RankState& s : rs_) owned = owned || s.stream != nullptr;
    if (!owned) return;
    if (g_abort_) __atomic_store_n(g_abort_, 1u, __ATOMIC_RELEASE);
    // Captured graphs and DMA sources remain live until every reader is done,
    // including ranks allocated by an open() which failed before open_=true.
    for (int r = 0; r < kRanks; ++r) {
        On on(gpu_of_rank(r));
        if (rs_[r].stream) cudaStreamSynchronize(rs_[r].stream);
    }
    for (int r = 0; r < kRanks; ++r) {
        On on(gpu_of_rank(r));
        for (size_t i = 0; i < 36 * size_t(graph_buckets_); ++i) {
            if (gexecs_[r][i]) { cudaGraphExecDestroy(gexecs_[r][i]); gexecs_[r][i] = nullptr; }
            if (graphs_[r][i]) { cudaGraphDestroy(graphs_[r][i]); graphs_[r][i] = nullptr; }
        }
        if (step_dev_[r]) { cudaFree(step_dev_[r]); step_dev_[r] = nullptr; }
    }
    if (x_stage_) { cudaFreeHost(x_stage_); x_stage_ = nullptr; }
    if (step_stage_) { cudaFreeHost(step_stage_); step_stage_ = nullptr; }
    for (int r = 0; r < kRanks; ++r) {
        RankState& s = rs_[r];
        On on(gpu_of_rank(r));
        if (s.stream) cudaStreamDestroy(s.stream);
        cudaFree(s.wqkv); cudaFree(s.wg);
        cudaFree(s.wa); cudaFree(s.wb);
        cudaFree((void*)s.wc); cudaFree((void*)s.wdt); cudaFree((void*)s.wsa); cudaFree((void*)s.wnm);
        for (int64_t i = 0; i < n_gdn_layers_; ++i) {
            if (s.state) cudaFree(s.state[i]);
        }
        delete[] s.state; delete[] s.conv;
        cudaFree(s.state_scratch); cudaFree(s.conv_scratch);
        cudaFree(s.qkv); cudaFree(s.hb); cudaFree(s.gate); cudaFree(s.beta);
        cudaFree(s.z); cudaFree(s.y);
        cudaFree(s.x_dev); cudaFree(s.xq);
        if (pcounters_[r]) { cudaFree(pcounters_[r]); pcounters_[r] = nullptr; }
        if (trace_dev_[r]) cudaFree(trace_dev_[r]);
        std::memset(&s, 0, sizeof(s));
    }
    if (inbox_) cudaFreeHost(inbox_);
    if (credits_) cudaFreeHost(credits_);
    if (g_abort_) cudaFreeHost(g_abort_);
    if (exp_stage_) { cudaFreeHost(exp_stage_); exp_stage_ = nullptr; }
    if (exp_rank_) { cudaFreeHost(exp_rank_); exp_rank_ = nullptr; }
    if (n_keep_h_) { cudaFreeHost(n_keep_h_); n_keep_h_ = nullptr; }
    for (int r = 0; r < kRanks; ++r) {
        On on(gpu_of_rank(r));
        if (n_keep_dev_[r]) { cudaFree(n_keep_dev_[r]); n_keep_dev_[r] = nullptr; }
    }
    inbox_ = nullptr; credits_ = nullptr; g_abort_ = nullptr;
    std::memset(inbox_dev_r_, 0, sizeof(inbox_dev_r_));
    std::memset(g_abort_dev_r_, 0, sizeof(g_abort_dev_r_));
    std::memset(trace_dev_, 0, sizeof(trace_dev_));
    open_ = false;
    n_gdn_layers_ = 0;
    epoch_ = 0;
    vram_bytes_ = 0;
}

bool Tp5GdnTier::open(const std::vector<std::string>& gguf_shards, int64_t n_embd, std::string& err) {
    close();
    r4_gpu_ = 2;   // the default placement; STRATA_TP5_GDN_R4_GPU overrides (the trace experiment)
    if (const char* v = std::getenv("STRATA_TP5_GDN_R4_GPU")) {
        const int g = std::atoi(v);
        if (g == 0) {
            err = "tp5 gdn: STRATA_TP5_GDN_R4_GPU=0 is unsupported (GPU0's primary queue belongs to the verifier and FFN rank 4; use 1..4)";
            return false;
        }
        if (g >= 1 && g <= 4) r4_gpu_ = g;
    }
    const char* fix_env = std::getenv("STRATA_TP5_GDN_FIXTURE");
    const bool have_fixture = fix_env != nullptr && *fix_env != '\0';
    if ((gguf_shards.empty() && !have_fixture) || n_embd != H) {
        err = "tp5 gdn: needs the model's shards (or STRATA_TP5_GDN_FIXTURE) and n_embd 2560";
        return false;
    }
    // the model's GDN layers: l % 4 != 3, 36 of them at 48 layers
    std::vector<int64_t> gdn_layers;
    for (int64_t l = 0; l < 48; ++l)
        if (l % 4 != 3) gdn_layers.push_back(l);
    n_gdn_layers_ = int64_t(gdn_layers.size());
    if (n_gdn_layers_ != 36) { err = "tp5 gdn: expected 36 GDN layers"; return false; }
    for (int r = 0; r < kRanks; ++r) c_count_[r] = 3 * S * v_count_[r];

    // ---- the weights: the GGUF or the FIXTURE (GDN layer 1's bytes)
    struct Lay {
        const strata::TensorInfo *qkv, *g, *a, *b, *c, *nm, *dt, *sa;
        size_t sq, sg, sa_, sb, sc, sn, sd, ss;
        const uint8_t *fqkv, *fg, *fa, *fb, *fc, *fnm, *fdt, *fsa;   // fixture bytes
    };
    std::vector<Lay> lays{std::vector<Lay>(size_t(n_gdn_layers_))};
    std::vector<std::vector<uint8_t>> fix;
    std::unique_ptr<strata::GgufModel> model;
    if (have_fixture) {
        auto read_fix = [&](const char* name) -> std::vector<uint8_t> {
            const std::string path = std::string(fix_env) + "/blk_1_" + name + ".bin";
            std::FILE* f = std::fopen(path.c_str(), "rb");
            if (f == nullptr) return {};
            std::fseek(f, 0, SEEK_END);
            const long n = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            std::vector<uint8_t> b(size_t(n < 0 ? 0 : n));
            const size_t got = b.empty() ? 0 : std::fread(b.data(), 1, b.size(), f);
            std::fclose(f);
            if (got != b.size()) return {};
            return b;
        };
        fix.resize(8);
        fix[0] = read_fix("attn_qkv_weight");    fix[1] = read_fix("attn_gate_weight");
        fix[2] = read_fix("ssm_alpha_weight");   fix[3] = read_fix("ssm_beta_weight");
        fix[4] = read_fix("ssm_conv1d_weight");  fix[5] = read_fix("ssm_norm_weight");
        fix[6] = read_fix("ssm_dt_bias");        fix[7] = read_fix("ssm_a");
        for (const auto& b : fix)
            if (b.empty()) { err = "tp5 gdn: the fixture directory is missing a tensor"; return false; }
        for (auto& L : lays) {
            L.fqkv = fix[0].data(); L.fg = fix[1].data();
            L.fa = fix[2].data(); L.fb = fix[3].data(); L.fc = fix[4].data();
            L.fnm = fix[5].data(); L.fdt = fix[6].data(); L.fsa = fix[7].data();
        }
    } else {
        model = std::make_unique<strata::GgufModel>(gguf_shards);
        for (size_t i = 0; i < lays.size(); ++i) {
            const std::string pre = "blk." + std::to_string(gdn_layers[i]) + ".";
            Lay& L = lays[i];
            auto get = [&](const char* suf, size_t& sh) -> const strata::TensorInfo* {
                return model->find(pre + suf, &sh);
            };
            L.qkv = get("attn_qkv.weight", L.sq);
            L.g = get("attn_gate.weight", L.sg);
            L.a = get("ssm_alpha.weight", L.sa_);
            L.b = get("ssm_beta.weight", L.sb);
            L.c = get("ssm_conv1d.weight", L.sc);
            L.nm = get("ssm_norm.weight", L.sn);
            L.dt = get("ssm_dt.bias", L.sd);
            L.sa = get("ssm_a", L.ss);
            if (!L.qkv || !L.g || !L.a || !L.b || !L.c || !L.nm || !L.dt || !L.sa) {
                err = "tp5 gdn: layer " + std::to_string(gdn_layers[i]) + " is missing a GDN tensor";
                return false;
            }
        }
    }
    auto lay_type = [&](const strata::TensorInfo* T, int fixt) { return have_fixture ? fixt : int(T->type); };
    // the fixture's layer-1 types: qkv=IQ4_XS(23), gate=IQ4_XS(23), out=Q4_K(12)
    for (size_t i = 0; i < lays.size(); ++i) {
        lay_qkv_type_[i] = lay_type(lays[i].qkv, 23);
        lay_g_type_[i] = lay_type(lays[i].g, 23);
    }
    // per-rank per-layer byte spans (the blob offsets are prefix sums - the QSA
    // tier's layer-offset lesson: the types differ per layer, no multiplication)
    for (int r = 0; r < kRanks; ++r) {
        int64_t qoff = 0, goff = 0;
        for (size_t i = 0; i < lays.size(); ++i) {
            lay_qkv_span_[i * kRanks + r] = qoff;
            lay_g_span_[i * kRanks + r] = goff;
            int be = 0, bb = 0;
            strata::block_geometry(lay_qkv_type_[i], be, bb);
            qoff += int64_t(c_count_[r]) * (H / be) * bb;
            strata::block_geometry(lay_g_type_[i], be, bb);
            goff += int64_t(v_count_[r]) * S * (H / be) * bb;
        }
    }

    // the RELAY inbox (the GDN's own 6144-wide payload)
    const uint64_t stride = gdn_inbox_bytes(int(kWindowCap));
    const size_t total = size_t(stride) * strata::kernels::tp5::kBanks * kRanks;
    unsigned mapped_flags = cudaHostAllocMapped | cudaHostAllocPortable;
#if defined(__HIPCC__)
    mapped_flags |= cudaHostAllocCoherent;
#endif
    if (!ck(cudaHostAlloc((void **)&inbox_, total, mapped_flags), "inbox", err)) {
        close();
        return false;
    }
    std::memset(inbox_, 0, total);
    if (!ck(cudaHostAlloc((void **)&credits_, sizeof(strata::kernels::tp5::Credit) * strata::kernels::tp5::kBanks,
                          mapped_flags),
            "credits", err)) {
        close();
        return false;
    }
    std::memset(credits_, 0, sizeof(strata::kernels::tp5::Credit) * strata::kernels::tp5::kBanks);
    if (!ck(cudaHostAlloc((void **)&g_abort_, sizeof(unsigned), mapped_flags), "g_abort", err)) {
        close();
        return false;
    }
    *g_abort_ = 0;
    if (!ck(cudaHostAlloc((void**)&n_keep_h_, size_t(kRanks) * sizeof(*n_keep_h_),
                          cudaHostAllocPortable), "n_keep pin", err)) {
        close();
        return false;
    }
    std::memset(n_keep_h_, 0, size_t(kRanks) * sizeof(*n_keep_h_));

    // ---- per-rank state
    for (int r = 0; r < kRanks; ++r) {
        RankState &s = rs_[r];
        On on(gpu_of_rank(r));
        if (!ck(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking), "stream", err))
            { close(); return false; }
        const int vc = v_count_[r], vs = v_start_[r], cc = c_count_[r];

        // the rank's channel map, in the rank's OWN order:
        //   q:  local slot i (V head i) holds qk head (vs+i)%16's q rows
        //   k:  the same heads' k rows
        //   v:  V head vs+i's v rows
        // The q|k rows in V-head order make the recurrence kernel's
        // `qh = head % h_k` (h_k = v_count) the identity - the shared kernel
        // needs no change.
        std::vector<int> chan{std::vector<int>(size_t(cc))};
        {
            int p = 0;
            for (int i = 0; i < vc; ++i) {
                const int qk = (vs + i) % HK;
                for (int j = 0; j < S; ++j) chan[size_t(p++)] = qk * S + j;             // q rows
            }
            for (int i = 0; i < vc; ++i) {
                const int qk = (vs + i) % HK;
                for (int j = 0; j < S; ++j) chan[size_t(p++)] = S * HK + qk * S + j;   // k rows: [2048, 4096)
            }
            for (int i = 0; i < vc; ++i)
                for (int j = 0; j < S; ++j) chan[size_t(p++)] = 2 * S * HK + vs * S + i * S + j;  // v rows
        }
        std::memcpy(chan_[r], chan.data(), size_t(cc) * sizeof(int));
        // gather the projections layer by layer into one host staging blob each
        // gather the projections layer by layer into one host staging blob each
        std::vector<uint8_t> hqkv, hg;
        std::vector<uint16_t> ha, hb;
        std::vector<float> hc, hdt, hsa, hnm;
        for (size_t i = 0; i < lays.size(); ++i) {
            const Lay& L = lays[i];
            {   // qkv: gather the rank's channel rows
                int be = 0, bb = 0;
                strata::block_geometry(lay_qkv_type_[i], be, bb);
                const int64_t blocks_per_row = H / be;
                const uint8_t* src = have_fixture ? L.fqkv : model->shard(L.sq).tensor_data(*L.qkv);
                const size_t row_bytes = size_t(blocks_per_row) * bb;
                for (int c = 0; c < cc; ++c) {
                    const uint8_t* rs = src + size_t(chan[size_t(c)]) * row_bytes;
                    hqkv.insert(hqkv.end(), rs, rs + row_bytes);
                }
            }
            {   // gate: the rank's V-head rows of the ZV=6144 output rows
                int be = 0, bb = 0;
                strata::block_geometry(lay_g_type_[i], be, bb);
                const int64_t blocks_per_row = H / be;
                const uint8_t* src = have_fixture ? L.fg : model->shard(L.sg).tensor_data(*L.g);
                const size_t row_bytes = size_t(blocks_per_row) * bb;
                for (int c = 0; c < vc * S; ++c) {
                    const uint8_t* rs = src + size_t(vs * S + c) * row_bytes;
                    hg.insert(hg.end(), rs, rs + row_bytes);
                }
            }
            {   // alpha/beta: the rank's V-head rows (BF16, [2560, 48] - row = head)
                const uint8_t* srca = have_fixture ? L.fa : model->shard(L.sa_).tensor_data(*L.a);
                const uint8_t* srcb = have_fixture ? L.fb : model->shard(L.sb).tensor_data(*L.b);
                for (int c = 0; c < vc; ++c) {
                    ha.insert(ha.end(), reinterpret_cast<const uint16_t*>(srca + size_t(vs + c) * H * 2),
                              reinterpret_cast<const uint16_t*>(srca + size_t(vs + c) * H * 2) + H);
                    hb.insert(hb.end(), reinterpret_cast<const uint16_t*>(srcb + size_t(vs + c) * H * 2),
                              reinterpret_cast<const uint16_t*>(srcb + size_t(vs + c) * H * 2) + H);
                }
            }
            {   // conv1d [4, 10240] channel-major (w[c*4+j]): the rank's channels
                const float* src = have_fixture ? reinterpret_cast<const float*>(L.fc)
                                                : reinterpret_cast<const float*>(model->shard(L.sc).tensor_data(*L.c));
                for (int c = 0; c < cc; ++c) {
                    const float* rc = src + size_t(chan[size_t(c)]) * 4;
                    hc.insert(hc.end(), rc, rc + 4);
                }
            }
            {   // dt/ssm_a [48]: the rank's V heads; norm [128] PER LAYER (the
                // norms differ by layer - the first version mirrored ONE layer's
                // norm for all 36, which the fixture's identical layers hid)
                const float* sdt = have_fixture ? reinterpret_cast<const float*>(L.fdt)
                                                : reinterpret_cast<const float*>(model->shard(L.sd).tensor_data(*L.dt));
                const float* ssa = have_fixture ? reinterpret_cast<const float*>(L.fsa)
                                                : reinterpret_cast<const float*>(model->shard(L.ss).tensor_data(*L.sa));
                for (int c = 0; c < vc; ++c) {
                    hdt.push_back(sdt[vs + c]);
                    hsa.push_back(ssa[vs + c]);
                }
                {
                    const uint8_t* src = have_fixture ? L.fnm : model->shard(L.sn).tensor_data(*L.nm);
                    hnm.resize(hnm.size() + S);
                    std::memcpy(hnm.data() + hnm.size() - S, src, S * 4);
                }
            }
        }
        auto up = [&](void** dst, const void* src, size_t bytes, const char* what) {
            return ck(cudaMalloc(dst, bytes), what, err) &&
                   ck(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice), what, err);
        };
        if (!up((void**)&s.wqkv, hqkv.data(), hqkv.size(), "wqkv") ||
            !up((void**)&s.wg, hg.data(), hg.size(), "wg") ||
            !up((void**)&s.wa, ha.data(), ha.size() * 2, "wa") ||
            !up((void**)&s.wb, hb.data(), hb.size() * 2, "wb") ||
            !up((void**)&s.wc, hc.data(), hc.size() * 4, "wc") ||
            !up((void**)&s.wdt, hdt.data(), size_t(n_gdn_layers_) * vc * 4, "wdt") ||
            !up((void**)&s.wsa, hsa.data(), size_t(n_gdn_layers_) * vc * 4, "wsa") ||
            !up((void**)&s.wnm, hnm.data(), hnm.size() * 4, "wnm"))
            { close(); return false; }

        // the state and the conv history, per GDN layer
        s.state = new float*[size_t(n_gdn_layers_)]{};
        s.conv = new float*[size_t(n_gdn_layers_)]{};
        const size_t st_fl = size_t(vc) * S * S;
        const size_t cv_fl = size_t(cc) * 3;
        for (int64_t i = 0; i < n_gdn_layers_; ++i) {
            if (!ck(cudaMalloc((void**)&s.state[i], (st_fl + cv_fl) * 4), "state", err))
                { close(); return false; }
            s.conv[i] = s.state[i] + st_fl;
            cudaMemsetAsync(s.state[i], 0, (st_fl + cv_fl) * 4, s.stream);
        }
        if (!ck(cudaMalloc((void**)&s.state_scratch, st_fl * 4), "state scratch", err) ||
            !ck(cudaMalloc((void**)&s.conv_scratch, cv_fl * 4), "conv scratch", err) ||
            !ck(cudaMalloc((void**)&s.qkv, size_t(kWindowCap) * cc * 4), "qkv", err) ||
            !ck(cudaMalloc((void**)&s.hb, size_t(kWindowCap) * cc * 4), "hb", err) ||
            !ck(cudaMalloc((void**)&s.gate, size_t(kWindowCap) * vc * 4), "gate", err) ||
            !ck(cudaMalloc((void**)&s.beta, size_t(kWindowCap) * vc * 4), "beta", err) ||
            !ck(cudaMalloc((void**)&s.z, size_t(kWindowCap) * vc * S * 4), "z", err) ||
            !ck(cudaMalloc((void**)&s.y, size_t(kWindowCap) * vc * S * 4), "y", err) ||
            !ck(cudaMalloc((void**)&s.x_dev, size_t(kWindowCap) * H * 4), "x_dev", err) ||
            !ck(cudaMalloc(&s.xq, strata::kernels::native_q8_1_bytes(H, int(kWindowCap))), "xq", err) ||
            !ck(cudaMalloc(&pcounters_[r], 16), "pcounters", err) ||
            !ck(cudaMalloc((void**)&n_keep_dev_[r], 4), "n_keep", err))
            { close(); return false; }
        cudaMemset(pcounters_[r], 0, 16);
        cudaMemset(n_keep_dev_[r], 0, 4);
        void* dp = nullptr;
        if (!ck(cudaHostGetDevicePointer(&dp, inbox_, 0), "inbox dev", err)) { close(); return false; }
        inbox_dev_r_[r] = static_cast<uint8_t*>(dp);
        if (!ck(cudaHostGetDevicePointer(&dp, g_abort_, 0), "g_abort dev", err)) { close(); return false; }
        g_abort_dev_r_[r] = static_cast<unsigned*>(dp);
        cudaStreamSynchronize(s.stream);
        vram_bytes_ += int64_t(kRanks) * int64_t(n_gdn_layers_) * (st_fl + cv_fl) * 4;
    }

    // 06 §5.3 (STRATA_TP5_TRACE=1): per-rank stamp buffers.  Off by default.
    trace_on_ = [] { const char* v = std::getenv("STRATA_TP5_TRACE"); return v != nullptr && std::atoi(v) != 0; }();
    if (trace_on_) {
        for (int r = 0; r < kRanks; ++r) {
            On on(gpu_of_rank(r));
            if (!ck(cudaMalloc((void**)&trace_dev_[r], 24), "trace stamps", err)) { close(); return false; }
            cudaMemset(trace_dev_[r], 0, 24);
        }
    }

    // 06 §4.2/M3-for-GDN (STRATA_TP5_GDN_GRAPH=1): the per-(layer,bucket) graphs.
    // Opt-in like the FFN tier's M3; the eager path is untouched when off.
    graph_ = graph_mode_default();
    {   // 06 §6.3 launch proof: host-side tally only, the FFN tier's env
        const char* lc = std::getenv("STRATA_TP5_LAUNCH_COUNT");
        launch_count_on_ = lc && (lc[0] == '1' || lc[0] == 'y' || lc[0] == 'Y');
        for (int r = 0; r < kRanks; ++r) graph_launches_[r] = 0;
        graph_eager_fallbacks_ = 0;
    }
    if (graph_) {
        if (!ck(cudaHostAlloc((void**)&x_stage_, size_t(kWindowCap) * H * 4,
                              cudaHostAllocPortable), "x stage", err) ||
            !ck(cudaHostAlloc((void**)&step_stage_, sizeof(tp5::StepState),
                              cudaHostAllocPortable), "step stage", err))
            { close(); return false; }
        std::memset(x_stage_, 0, size_t(kWindowCap) * H * 4);
        *step_stage_ = tp5::StepState{0, 0, 0};
        for (int r = 0; r < kRanks; ++r) {
            On on(gpu_of_rank(r));
            if (!ck(cudaMalloc((void**)&step_dev_[r], sizeof(tp5::StepState)), "step dev", err))
                { close(); return false; }
            cudaMemset(step_dev_[r], 0, sizeof(tp5::StepState));
        }
        for (int r = 0; r < kRanks; ++r)
            for (int64_t gi = 0; gi < n_gdn_layers_; ++gi)
                for (int b = 0; b < graph_buckets_; ++b) {
                    if (!capture_gdn_rank(r, gi, b, err)) { close(); return false; }
                }
    }

    open_ = true;
    return true;
}

bool Tp5GdnTier::graph_mode_default() {
    const char* v = std::getenv("STRATA_TP5_GDN_GRAPH");
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

bool Tp5GdnTier::run(int64_t l, const float* x, int64_t n_tok, float* y, std::string& err,
                     const float* state0, int64_t commit_n, int64_t pos0) {
    if (!open_) { err = "tp5 gdn: tier not open"; return false; }
    if (!is_gdn_layer(l)) { err = "tp5 gdn: layer is not a GDN layer"; return false; }
    if (n_tok <= 0 || n_tok > kWindowCap) { err = "tp5 gdn: window out of range"; return false; }

    t_ = Timings{};
    const int64_t gi = lay_of(l);
    if (gi >= n_gdn_layers_) { err = "tp5 gdn: layer index out of range"; return false; }

    ++epoch_;
    const int64_t epoch = int64_t(epoch_);
    const uint32_t bank = uint32_t((epoch - 1) & 1);
    const auto t0 = std::chrono::steady_clock::now();
    if (epoch > 2) {
        const int64_t prev = epoch - 2;
        if (int64_t(credits_[bank].consumed_epoch) < prev) {
            err = "tp5 gdn: bank reuse before credit (protocol)";
            return false;
        }
    }

    if (graph_ && state0 == nullptr && n_tok <= graph_buckets_)   // the graph path (M3-for-GDN)
        return run_graph(l, x, n_tok, y, err, epoch, bank, gi, t0);
    if (graph_) ++graph_eager_fallbacks_;   // the test path or a >5-row window: eager

    for (int r = 0; r < kRanks; ++r) {
        RankState& s = rs_[r];
        On on(gpu_of_rank(r));
        const cudaStream_t st = s.stream;
        const int vc = v_count_[r], cc = c_count_[r];
        const size_t st_fl = size_t(vc) * S * S, cv_fl = size_t(cc) * 3;
        if (!ck(cudaMemcpyAsync(s.x_dev, x, size_t(n_tok) * H * 4, cudaMemcpyHostToDevice, st),
                "x copy", err))
            return false;
        // the recurrence runs on a SCRATCH copy of the state: the window's rows
        // include rejected ones, and the state must stay untouched until the commit
        // (the engine's own contract - `self_commit` only for one-token windows).
        // `state0` (TEST ONLY): the gate's initial state - the engine's is zero and
        // the gate's is random, so it is copied in (the rank's own slice: V heads
        // [vs, vs+vc) of the 48-head state, then the rank's conv channels).
        {
            if (state0 != nullptr) {
                // the kernel's state layout is [row][head][col]: element (r, h, c)
                // sits at state[(r*h_v + h)*S + c] - a head's 128 rows are NOT
                // contiguous (each row steps h_v*S).  The host-side gather copies
                // the rank's V heads row by row.
                {
                    std::vector<float> tmp{std::vector<float>(size_t(vc) * S)};
                    for (int rr = 0; rr < S; ++rr) {
                        for (int i = 0; i < vc; ++i)
                            std::memcpy(tmp.data() + size_t(i) * S,
                                        state0 + size_t((size_t(rr) * HV + (v_start_[r] + i)) * S),
                                        S * 4);
                        if (!ck(cudaMemcpy(s.state_scratch + size_t(rr) * vc * S, tmp.data(),
                                           size_t(vc) * S * 4, cudaMemcpyHostToDevice),
                                "state0 copy", err)) {
                            return false;
                        }
                    }
                }
                for (int c = 0; c < cc; ++c)
                    if (!ck(cudaMemcpy(s.conv_scratch + size_t(c) * 3,
                                       state0 + size_t(HV) * S * S + size_t(chan_[r][c]) * 3,
                                       3 * 4, cudaMemcpyHostToDevice),
                            "conv0 copy", err))
                        return false;
            } else {
                cudaMemcpyAsync(s.state_scratch, s.state[gi], st_fl * 4, cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(s.conv_scratch, s.conv[gi], cv_fl * 4, cudaMemcpyDeviceToDevice, st);
            }
        }
        if (!ck(cudaMemsetAsync(pcounters_[r] + (epoch & 3), 0, 4, st), "pc zero", err))
            return false;
        if (trace_on_)   // 06 §5.3: the rank's own chain, start (before the quantize)
            strata::kernels::gpu_stamp(trace_dev_[r], 0, st);
        strata::kernels::native_quantize_q8_1(s.x_dev, s.xq, (int) H, (int) n_tok, st);
        // the qkv projection: the rank's gathered channel rows
        {
            const uint8_t* wq = s.wqkv + size_t(lay_qkv_span_[size_t(gi) * kRanks + r]);
            strata::kernels::native_mmvq(lay_qkv_type_[size_t(gi)], wq, s.xq, s.qkv, (int) H, cc,
                                         (int) n_tok, st);
        }
        // Probe only.  The engine dump (STRATA_TP5_GDN_XDUMP) captures h_gx_
        // and qkv_L_ at the same doorbell.  This writes the rank's projected
        // channels of that same x.  Sync copy is probe-only; missing path does
        // not fail the window.
        if (r == 0 && l == 0 && (n_tok == 1 || n_tok == 2)) {
            const char* qdump = std::getenv("STRATA_TP5_GDN_QDUMP");
            if (qdump != nullptr && qdump[0] != '\0') {
                cudaStreamSynchronize(st);
                const size_t st_fl = size_t(vc) * S * S;
                std::vector<float> q((size_t) n_tok * (size_t) cc), stin(st_fl), cvin(size_t(cc) * 3);
                const bool ok = cudaMemcpy(q.data(), s.qkv, q.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                                cudaMemcpy(stin.data(), s.state_scratch, stin.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess &&
                                cudaMemcpy(cvin.data(), s.conv_scratch, cvin.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
                if (ok) {
                    char path[768];
                    std::snprintf(path, sizeof(path), "%s.pos%lld", qdump, (long long) pos0);
                    if (FILE* f = std::fopen(path, "w")) {
                        std::fprintf(f, "layer=%lld n_tok=%lld rank=0 c_count=%d v_count=%d v_start=%d pos0=%lld\n",
                                     (long long) l, (long long) n_tok, cc, vc, v_start_[r], (long long) pos0);
                        std::fprintf(f, "tier_qkv n=%zu\n", q.size());
                        for (float v : q) std::fprintf(f, " %.9g", v);
                        std::fprintf(f, "\ntier_state_in n=%zu\n", stin.size());
                        for (float v : stin) std::fprintf(f, " %.9g", v);
                        std::fprintf(f, "\ntier_conv_in n=%zu\n", cvin.size());
                        for (float v : cvin) std::fprintf(f, " %.9g", v);
                        std::fprintf(f, "\ndumped=1\n");
                        std::fclose(f);
                    }
                }
            }
        }
        // the conv (the rank's channels; the L2 norm's qk_heads = vc) on the scratch history
        // qk_heads = the L2's block count: the rank's q AND k regions (2*vc blocks
        // of S channels) - the engine's own 2*HK for its 16+16 heads
        strata::kernels::gdn_conv_l2_multi(s.conv_scratch, s.qkv, s.wc + size_t(gi) * cc * 4, s.hb, cc, 2 * vc,
                                           1e-6f, (int) n_tok, st, 0, false);
        // Probe only.  Same gate as the qkv dump above: this window's hb, before
        // the next layer reuses s.hb.  Sync copy stays inside the env gate.
        if (r == 0 && l == 0 && n_tok == 2) {
            const char* hdump = std::getenv("STRATA_TP5_GDN_HBDUMP");
            if (hdump != nullptr && hdump[0] != '\0') {
                cudaStreamSynchronize(st);
                std::vector<float> hb((size_t) n_tok * (size_t) cc);
                if (cudaMemcpy(hb.data(), s.hb, hb.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess) {
                    char hpath[768];
                    std::snprintf(hpath, sizeof(hpath), "%s.pos%lld", hdump, (long long) pos0);
                    if (FILE* f = std::fopen(hpath, "w")) {
                        std::fprintf(f, "layer=%lld n_tok=%lld rank=0 c_count=%d pos0=%lld\n", (long long) l,
                                     (long long) n_tok, cc, (long long) pos0);
                        std::fprintf(f, "tier_hb n=%zu\n", hb.size());
                        for (float v : hb) std::fprintf(f, " %.9g", v);
                        std::fprintf(f, "\nhb_dumped=1\n");
                        std::fclose(f);
                    }
                }
            }
        }
        // the a/b projections (BF16 gemv on the raw x rows) - gate/beta [n_tok, vc]
        // wa/wb are uint16_t*: the layer offset is in uint16 UNITS (vc*H per
        // layer) - the first version multiplied by 2 for bytes and read 2 layers
        // ahead (the fixture's byte-identical layers hid it; the engine's layer
        // 26 read 2 layers ahead and the last layers read past the allocation)
        strata::kernels::gdn_ab_multi(s.x_dev, s.wa + size_t(gi) * vc * H,
                                      s.wb + size_t(gi) * vc * H,
                                      s.wdt + size_t(gi) * vc, s.wsa + size_t(gi) * vc,
                                      s.gate, s.beta, (int) H, vc, (int) n_tok, st);
        {   // the gate projection (the rank's V-head rows of ZV) at its
            // original site
            const uint8_t* w = s.wg + size_t(lay_g_span_[size_t(gi) * kRanks + r]);
            strata::kernels::native_mmvq(lay_g_type_[size_t(gi)], w, s.xq, s.z, (int) H, vc * S,
                                         (int) n_tok, st);
        }
        // the recurrence: h_k = vc (the q/k rows are in V-head order, so
        // head % h_k is the identity), h_v = vc, the rank's own scratch state
        strata::kernels::gdn_step_norm_multi(s.state_scratch, s.hb, cc, s.gate, s.beta, s.z,
                                             (const float*) s.wnm + size_t(gi) * S, 1e-6f, s.y, vc, vc,
                                             (int) n_tok, nullptr, st, 0, nullptr);
        if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
            err = std::string("tp5 gdn: rank ") + std::to_string(r) + " chain: " + cudaGetErrorString(le);
            return false;
        }
        // publish the rank's recurrence-output slice (the out projection is the
        // graph's, on the full y - see the kernel's comment)
        {
            const unsigned blocks_y = unsigned((v_count_[r] * S + 255) / 256);
            publish_partial_kernel<<<dim3(blocks_y, unsigned(n_tok)), 256, 0, st>>>(
                s.y, inbox_dev_r_[r], gdn_inbox_bytes(int(kWindowCap)),
                epoch, bank, r, v_count_[r],
                pcounters_[r] + (epoch & 3), nullptr);
            if (trace_on_) {  // 06 §5.3: ... end; slot 2 = the per-stage fold stamp
                strata::kernels::gpu_stamp(trace_dev_[r], 1, st);
                strata::kernels::gpu_stamp(trace_dev_[r], 2, st);
            }
            if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
                err = std::string("tp5 gdn: publish launch (rank ") + std::to_string(r) +
                      "): " + cudaGetErrorString(le);
                return false;
            }
        }
        (void) commit_n;
    }
    return relay_tail(epoch, bank, n_tok, y, t0, err);
}


bool Tp5GdnTier::capture_gdn_rank(int r, int64_t gi, int bucket, std::string& err) {
    // The graph for (layer gi, n_tok = bucket+1) on rank r: the SAME chain the
    // eager run() issues, with the layer's pointers baked in (they are fixed for
    // the pair) and the varying inputs - x, epoch, bank - riding pinned staging
    // the graph's H2D nodes refresh per launch.  The state/conv scratch D2D and
    // the counter memset are graph nodes, ordered before their consumers.
    On on(gpu_of_rank(r));
    RankState& s = rs_[r];
    const cudaStream_t st = s.stream;
    const int vc = v_count_[r], cc = c_count_[r];
    const int n_tok = bucket + 1;
    const size_t st_fl = size_t(vc) * S * S, cv_fl = size_t(cc) * 3;
    const size_t slot = size_t(gi) * graph_buckets_ + size_t(bucket);

    *step_stage_ = tp5::StepState{1, 0, 0};
    std::memset(x_stage_, 0, size_t(n_tok) * H * 4);

    if (!ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "gdn begin cap", err))
        return false;
    // slot 0 of the publish counter: the graph's first node zeroes it every
    // replay, so the eager path's (epoch & 3) rotation is not needed here (the
    // graph serializes the memset with the publish that reads it)
    ck(cudaMemsetAsync(pcounters_[r], 0, 4, st), "gdn cap pc zero", err);
    ck(cudaMemcpyAsync(s.x_dev, x_stage_, size_t(n_tok) * H * 4,
                       cudaMemcpyHostToDevice, st), "gdn cap x", err);
    ck(cudaMemcpyAsync(step_dev_[r], step_stage_, sizeof(tp5::StepState),
                       cudaMemcpyHostToDevice, st), "gdn cap step", err);
    ck(cudaMemcpyAsync(s.state_scratch, s.state[gi], st_fl * 4,
                       cudaMemcpyDeviceToDevice, st), "gdn cap state", err);
    ck(cudaMemcpyAsync(s.conv_scratch, s.conv[gi], cv_fl * 4,
                       cudaMemcpyDeviceToDevice, st), "gdn cap conv", err);
    strata::kernels::native_quantize_q8_1(s.x_dev, s.xq, (int) H, (int) n_tok, st);
    {
        const uint8_t* wq = s.wqkv + size_t(lay_qkv_span_[size_t(gi) * kRanks + r]);
        strata::kernels::native_mmvq(lay_qkv_type_[size_t(gi)], wq, s.xq, s.qkv, (int) H, cc,
                                     (int) n_tok, st);
    }
    strata::kernels::gdn_conv_l2_multi(s.conv_scratch, s.qkv, s.wc + size_t(gi) * cc * 4, s.hb, cc, 2 * vc,
                                       1e-6f, (int) n_tok, st, 0, false);
    strata::kernels::gdn_ab_multi(s.x_dev, s.wa + size_t(gi) * vc * H,
                                  s.wb + size_t(gi) * vc * H,
                                  s.wdt + size_t(gi) * vc, s.wsa + size_t(gi) * vc,
                                  s.gate, s.beta, (int) H, vc, (int) n_tok, st);
    {   // the gate projection at its original site
        const uint8_t* w = s.wg + size_t(lay_g_span_[size_t(gi) * kRanks + r]);
        strata::kernels::native_mmvq(lay_g_type_[size_t(gi)], w, s.xq, s.z, (int) H, vc * S, (int) n_tok, st);
    }
    strata::kernels::gdn_step_norm_multi(s.state_scratch, s.hb, cc, s.gate, s.beta, s.z,
                                         (const float*) s.wnm + size_t(gi) * S, 1e-6f, s.y, vc, vc,
                                         (int) n_tok, nullptr, st, 0, nullptr);
    {
        const unsigned blocks_y = unsigned((v_count_[r] * S + 255) / 256);
        publish_partial_kernel<<<dim3(blocks_y, unsigned(n_tok)), 256, 0, st>>>(
            s.y, inbox_dev_r_[r], gdn_inbox_bytes(int(kWindowCap)),
            0, 0, r, v_count_[r],
            pcounters_[r], step_dev_[r]);
        if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
            err = std::string("tp5 gdn: graph publish launch: ") + cudaGetErrorString(le);
            return false;
        }
    }
    cudaError_t e = cudaStreamEndCapture(st, &graphs_[r][slot]);
    if (e != cudaSuccess) {
        err = std::string("tp5 gdn: end cap (rank ") + std::to_string(r) + ", layer " +
              std::to_string(gi) + ", bucket " + std::to_string(bucket) + "): " + cudaGetErrorString(e);
        return false;
    }
    e = cudaGraphInstantiate(&gexecs_[r][slot], graphs_[r][slot], nullptr, nullptr, 0);
    if (e != cudaSuccess) {
        err = std::string("tp5 gdn: instantiate (rank ") + std::to_string(r) + ", layer " +
              std::to_string(gi) + ", bucket " + std::to_string(bucket) + "): " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool Tp5GdnTier::run_graph(int64_t l, const float* x, int64_t n_tok, float* y, std::string& err,
                           int64_t epoch, uint32_t bank, int64_t gi,
                           const std::chrono::steady_clock::time_point& t0) {
    const int bucket = int(n_tok) - 1;
    const size_t slot = size_t(gi) * graph_buckets_ + size_t(bucket);
    // fill every rank's staging first (host memory - no device toggles), then
    // launch all five graphs in one pass: the FFN tier's M3 lesson (the fill
    // loop's 5 device toggles were ~0.2 ms of the 0.57 ms per-layer ready)
    std::memcpy(x_stage_, x, size_t(n_tok) * H * 4);
    *step_stage_ = tp5::StepState{uint64_t(epoch), bank, 0};
    {
        On restore(gpu_of_rank(0));
        for (int r = 0; r < kRanks; ++r) {
            if (r != 0 && !ck(cudaSetDevice(gpu_of_rank(r)), "graph rank device", err)) return false;
            if (launch_count_on_) ++graph_launches_[r];
            if (!ck(cudaGraphLaunch(gexecs_[r][slot], rs_[r].stream), "gdn graph launch", err))
                return false;
            if (!ck(cudaGetLastError(), "gdn graph launch status", err)) return false;
        }
    }
    // the same CPU relay tail as the eager path (wait ready, gather, credit)
    return relay_tail(epoch, bank, n_tok, y, t0, err);
}

bool Tp5GdnTier::sync_state(int64_t l, const float* engine_state, std::string& err) {
    if (!open_) { err = "tp5 gdn: tier not open"; return false; }
    if (!is_gdn_layer(l)) { err = "tp5 gdn: layer is not a GDN layer"; return false; }
    const int64_t gi = lay_of(l);
    // the per-layer form kept for the fixture test (tests/hip/tp5_gdn_test.cpp):
    // the ENGINE uses the batched span form below - this one's 5 device syncs
    // and 128 row copies per layer are the 8.7 s/request the span form removes
    // the engine's state is on GPU0; the ranks' on their own - copy through the
    // host (a one-time cost after the prefill, not per window)
    std::vector<float> host{std::vector<float>(size_t(HV) * S * S + 10240 * 3)};
    {
        int prev = 0;
        cudaGetDevice(&prev);
        cudaSetDevice(0);
        if (!ck(cudaMemcpy(host.data(), engine_state, host.size() * 4, cudaMemcpyDeviceToHost),
                "state sync read", err))
            return false;
        cudaSetDevice(prev);
    }
    for (int r = 0; r < kRanks; ++r) {
        RankState& st = rs_[r];
        On on(gpu_of_rank(r));
        // the previous request's tier commit (or commit_light) may still be queued
        // on this rank's stream: the sync's synchronous H2D (the default stream)
        // would race it and the tier's state could end mixed.  Drain first (a
        // once-per-request cost).  CHECKED: an async kernel failure surfaces here
        // (HIP leaves it sticky otherwise, and the next unrelated launch's
        // launch_check picks it up and mis-attributes the crash).
        if (const cudaError_t se = cudaDeviceSynchronize(); se != cudaSuccess) {
            err = std::string("tp5 gdn: sync_state drain (rank ") + std::to_string(r) + "): " + cudaGetErrorString(se);
            return false;
        }
        const int vc = v_count_[r], cc = c_count_[r];
        const size_t st_fl = size_t(vc) * S * S;
        // [row][head][col]: the rank's (rr, i) = engine's (rr, vs+i)
        std::vector<float> tmp{std::vector<float>(size_t(vc) * S)};
        for (int rr = 0; rr < S; ++rr) {
            for (int i = 0; i < vc; ++i)
                std::memcpy(tmp.data() + size_t(i) * S,
                            host.data() + size_t((size_t(rr) * HV + (v_start_[r] + i)) * S),
                            S * 4);
            if (!ck(cudaMemcpy(st.state[gi] + size_t(rr) * vc * S, tmp.data(),
                               size_t(vc) * S * 4, cudaMemcpyHostToDevice),
                    "state sync write", err))
                return false;
        }
        // the conv history [cc][3]: the rank's channels via the chan map
        for (int c = 0; c < cc; ++c)
            if (!ck(cudaMemcpy(st.conv[gi] + size_t(c) * 3,
                               host.data() + size_t(HV) * S * S + size_t(chan_[r][c]) * 3,
                               3 * 4, cudaMemcpyHostToDevice),
                    "conv sync write", err))
                return false;
    }
    return true;
}

bool Tp5GdnTier::sync_state_span(int64_t l0, int64_t l1, const float* estate, uint64_t gdn_floats,
                                    std::string& err) {
    if (!open_) { err = "tp5 gdn: tier not open"; return false; }
    if (l1 <= l0) { err = "tp5 gdn: empty sync span"; return false; }
    if (gdn_floats < uint64_t(HV) * S * S + 10240 * 3) { err = "tp5 gdn: sync span stride too small"; return false; }
    // ONE drain of every rank's device before the first write: the previous
    // request's queued commits (or commit_lights) are what the per-layer form's
    // 180 cudaDeviceSynchronize calls were each waiting for - one wait covers
    // them all (nothing else queues onto the rank streams between these writes;
    // the caller has already waited the engine's own commit).
    On home(0); // restore the caller's device on success and every early error
    bool drained[kRanks] = {};
    for (int r = 0; r < kRanks; ++r) {
        const int device = gpu_of_rank(r);
        if (drained[device]) continue;
        On on(device);
        if (!ck(cudaDeviceSynchronize(), "state sync drain", err)) return false;
        drained[device] = true;
    }
    const size_t blk = size_t(HV) * S * S + 10240 * 3;
    size_t rk_max = 0;
    for (int r = 0; r < kRanks; ++r) rk_max = std::max(rk_max, size_t(v_count_[r]) * S * S + size_t(c_count_[r]) * 3);
    if (exp_stage_ == nullptr) {
        const cudaError_t e1 = cudaHostAlloc((void**)&exp_stage_, blk * 4, cudaHostAllocPortable);
        const cudaError_t e2 = e1 == cudaSuccess
                                   ? cudaHostAlloc((void**)&exp_rank_, rk_max * 4, cudaHostAllocPortable)
                                   : cudaSuccess;
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            if (exp_stage_) { cudaFreeHost(exp_stage_); exp_stage_ = nullptr; }
            if (exp_rank_) { cudaFreeHost(exp_rank_); exp_rank_ = nullptr; }
            err = std::string("tp5 gdn: state staging: ") +
                  cudaGetErrorString(e1 != cudaSuccess ? e1 : e2);
            return false;
        }
    }
    for (int64_t l = l0; l < l1; ++l) {
        if (!is_gdn_layer(l)) continue;
        const int64_t gi = lay_of(l);
        {   // the engine's layer block, one D2H per layer
            if (!ck(cudaMemcpy(exp_stage_, estate + size_t(gi) * gdn_floats, blk * 4,
                               cudaMemcpyDeviceToHost), "state sync read", err))
                return false;
        }
        for (int r = 0; r < kRanks; ++r) {
            RankState& st = rs_[r];
            On on(gpu_of_rank(r));
            const int vc = v_count_[r], cc = c_count_[r];
            const size_t st_fl = size_t(vc) * S * S;
            for (int rr = 0; rr < S; ++rr)
                for (int i = 0; i < vc; ++i)
                    std::memcpy(exp_rank_ + (size_t(rr) * vc + size_t(i)) * S,
                                exp_stage_ + (size_t(rr) * HV + size_t(v_start_[r] + i)) * S, size_t(S) * 4);
            for (int c = 0; c < cc; ++c)
                std::memcpy(exp_rank_ + st_fl + size_t(c) * 3,
                            exp_stage_ + size_t(HV) * S * S + size_t(chan_[r][c]) * 3, 3 * 4);
            if (!ck(cudaMemcpy(st.state[gi], exp_rank_, (st_fl + size_t(cc) * 3) * 4, cudaMemcpyHostToDevice),
                    "state sync write", err))
                return false;
        }
    }
    return true;
}

bool Tp5GdnTier::export_state_span(int64_t l0, int64_t l1, float* estate, uint64_t gdn_floats,
                                   std::string& err) {
    if (!open_) { err = "tp5 gdn: tier not open"; return false; }
    if (l1 <= l0) { err = "tp5 gdn: empty export span"; return false; }
    if (gdn_floats < uint64_t(HV) * S * S + 10240 * 3) { err = "tp5 gdn: export span stride too small"; return false; }
    if (estate == nullptr) { err = "tp5 gdn: export estate is null"; return false; }
    On home(0); // restore the caller's device on success and every early error

    // ONE drain of every RANK device before the first read: the tier's queued
    // commits (commit/commit_span/commit_light) are what the caller wants in
    // the export, and a CHECKED sync surfaces a failed rank kernel here (HIP
    // leaves it sticky otherwise and the next unrelated launch mis-attributes
    // the crash).  GPU0 is NOT device-synced: the caller's wait_commit already
    // ordered the engine side, and a verifier graph spinning on a HOST flag
    // (the doorbell wait) would deadlock a GPU0 device sync - the header's
    // precondition states this.
    bool drained[kRanks] = {};
    for (int r = 0; r < kRanks; ++r) {
        const int device = gpu_of_rank(r);
        if (drained[device]) continue;
        On on(device);
        if (!ck(cudaDeviceSynchronize(), "state export drain", err)) return false;
        drained[device] = true;
    }

    // the staging, allocated ONCE (kept to close() - no per-layer allocation):
    // one canonical layer block (the H2D source) and one rank's state+conv
    // (the D2H target - the largest rank's [vc*S*S + cc*3] floats).  Pinned.
    const size_t blk = size_t(HV) * S * S + 10240 * 3;
    size_t rk_max = 0;
    for (int r = 0; r < kRanks; ++r) rk_max = std::max(rk_max, size_t(v_count_[r]) * S * S + size_t(c_count_[r]) * 3);
    if (exp_stage_ == nullptr) {
        const cudaError_t e1 = cudaHostAlloc((void**)&exp_stage_, blk * 4, cudaHostAllocPortable);
        const cudaError_t e2 = e1 == cudaSuccess
                                    ? cudaHostAlloc((void**)&exp_rank_, rk_max * 4, cudaHostAllocPortable)
                                    : cudaSuccess;
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            if (exp_stage_) { cudaFreeHost(exp_stage_); exp_stage_ = nullptr; }
            if (exp_rank_) { cudaFreeHost(exp_rank_); exp_rank_ = nullptr; }
            err = std::string("tp5 gdn: export staging: ") +
                  cudaGetErrorString(e1 != cudaSuccess ? e1 : e2);
            return false;
        }
    }

    // the exact inverse of sync_state_span's gather, per layer: the five ranks'
    // disjoint V-head sets tile the engine's 48, so the scatter is a pure copy -
    // engine (rr, v_start+i) <- rank (rr, i), and the conv channels back through
    // the rank's own channel map.  ONE H2D per layer of the full canonical block
    // All V heads and the union of q/k/v channels are covered; no stale engine
    // state needs a read/modify/write. Only this block is uploaded, preserving
    // any stride tail and every layer outside the span.
    for (int64_t l = l0; l < l1; ++l) {
        if (!is_gdn_layer(l)) continue;
        const int64_t gi = lay_of(l);
        for (int r = 0; r < kRanks; ++r) {
            RankState& s = rs_[r];
            On on(gpu_of_rank(r));
            const int vc = v_count_[r], cc = c_count_[r];
            const size_t st_fl = size_t(vc) * S * S, cv_fl = size_t(cc) * 3;
            // ONE D2H of the rank's whole layer (the state and conv are ONE
            // allocation - conv sits at st_fl)
            if (!ck(cudaMemcpy(exp_rank_, s.state[gi], (st_fl + cv_fl) * 4, cudaMemcpyDeviceToHost),
                    "state export rank read", err))
                return false;
            // scatter the state: engine (rr, vs+i) <- rank (rr, i)
            for (int rr = 0; rr < S; ++rr)
                for (int i = 0; i < vc; ++i)
                    std::memcpy(exp_stage_ + (size_t(rr) * HV + size_t(v_start_[r] + i)) * S,
                                exp_rank_ + (size_t(rr) * vc + size_t(i)) * S, size_t(S) * 4);
            // scatter the conv history: engine chan_[r][c] <- rank c
            for (int c = 0; c < cc; ++c)
                std::memcpy(exp_stage_ + size_t(HV) * S * S + size_t(chan_[r][c]) * 3,
                            exp_rank_ + st_fl + size_t(c) * 3, 3 * 4);
        }
        {   // ONE H2D per layer of the full canonical block
            if (!ck(cudaMemcpy(estate + size_t(gi) * gdn_floats, exp_stage_, blk * 4,
                               cudaMemcpyHostToDevice), "state export write", err))
                return false;
        }
    }
    return true;
}

bool Tp5GdnTier::commit_light(int64_t l, std::string& err) {
    if (!open_) { err = "tp5 gdn: tier not open"; return false; }
    if (!is_gdn_layer(l)) { err = "tp5 gdn: layer is not a GDN layer"; return false; }
    const int64_t gi = lay_of(l);
    for (int r = 0; r < kRanks; ++r) {
        RankState& s = rs_[r];
        On on(gpu_of_rank(r));
        const cudaStream_t st = s.stream;
        const int vc = v_count_[r], cc = c_count_[r];
        // the run's intermediates hold THIS window's rows; the recurrence over
        // the single kept row from the TRUE state advances it in place
        n_keep_h_[r] = 1;
        if (!ck(cudaMemcpyAsync(n_keep_dev_[r], n_keep_h_ + r, 4, cudaMemcpyHostToDevice, st),
                "n_keep copy", err))
            return false;
        // HIP's one_token_self_commit() is off unless STRATA_ONE_TOKEN_COMMIT=1, so
        // the graph does NOT advance conv (commit=false).  The engine advances it
        // once in the commit graph.  This is the tier's one advance of the same row.
        // Verifier::commit skips a second tier replay when n_keep==1 && last_t_==1.
        // Skipping this left the tier hist a window behind (hb off ~5x).
        strata::kernels::gdn_conv_commit(s.conv[gi], s.qkv, cc, n_keep_dev_[r], st);
        strata::kernels::gdn_step_norm_multi(s.state[gi], s.hb, cc, s.gate, s.beta, s.z,
                                             (const float*) s.wnm + size_t(gi) * S, 1e-6f, s.y, vc, vc,
                                             1, n_keep_dev_[r], st, 0, nullptr);
        if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
            err = std::string("tp5 gdn: light commit (rank ") + std::to_string(r) + "): " + cudaGetErrorString(le);
            return false;
        }
    }
    return true;
}

bool Tp5GdnTier::commit(int64_t l, int64_t n_keep, const float* x, std::string& err, int64_t pos0) {
    if (!open_) { err = "tp5 gdn: tier not open"; return false; }
    if (!is_gdn_layer(l)) { err = "tp5 gdn: layer is not a GDN layer"; return false; }
    if (n_keep <= 0 || n_keep > kWindowCap) { err = "tp5 gdn: commit out of range"; return false; }
    const int64_t gi = lay_of(l);
    for (int r = 0; r < kRanks; ++r) {
        RankState& s = rs_[r];
        On on(gpu_of_rank(r));
        const cudaStream_t st = s.stream;
        const int vc = v_count_[r], cc = c_count_[r];
        if (!ck(cudaMemcpyAsync(s.x_dev, x, size_t(n_keep) * H * 4, cudaMemcpyHostToDevice, st),
                "x copy", err))
            return false;
        strata::kernels::native_quantize_q8_1(s.x_dev, s.xq, (int) H, (int) n_keep, st);
        {
            const uint8_t* wq = s.wqkv + size_t(lay_qkv_span_[size_t(gi) * kRanks + r]);
            strata::kernels::native_mmvq(lay_qkv_type_[size_t(gi)], wq, s.xq, s.qkv, (int) H, cc,
                                         (int) n_keep, st);
        }
        // Probe only.  This qkv is the commit's own re-projection of the x
        // pointer the verifier passed (h_gx_ at commit time, after later layers
        // have overwritten the window buffer).  gdn_conv_commit reads it next.
        // Sync copy stays inside the env gate.  Missing path does not fail.
        if (r == 0 && l == 0 && (n_keep == 1 || n_keep == 2)) {
            const char* cdump = std::getenv("STRATA_TP5_GDN_CQDUMP");
            if (cdump != nullptr && cdump[0] != '\0') {
                cudaStreamSynchronize(st);
                std::vector<float> q((size_t) n_keep * (size_t) cc);
                if (cudaMemcpy(q.data(), s.qkv, q.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess) {
                    char path[768];
                    std::snprintf(path, sizeof(path), "%s.pos%lld", cdump, (long long) pos0);
                    if (FILE* f = std::fopen(path, "w")) {
                        std::fprintf(f, "layer=0 n_keep=%lld rank=0 c_count=%d pos0=%lld\n",
                                     (long long) n_keep, cc, (long long) pos0);
                        std::fprintf(f, "commit_qkv n=%zu\n", q.size());
                        for (float v : q) std::fprintf(f, " %.9g", v);
                        std::fprintf(f, "\ncq_dumped=1\n");
                        std::fclose(f);
                    }
                }
            }
        }
        cudaMemcpyAsync(s.conv_scratch, s.conv[gi], size_t(cc) * 3 * 4, cudaMemcpyDeviceToDevice, st);
        // the conv history's replay: the last three of [hist | x_0..x_{n-1}]
        // (gdn_conv_l2's commit path is single-token only; the engine's own
        // commit uses this kernel)
        // the commit's n_keep on the device: the kernels read it as *n_keep.  The
        // copy must be from a PERSISTENT location (a stack address can outlive an
        // async copy) but SYNCHRONOUS copies stall the whole device 180x per
        // window (36 layers x 5 ranks) and starve the FFN rank sharing the GPU -
        // a pinned member buffer + an async copy is both safe and non-stalling
        n_keep_h_[r] = int32_t(n_keep);
        if (!ck(cudaMemcpyAsync(n_keep_dev_[r], n_keep_h_ + r, 4, cudaMemcpyHostToDevice, st),
                "n_keep copy", err))
            return false;
        strata::kernels::gdn_conv_commit(s.conv[gi], s.qkv, cc, n_keep_dev_[r], st);
        // hb is NOT needed for the state replay, but the conv output of the kept
        // rows IS (the recurrence reads h): recompute it (no history side effects)
        strata::kernels::gdn_conv_l2_multi(s.conv_scratch, s.qkv, s.wc + size_t(gi) * cc * 4, s.hb, cc, 2 * vc,
                                           1e-6f, (int) n_keep, st, 0, false);
        strata::kernels::gdn_ab_multi(s.x_dev, s.wa + size_t(gi) * vc * H,
                                      s.wb + size_t(gi) * vc * H,
                                      s.wdt + size_t(gi) * vc, s.wsa + size_t(gi) * vc,
                                      s.gate, s.beta, (int) H, vc, (int) n_keep, st);
        {
            const uint8_t* w = s.wg + size_t(lay_g_span_[size_t(gi) * kRanks + r]);
            strata::kernels::native_mmvq(lay_g_type_[size_t(gi)], w, s.xq, s.z, (int) H, vc * S,
                                         (int) n_keep, st);
        }
        // the recurrence over the KEPT rows from the TRUE state (the engine's own
        // commit contract): gdn_step_norm_multi with n_keep advances `state` in
        // place when n_keep is passed (its ALL_OUT=false + n_keep path)
        strata::kernels::gdn_step_norm_multi(s.state[gi], s.hb, cc, s.gate, s.beta, s.z,
                                             (const float*) s.wnm + size_t(gi) * S, 1e-6f, s.y, vc, vc,
                                             (int) n_keep, n_keep_dev_[r], st, 0, nullptr);
        if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
            err = std::string("tp5 gdn: commit (rank ") + std::to_string(r) + "): " + cudaGetErrorString(le);
            return false;
        }
    }
    return true;
}


bool Tp5GdnTier::commit_span(int64_t l0, int64_t l1, int64_t n_keep, const float* x_base,
                             size_t rows, std::string& err, int64_t pos0) {
    if (!open_) { err = "tp5 gdn: tier not open"; return false; }
    if (l1 <= l0) { err = "tp5 gdn: empty commit span"; return false; }
    if (n_keep <= 0 || n_keep > kWindowCap) { err = "tp5 gdn: commit out of range"; return false; }
    // the per-layer form above stays for the fixture test and the layer-split
    // fallback; the ENGINE's post-window commit is this rank-major batch: the
    // same chain per (layer, rank), but one device context per rank instead of
    // 36 layer passes each re-toggling all 5 devices.  The chains have no
    // cross-layer dependency (each layer reads its own doorbell rows and
    // writes its own state/conv slot), so the reordering is safe.
    for (int r = 0; r < kRanks; ++r) {
        RankState& s = rs_[r];
        On on(gpu_of_rank(r));
        const cudaStream_t st = s.stream;
        const int vc = v_count_[r], cc = c_count_[r];
        n_keep_h_[r] = int32_t(n_keep);
        if (!ck(cudaMemcpyAsync(n_keep_dev_[r], n_keep_h_ + r, 4, cudaMemcpyHostToDevice, st),
                "n_keep copy", err))
            return false;
        for (int64_t l = l0; l < l1; ++l) {
            if (!is_gdn_layer(l)) continue;   // the tier's own predicate: QSA layers
            // are exactly the non-GDN ones (l % 4 == 3)
            const int64_t gi = lay_of(l);
            const float* x = x_base + size_t(l) * rows;
            if (!ck(cudaMemcpyAsync(s.x_dev, x, size_t(n_keep) * H * 4, cudaMemcpyHostToDevice, st),
                    "x copy", err))
                return false;
            strata::kernels::native_quantize_q8_1(s.x_dev, s.xq, (int) H, (int) n_keep, st);
            {
                const uint8_t* wq = s.wqkv + size_t(lay_qkv_span_[size_t(gi) * kRanks + r]);
                strata::kernels::native_mmvq(lay_qkv_type_[size_t(gi)], wq, s.xq, s.qkv, (int) H, cc,
                                             (int) n_keep, st);
            }
            // Probe only (the per-layer form's CQDUMP, same gate): rank 0,
            // layer l0, kept rows 1-2.  Sync copy stays inside the env gate.
            if (r == 0 && l == l0 && (n_keep == 1 || n_keep == 2)) {
                const char* cdump = std::getenv("STRATA_TP5_GDN_CQDUMP");
                if (cdump != nullptr && cdump[0] != '\0') {
                    cudaStreamSynchronize(st);
                    std::vector<float> q((size_t) n_keep * (size_t) cc);
                    if (cudaMemcpy(q.data(), s.qkv, q.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess) {
                        char path[768];
                        std::snprintf(path, sizeof(path), "%s.pos%lld", cdump, (long long) pos0);
                        if (FILE* f = std::fopen(path, "w")) {
                            std::fprintf(f, "layer=%lld n_keep=%lld rank=0 c_count=%d pos0=%lld\n",
                                         (long long) l, (long long) n_keep, cc, (long long) pos0);
                            std::fprintf(f, "commit_qkv n=%zu\n", q.size());
                            for (float v : q) std::fprintf(f, " %.9g", v);
                            std::fprintf(f, "\ncq_dumped=1\n");
                            std::fclose(f);
                        }
                    }
                }
            }
            cudaMemcpyAsync(s.conv_scratch, s.conv[gi], size_t(cc) * 3 * 4, cudaMemcpyDeviceToDevice, st);
            // the conv history's replay: the last three of [hist | x_0..x_{n-1}]
            // (gdn_conv_l2's commit path is single-token only; the engine's own
            // commit uses this kernel)
            strata::kernels::gdn_conv_commit(s.conv[gi], s.qkv, cc, n_keep_dev_[r], st);
            // hb is NOT needed for the state replay, but the conv output of the kept
            // rows IS (the recurrence reads h): recompute it (no history side effects)
            strata::kernels::gdn_conv_l2_multi(s.conv_scratch, s.qkv, s.wc + size_t(gi) * cc * 4, s.hb, cc, 2 * vc,
                                               1e-6f, (int) n_keep, st, 0, false);
            strata::kernels::gdn_ab_multi(s.x_dev, s.wa + size_t(gi) * vc * H,
                                          s.wb + size_t(gi) * vc * H,
                                          s.wdt + size_t(gi) * vc, s.wsa + size_t(gi) * vc,
                                          s.gate, s.beta, (int) H, vc, (int) n_keep, st);
            {
                const uint8_t* w = s.wg + size_t(lay_g_span_[size_t(gi) * kRanks + r]);
                strata::kernels::native_mmvq(lay_g_type_[size_t(gi)], w, s.xq, s.z, (int) H, vc * S,
                                             (int) n_keep, st);
            }
            // the recurrence over the KEPT rows from the TRUE state (the engine's own
            // commit contract): gdn_step_norm_multi with n_keep advances `state` in
            // place when n_keep is passed (its ALL_OUT=false + n_keep path)
            strata::kernels::gdn_step_norm_multi(s.state[gi], s.hb, cc, s.gate, s.beta, s.z,
                                                 (const float*) s.wnm + size_t(gi) * S, 1e-6f, s.y, vc, vc,
                                                 (int) n_keep, n_keep_dev_[r], st, 0, nullptr);
            if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
                err = std::string("tp5 gdn: commit span (rank ") + std::to_string(r) + ", layer " +
                      std::to_string(l) + "): " + cudaGetErrorString(le);
                return false;
            }
        }
    }
    return true;
}


bool Tp5GdnTier::relay_tail(int64_t epoch, uint32_t bank, int64_t n_rows, float* y,
                            const std::chrono::steady_clock::time_point& t0, std::string& err) {
    const uint64_t stride = gdn_inbox_bytes(int(kWindowCap));
    const auto dl = std::chrono::steady_clock::now() + std::chrono::milliseconds(30000);
    volatile uint64_t* ready[kRanks];
    for (int r = 0; r < kRanks; ++r)
        ready[r] = reinterpret_cast<volatile uint64_t*>(
            inbox_ + (size_t(bank) * kRanks + size_t(r)) * stride +
            gdn_flag_off(int(Tp5GdnTier::kWindowCap)));
    if (trace_on_)   // 06 §5.3 E_RELAY: per-rank first-seen host stamps (no fold)
        for (int r = 0; r < kRanks; ++r) tr_.ready_seen_ns[r] = 0;
    int wait_rank = 0;
    while (true) {
        bool all = true;
        for (int r = 0; r < kRanks; ++r)
            if (*ready[r] >= uint64_t(epoch)) {
                if (trace_on_ && tr_.ready_seen_ns[r] == 0)
                    tr_.ready_seen_ns[r] = double(now_ns());   // the rank's ready_first
            } else {
                if (all) wait_rank = r;
                all = false;
            }
        if (all) break;
        if (std::chrono::steady_clock::now() > dl) {
            err = "tp5 gdn: ready timeout (a rank's partial never landed)";
            __atomic_store_n(g_abort_, 1u, __ATOMIC_RELEASE);
            return false;
        }
        for (int spin = 0; spin < 4096; ++spin) {
            if (*ready[wait_rank] >= uint64_t(epoch)) break;
            _mm_pause();
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (trace_on_) {   // 06 §5.3 E_RELAY: the host-clock endpoints, host ns
        t_.t0_host_ns = std::chrono::duration<double, std::nano>(
                            t0.time_since_epoch()).count();
        t_.ready_exit_host_ns = std::chrono::duration<double, std::nano>(
                                    t1.time_since_epoch()).count();
    }
    if (!collect_trace(err)) return false;
    if (!ck(cudaGetLastError(), "gdn relay status", err)) return false;
    {   // the gather: each rank's slice is DISJOINT (V heads [vs, vs+vc)) - copy,
        // not sum. Each rank slot is packed (cnt floats/row) and placed at off.
        const int ZV = HV * S;
        for (int r = 0; r < kRanks; ++r) {
            const float* p = reinterpret_cast<const float*>(
                inbox_ + (size_t(bank) * kRanks + size_t(r)) * stride);
            const int off = v_start_[r] * S, cnt = v_count_[r] * S;
            for (int64_t t = 0; t < n_rows; ++t)
                std::memcpy(y + size_t(t) * ZV + off, p + size_t(t) * cnt, size_t(cnt) * 4);
        }
    }
    const auto t2 = std::chrono::steady_clock::now();
    _mm_sfence();
    __atomic_store_n(&credits_[bank].consumed_epoch, uint64_t(epoch), __ATOMIC_RELEASE);
    t_.ready_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    t_.sum_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    t_.total_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool Tp5GdnTier::collect_trace(std::string& err) {
    // 06 §5.3 (STRATA_TP5_TRACE=1): same readback as the FFN tier's - the ready wait
    // has landed, so every end stamp is written.
    if (!trace_on_) return true;
    for (int r = 0; r < kRanks; ++r) {
        On on(gpu_of_rank(r));
        // the end stamp is QUEUED on the rank's stream after the publish; a blocking
        // cudaMemcpy on the default stream does NOT wait for it - in the engine the
        // rank's stream is already busy with the next layer, and the read raced ahead
        // (every span read 0).  Sync the stream the stamp lives on first.
        if (!ck(cudaStreamSynchronize(rs_[r].stream), "trace drain", err)) return false;
        // 06 §5.3: the fold anchor - see the FFN tier's comment (sync -> anchor -> memcpy).
        const long long anchor_ns = now_ns();
        unsigned long long h[3] = {0, 0, 0};
        // The optional primary-rank queue is legacy-blocking. Never introduce
        // a NULL-stream copy while GPU0's verifier is waiting on the host.
        if (!ck(cudaMemcpyAsync(h, trace_dev_[r], sizeof(h), cudaMemcpyDeviceToHost, rs_[r].stream),
                "trace read", err) ||
            !ck(cudaStreamSynchronize(rs_[r].stream), "trace read drain", err)) return false;
        if (h[1] >= h[0] && h[0] != 0)
            tr_.prod_span_ns[r] = double(h[1] - h[0]);
        else
            tr_.prod_span_ns[r] = 0;   // a missing/out-of-order stamp contributes nothing
        // E_RELAY's consumer side: slot 2 stamped right after slot 1 on the same
        // stream, and this point is after the stream sync, so the host clock at
        // `now` corresponds to the GPU clock at h[2].  The fold is per stage (the
        // GPU counters drift; a startup-only calibration would not see it):
        //   prod_end_host_ns[r] = now + (h[1] - h[2])   (both GPU-clock differences)
        // The sync's own latency (a few hundred us) shifts BOTH the fold anchor and
        // the host's ready stamp equally for every rank, so E_RELAY differences
        // between ranks stay exact.
        if (h[1] != 0 && h[2] >= h[1])
            tr_.prod_end_host_ns[r] = double(anchor_ns) + double(h[1]) - double(h[2]);
        else
            tr_.prod_end_host_ns[r] = 0;
    }
    return true;
}

}  // namespace strata::core
