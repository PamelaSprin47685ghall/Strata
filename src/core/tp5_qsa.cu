// src/core/tp5_qsa.cu - 06-phase4 §4.2.3: the 5-rank partial QSA tier (see the header).

#include "strata/core/tp5_qsa.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/bf16_gemv.hpp"

#include <immintrin.h>

#include <algorithm>
#include <memory>
#include <cstdio>
#include <cstring>
#include <set>

namespace strata::core {

namespace {

using strata::kernels::tp5::kRanks;

constexpr int H = 2560;        // n_embd
constexpr int HD = 256;        // head_dim
constexpr int IQ = 4, ID = 128;  // indexer query heads / key dim
constexpr int64_t kEntries = 8;   // n_tok cap * k? no - QSA has no entries; rows = n_tok

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("tp5 qsa: ") + what + ": " + cudaGetErrorString(e);
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

/// The RELAY producer for the mixer partials: copy the rank's partial rows
/// (n_tok rows of H floats) into the inbox payload, then the last arriving
/// block fences system-scope and publishes ready.  Same contract as the FFN
/// tier's publish_partial_kernel, with the QSA inbox's own row count.
__global__ void publish_partial_kernel(const float* __restrict__ y_dev,
                                       uint8_t* __restrict__ inbox_base, uint64_t stride,
                                       int64_t epoch, uint32_t bank, int rank, int nblocks,
                                       unsigned int* __restrict__ counter) {
    const int64_t rows = int64_t(gridDim.y);   // n_tok, the launch's own grid
    uint8_t* ib = inbox_base + (uint64_t(bank) * kRanks + rank) * stride;
    float* payload = reinterpret_cast<float*>(ib);
    volatile uint64_t* ready =
        reinterpret_cast<volatile uint64_t*>(ib + strata::kernels::tp5::Inbox::flag_off(int(Tp5QsaTier::kMaxTok)));
    const int64_t n = rows * H;
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + int(threadIdx.x);
    for (int64_t k = i; k < n; k += int64_t(gridDim.x) * blockDim.x) payload[k] = y_dev[k];
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        const unsigned arrived = atomicAdd(counter, 1u) + 1;
        if (arrived == unsigned(nblocks)) {
            __threadfence_system();
            *ready = uint64_t(epoch);
        }
    }
}

/// The five-partial sum, rank order, AVX2 - the FFN tier's own helper, verbatim
/// (the lane order is fixed, so the values are the scalar loop's bit for bit).
__attribute__((target("avx2")))
void sum5_avx2(const float* a, const float* b, const float* c, const float* d,
               const float* e, float* y, size_t elems) {
    const size_t vec = elems / 8 * 8;
    for (size_t i = 0; i < vec; i += 8) {
        __m256 acc = _mm256_loadu_ps(a + i);
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(b + i));
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(c + i));
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(d + i));
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(e + i));
        _mm256_storeu_ps(y + i, acc);
    }
    for (size_t i = vec; i < elems; ++i)
        y[i] = a[i] + b[i] + c[i] + d[i] + e[i];
}

}  // namespace

Tp5QsaTier::~Tp5QsaTier() { close(); }

int64_t Tp5QsaTier::vram_bytes() const { return vram_bytes_; }

void Tp5QsaTier::close() {
    if (!open_) return;
    __atomic_store_n(g_abort_, 1u, __ATOMIC_RELEASE);
    for (int r = 0; r < kRanks; ++r) {
        RankState& s = rs_[r];
        On on(gpu_of_rank(r));
        if (s.stream) { cudaStreamSynchronize(s.stream); cudaStreamDestroy(s.stream); }
        cudaFree(s.wq); cudaFree(s.wk); cudaFree(s.wv); cudaFree(s.wo);
        cudaFree(s.wiq); cudaFree(s.wik);
        cudaFree((void*)s.wqn); cudaFree((void*)s.wkn); cudaFree((void*)s.wiqn); cudaFree((void*)s.wikn);
        for (int64_t i = 0; i < n_qsa_layers_; ++i) {
            if (s.k_q) cudaFree(s.k_q[i]);
            if (s.v_q) cudaFree(s.v_q[i]);
            if (s.k_scale) cudaFree(s.k_scale[i]);
            if (s.v_scale) cudaFree(s.v_scale[i]);
            if (s.page_table) cudaFree(s.page_table[i]);
            if (s.idx_tail) cudaFree(s.idx_tail[i]);
            if (s.idx_dead) cudaFree(s.idx_dead[i]);
            if (s.idx_pooled) cudaFree(s.idx_pooled[i]);
            if (s.idx_block_pos) cudaFree(s.idx_block_pos[i]);
        }
        delete[] s.k_q; delete[] s.v_q; delete[] s.k_scale; delete[] s.v_scale;
        delete[] s.page_table; delete[] s.idx_tail; delete[] s.idx_dead;
        delete[] s.idx_pooled; delete[] s.idx_block_pos;
        cudaFree(s.qfull); cudaFree(s.qcur); cudaFree(s.kcur); cudaFree(s.vcur);
        cudaFree(s.qidx); cudaFree(s.idx_raw); cudaFree(s.scores); cudaFree(s.attn);
        cudaFree(s.attn32); cudaFree(s.sel); cudaFree(s.scratch_attn);
        cudaFree(s.xq); cudaFree(s.step_dev); cudaFree(s.pos_q_dev);
        cudaFree(s.pos_k_dev); cudaFree(s.pos_i_dev);
        cudaFree(s.y_partial); cudaFree(s.x_dev);
        cudaFree((void*)s.cos_tab); cudaFree((void*)s.sin_tab);
        if (pcounters_[r]) cudaFree(pcounters_[r]);
        if (trace_dev_[r]) cudaFree(trace_dev_[r]);
        std::memset(&s, 0, sizeof(s));
    }
    if (inbox_) cudaFreeHost(inbox_);
    if (credits_) cudaFreeHost(credits_);
    if (g_abort_) cudaFreeHost(g_abort_);
    inbox_ = nullptr; credits_ = nullptr; g_abort_ = nullptr;
    std::memset(trace_dev_, 0, sizeof(trace_dev_));
    open_ = false;
}

bool Tp5QsaTier::open(const std::vector<std::string>& gguf_shards, int64_t n_embd, std::string& err) {
    close();
    const char* fix_env = std::getenv("STRATA_TP5_QSA_FIXTURE");
    const bool have_fixture = fix_env != nullptr && *fix_env != '\0';
    if ((gguf_shards.empty() && !have_fixture) || n_embd != H) {
        err = "tp5 qsa: needs the model's shards (or STRATA_TP5_QSA_FIXTURE) and n_embd 2560";
        return false;
    }
    if (!strata::kernels::native_qsa_enabled() || !strata::kernels::native_qsa_indexer_enabled()) {
        err = "tp5 qsa: needs the native QSA kernels (run with --native)";
        return false;
    }
    // the geometry: 24 q heads / 2 kv heads, the plan's GQA-aligned split
    constexpr int NQ = 24, NKV = 2;
    for (int r = 0; r < kRanks; ++r) {
        const int qs = q_start_[r], qc = q_count_[r];
        if (qs / (NQ / NKV) != kv_head_[r] || qs + qc > NQ) {
            err = "tp5 qsa: the head table is not GQA aligned";
            return false;
        }
    }
    // the model's QSA layers: l % 4 == 3, 12 of them at 48 layers
    std::vector<int64_t> qsa_layers;
    for (int64_t l = 0; qsa_layers.size() < 12 && l < 48; ++l)
        if (l % 4 == 3) qsa_layers.push_back(l);
    n_qsa_layers_ = int64_t(qsa_layers.size());
    if (n_qsa_layers_ != 12) { err = "tp5 qsa: expected 12 QSA layers"; return false; }

    // ---- the weights.  Two sources:
    //   * the GGUF (the engine's own spelling - the caller hands in the resolved
    //     shard list), or
    //   * a FIXTURE directory (STRATA_TP5_QSA_FIXTURE): one .bin per tensor of
    //     QSA layer 3, blk_3_<name>.bin, the raw payload bytes.  A test binary's
    //     openat of the >32 GiB shard 1 is UNRELIABLE on this stack - the same
    //     call succeeds from bash, from a plain gcc binary and from the engine,
    //     and fails or works with the binary's own layout (an hour of bisection;
    //     the mechanism stayed unexplained).  The 22 MB fixture sidesteps it, and
    //     every layer's weights are byte-identical to layer 3's (the fixture is
    //     what the gate compares against).
    const char* fixture = fix_env;
    const bool use_fix = have_fixture;
    struct Lay {
        const strata::TensorInfo *q, *k, *v, *o, *iq, *ik, *qn, *kn, *iqn, *ikn;
        size_t sq, sk, sv, so, siq, sik;   // shard indices
        const uint8_t *fq, *fk, *fv, *fo, *fiq, *fik, *fqn, *fkn, *fiqn, *fikn;   // fixture bytes
    };
    std::vector<Lay> lays{std::vector<Lay>(size_t(n_qsa_layers_))};
    std::vector<std::vector<uint8_t>> fix;   // owns the fixture bytes
    std::unique_ptr<strata::GgufModel> model;
    if (fixture != nullptr && *fixture != '\0') {
        auto read_fix = [&](const char* name) -> std::vector<uint8_t> {
            const std::string path = std::string(fixture) + "/blk_3_" + name + ".bin";
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
        fix.resize(10);
        fix[0] = read_fix("attn_q_weight");        fix[1] = read_fix("attn_k_weight");
        fix[2] = read_fix("attn_v_weight");        fix[3] = read_fix("attn_output_weight");
        fix[4] = read_fix("indexer_q_proj_weight"); fix[5] = read_fix("indexer_k_proj_weight");
        fix[6] = read_fix("attn_q_norm_weight");   fix[7] = read_fix("attn_k_norm_weight");
        fix[8] = read_fix("indexer_q_norm_weight"); fix[9] = read_fix("indexer_k_norm_weight");
        for (const auto& b : fix)
            if (b.empty()) { err = "tp5 qsa: the fixture directory is missing a tensor"; return false; }
        for (size_t i = 0; i < lays.size(); ++i) {
            Lay& L = lays[i];
            L.fq = fix[0].data(); L.fk = fix[1].data(); L.fv = fix[2].data(); L.fo = fix[3].data();
            L.fiq = fix[4].data(); L.fik = fix[5].data();
            L.fqn = fix[6].data(); L.fkn = fix[7].data(); L.fiqn = fix[8].data(); L.fikn = fix[9].data();
        }
    } else {
        if (gguf_shards.empty()) { err = "tp5 qsa: no shards and no fixture"; return false; }
        model = std::make_unique<strata::GgufModel>(gguf_shards);
        for (size_t i = 0; i < lays.size(); ++i) {
            const std::string pre = "blk." + std::to_string(qsa_layers[i]) + ".";
            Lay& L = lays[i];
            auto get = [&](const char* suf, size_t& sh) -> const strata::TensorInfo* {
                return model->find(pre + suf, &sh);
            };
            const strata::TensorInfo* q = get("attn_q.weight", L.sq);
            const strata::TensorInfo* k = get("attn_k.weight", L.sk);
            const strata::TensorInfo* v = get("attn_v.weight", L.sv);
            const strata::TensorInfo* o = get("attn_output.weight", L.so);
            const strata::TensorInfo* iq = get("indexer.q_proj.weight", L.siq);
            const strata::TensorInfo* ik = get("indexer.k_proj.weight", L.sik);
            const strata::TensorInfo* qn = get("attn_q_norm.weight", L.sq);
            const strata::TensorInfo* kn = get("attn_k_norm.weight", L.sk);
            const strata::TensorInfo* iqn = get("indexer.q_norm.weight", L.siq);
            const strata::TensorInfo* ikn = get("indexer.k_norm.weight", L.sik);
            if (!q || !k || !v || !o || !iq || !ik || !qn || !kn || !iqn || !ikn) {
                err = "tp5 qsa: layer " + std::to_string(qsa_layers[i]) + " is missing a QSA tensor (q=" +
                      std::to_string(q != nullptr) + " k=" + std::to_string(k != nullptr) + " v=" +
                      std::to_string(v != nullptr) + " o=" + std::to_string(o != nullptr) + " iq=" +
                      std::to_string(iq != nullptr) + " ik=" + std::to_string(ik != nullptr) + " qn=" +
                      std::to_string(qn != nullptr) + " kn=" + std::to_string(kn != nullptr) + " iqn=" +
                      std::to_string(iqn != nullptr) + " ikn=" + std::to_string(ikn != nullptr) + ", shards " +
                      std::to_string(gguf_shards.size()) + ", model shards " + std::to_string(model ? model->size() : 999) + ")";
                return false;
            }
            L.q = q; L.k = k; L.v = v; L.o = o; L.iq = iq; L.ik = ik;
            L.qn = qn; L.kn = kn; L.iqn = iqn; L.ikn = ikn;
        }
    }

    // the RELAY inbox: kMaxTok rows per (bank, rank)
    const uint64_t stride = strata::kernels::tp5::Inbox::bytes(int(kMaxTok));
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

    // the pool geometry: the engine's own max_cells (64K) - the caller's context bound
    max_cells_ = 65536;
    pages_ = max_cells_ / 4;         // page_size 4 (one indexer block)
    pool_rows_ = pages_ * 4;         // single head: page * page_size rows
    cap_ = strata::kernels::qsa_selection_width(2051, strata::kernels::qsa_real_shapes());
    max_blocks_ = max_cells_ / 4 + 2;

    // the rope tables (host, float64 build) shared by every rank
    const int64_t n_rot = 64;
    std::vector<float> cos_tab(size_t(max_cells_) * (n_rot / 2)), sin_tab(cos_tab.size());
    strata::kernels::build_rope_table((int) n_rot, strata::kernels::rope_scaling(), (int) max_cells_,
                                      cos_tab.data(), sin_tab.data());

    // per-layer types (the projections mix Q2_0/Q3_K/IQ4_XS/... by layer; the
    // fixture's are layer 3's own - every layer's weights ARE layer 3's bytes)
    for (size_t i = 0; i < lays.size(); ++i) {
        lay_q_type_[i] = use_fix ? 42 : int(lays[i].q->type);
        lay_k_type_[i] = use_fix ? 11 : int(lays[i].k->type);
        lay_v_type_[i] = use_fix ? 13 : int(lays[i].v->type);
        lay_o_type_[i] = use_fix ? 12 : int(lays[i].o->type);
    }
    // per-rank spans of the packed blobs (q_count differs by rank).  Each blob is
    // 12 layers back to back, so a layer's offset is its index times the SINGLE-layer
    // span (lay_*_span_) - the tier's own numeric gate only ran QSA layer 3 (index 0,
    // offset 0) and missed the bug where the first engine window reached layer 7
    // (index 1) and read the weights at index * TOTAL, 12x out of bounds.
    for (int r = 0; r < kRanks; ++r) {
        q_span_[r] = 0;
        k_span_[r] = 0;
        v_span_[r] = 0;
        wo_span_[r] = 0;
        for (size_t i = 0; i < lays.size(); ++i) {
            int be = 0, bb = 0;
            strata::block_geometry(lay_q_type_[i], be, bb);
            const int64_t qs = int64_t(q_count_[r]) * 512 * (2560 / be) * bb;
            q_span_[r] += qs;
            strata::block_geometry(lay_k_type_[i], be, bb);
            const int64_t ks = int64_t(256) * (2560 / be) * bb;
            k_span_[r] += ks;
            strata::block_geometry(lay_v_type_[i], be, bb);
            const int64_t vs = int64_t(256) * (2560 / be) * bb;
            v_span_[r] += vs;
            strata::block_geometry(lay_o_type_[i], be, bb);
            const int64_t os = int64_t(2560) * (int64_t(q_count_[r]) * 256 / be) * bb;
            wo_span_[r] += os;
            if (i == 0) {   // the single-layer spans (every layer's is the same: same types? NO - the
                // types differ per layer, so the offsets must be accumulated per layer below)
            }
        }
    }
    // the per-layer offsets: layer i's blob offset is the SUM of the single-layer
    // spans of layers [0, i) - the types differ per layer, so this is a prefix sum
    for (int r = 0; r < kRanks; ++r) {
        int64_t qoff = 0, koff = 0, voff = 0, ooff = 0;
        for (size_t i = 0; i < lays.size(); ++i) {
            int be = 0, bb = 0;
            strata::block_geometry(lay_q_type_[i], be, bb);
            lay_q_span_[i * kRanks + r] = qoff;
            qoff += int64_t(q_count_[r]) * 512 * (2560 / be) * bb;
            strata::block_geometry(lay_k_type_[i], be, bb);
            lay_k_span_[i * kRanks + r] = koff;
            koff += int64_t(256) * (2560 / be) * bb;
            strata::block_geometry(lay_v_type_[i], be, bb);
            lay_v_span_[i * kRanks + r] = voff;
            voff += int64_t(256) * (2560 / be) * bb;
            strata::block_geometry(lay_o_type_[i], be, bb);
            lay_wo_span_[i * kRanks + r] = ooff;
            ooff += int64_t(2560) * (int64_t(q_count_[r]) * 256 / be) * bb;
        }
    }

    // ---- per-rank state
    for (int r = 0; r < kRanks; ++r) {
        RankState& s = rs_[r];
        On on(gpu_of_rank(r));
        if (!ck(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking), "stream", err))
            { close(); return false; }
        const int qc = q_count_[r], qs = q_start_[r], kvh = kv_head_[r];

        // the split projections, layer by layer, into one host staging blob per
        // tensor.  `bytes_of(T, ptr)` unifies the two sources: the GGUF's
        // TensorInfo (the model path) or the fixture's raw pointer (whose types
        // are layer 3's own - the fixture IS layer 3's bytes)
        auto bytes_of = [&](const strata::TensorInfo* T, const uint8_t* fixp, int& type_out) {
            struct S { const uint8_t* p; int type; };
            if (T != nullptr) {
                const strata::TensorInfo& t = *T;
                return S{nullptr, int(t.type)};   // the pointer comes from the shard below
            }
            return S{fixp, -1};   // the fixture: the type is layer 3's (recorded at read time)
        };
        (void) bytes_of;
        // the fixture's types (layer 3's own; the GGUF path reads them from the
        // TensorInfos)
        auto lay_type = [&](const strata::TensorInfo* T, int fixt) {
            return use_fix ? fixt : int(T->type);
        };
        // the fixture's layer-3 types, from the files' own sizes:
        // q=42 (Q2_0), k=11 (Q3_K), v=13 (Q5_K), o=12 (Q4_K), iq/ik=30 (BF16), norms=0 (F32)
        std::vector<uint8_t> hq, hk, hv, ho, hiq, hik;
        std::vector<float> hqn, hkn, hiqn, hikn;
        for (size_t i = 0; i < lays.size(); ++i) {
            const Lay& L = lays[i];
            // q: rows [qs*512, (qs+qc)*512) - contiguous (q|gate interleave per head)
            {
                int be = 0, bb = 0;
                strata::block_geometry(lay_type(L.q, 42), be, bb);
                const int64_t row0 = int64_t(qs) * 512, nrows = int64_t(qc) * 512;
                const int64_t blocks_per_row = 2560 / be;
                const uint8_t* src = use_fix ? L.fq : model->shard(L.sq).tensor_data(*L.q);
                src += size_t(row0) * blocks_per_row * bb;
                const size_t bytes = size_t(nrows) * blocks_per_row * bb;
                hq.insert(hq.end(), src, src + bytes);
            }
            // k/v: rows [kvh*256, +256)
            for (int which = 0; which < 2; ++which) {
                const strata::TensorInfo* T = which == 0 ? L.k : L.v;
                const uint8_t* fixp = which == 0 ? L.fk : L.fv;
                const size_t sh = which == 0 ? L.sk : L.sv;
                int be = 0, bb = 0;
                strata::block_geometry(lay_type(T, which == 0 ? 11 : 13), be, bb);
                const int64_t row0 = int64_t(kvh) * 256, nrows = 256;
                const int64_t blocks_per_row = 2560 / be;
                const uint8_t* src = use_fix ? fixp : model->shard(sh).tensor_data(*T);
                src += size_t(row0) * blocks_per_row * bb;
                const size_t bytes = size_t(nrows) * blocks_per_row * bb;
                if (which == 0) hk.insert(hk.end(), src, src + bytes);
                else hv.insert(hv.end(), src, src + bytes);
            }
            // out: the CONTRACTION slice - per output row (2560) the input columns
            // [qs*256, +qc*256), stride-extracted to a compact n_in = qc*256
            {
                int be = 0, bb = 0;
                strata::block_geometry(lay_type(L.o, 12), be, bb);
                const int64_t blocks_per_row = 6144 / be;
                const int64_t col0 = int64_t(qs) * 256 / be, ncols = int64_t(qc) * 256 / be;
                const uint8_t* src = use_fix ? L.fo : model->shard(L.so).tensor_data(*L.o);
                const size_t row_bytes = size_t(ncols) * bb;
                for (int64_t row = 0; row < 2560; ++row) {
                    const uint8_t* rs = src + size_t(row) * blocks_per_row * bb + size_t(col0) * bb;
                    ho.insert(ho.end(), rs, rs + row_bytes);
                }
            }
            // indexer q/k: mirrored whole (BF16)
            {
                const size_t bqi = size_t(2560) * 2 * 512;    // [2560, 512]
                const uint8_t* src = use_fix ? L.fiq : model->shard(L.siq).tensor_data(*L.iq);
                hiq.insert(hiq.end(), src, src + bqi);
                const size_t bki = size_t(2560) * 2 * 128;    // [2560, 128]
                const uint8_t* skr = use_fix ? L.fik : model->shard(L.sik).tensor_data(*L.ik);
                hik.insert(hik.end(), skr, skr + bki);
            }
            // norms: mirrored whole (F32) - a BYTE copy (the tensor_data pointers are
            // uint8_t*; the first version used float-iterator arithmetic and copied
            // 4x the bytes from past the tensor's end, which the gamma dump caught)
            {
                const uint8_t* src = use_fix ? L.fqn : model->shard(L.sq).tensor_data(*L.qn);
                hqn.resize(hqn.size() + 256);
                std::memcpy(hqn.data() + hqn.size() - 256, src, 256 * 4);
                const uint8_t* srck = use_fix ? L.fkn : model->shard(L.sk).tensor_data(*L.kn);
                hkn.resize(hkn.size() + 256);
                std::memcpy(hkn.data() + hkn.size() - 256, srck, 256 * 4);
                const uint8_t* srcq = use_fix ? L.fiqn : model->shard(L.siq).tensor_data(*L.iqn);
                hiqn.resize(hiqn.size() + 128);
                std::memcpy(hiqn.data() + hiqn.size() - 128, srcq, 128 * 4);
                const uint8_t* srckn = use_fix ? L.fikn : model->shard(L.sik).tensor_data(*L.ikn);
                hikn.resize(hikn.size() + 128);
                std::memcpy(hikn.data() + hikn.size() - 128, srckn, 128 * 4);
            }
        }
        // upload
        auto up = [&](void** dst, const void* src, size_t bytes, const char* what) {
            return ck(cudaMalloc(dst, bytes), what, err) &&
                   ck(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice), what, err);
        };
        if (!up((void**)&s.wq, hq.data(), hq.size(), "wq") ||
            !up((void**)&s.wk, hk.data(), hk.size(), "wk") ||
            !up((void**)&s.wv, hv.data(), hv.size(), "wv") ||
            !up((void**)&s.wo, ho.data(), ho.size(), "wo") ||
            !up((void**)&s.wiq, hiq.data(), hiq.size(), "wiq") ||
            !up((void**)&s.wik, hik.data(), hik.size(), "wik") ||
            !up((void**)&s.wqn, hqn.data(), hqn.size() * 4, "wqn") ||
            !up((void**)&s.wkn, hkn.data(), hkn.size() * 4, "wkn") ||
            !up((void**)&s.wiqn, hiqn.data(), hiqn.size() * 4, "wiqn") ||
            !up((void**)&s.wikn, hikn.data(), hikn.size() * 4, "wikn"))
            { close(); return false; }
        // the rope table (one per rank)
        if (!up((void**)&s.cos_tab, cos_tab.data(), cos_tab.size() * 4, "cos") ||
            !up((void**)&s.sin_tab, sin_tab.data(), sin_tab.size() * 4, "sin"))
            { close(); return false; }

        // the KV pools (int8: codes + one fp16 scale per 64), the page tables (identity),
        // the indexer mirrors - one per QSA layer
        s.k_q = new int8_t*[size_t(n_qsa_layers_)];
        s.v_q = new int8_t*[size_t(n_qsa_layers_)];
        s.k_scale = new uint16_t*[size_t(n_qsa_layers_)];
        s.v_scale = new uint16_t*[size_t(n_qsa_layers_)];
        s.page_table = new int32_t*[size_t(n_qsa_layers_)];
        s.idx_tail = new float*[size_t(n_qsa_layers_)];
        s.idx_dead = new float*[size_t(n_qsa_layers_)];
        s.idx_pooled = new float*[size_t(n_qsa_layers_)];
        s.idx_block_pos = new int32_t*[size_t(n_qsa_layers_)];
        std::vector<int32_t> ident{std::vector<int32_t>(size_t(pages_))};
        for (int64_t i = 0; i < pages_; ++i) ident[size_t(i)] = int32_t(i);
        const int64_t pooled_rows = max_cells_ / 4 + 2;
        for (int64_t i = 0; i < n_qsa_layers_; ++i) {
            if (!ck(cudaMalloc((void**)&s.k_q[i], size_t(pool_rows_) * HD), "k_q", err) ||
                !ck(cudaMalloc((void**)&s.v_q[i], size_t(pool_rows_) * HD), "v_q", err) ||
                !ck(cudaMalloc((void**)&s.k_scale[i], size_t(pool_rows_) * (HD / 64) * 2), "k_scale", err) ||
                !ck(cudaMalloc((void**)&s.v_scale[i], size_t(pool_rows_) * (HD / 64) * 2), "v_scale", err) ||
                !ck(cudaMalloc((void**)&s.page_table[i], size_t(pages_) * 4), "page_table", err) ||
                !ck(cudaMalloc((void**)&s.idx_tail[i], size_t(3) * ID * 4), "idx_tail", err) ||
                !ck(cudaMalloc((void**)&s.idx_dead[i], size_t(ID) * 4), "idx_dead", err) ||
                !ck(cudaMalloc((void**)&s.idx_pooled[i], size_t(pooled_rows) * ID * 4), "idx_pooled", err) ||
                !ck(cudaMalloc((void**)&s.idx_block_pos[i], 4), "idx_block_pos", err))
                { close(); return false; }
            cudaMemsetAsync(s.k_q[i], 0, size_t(pool_rows_) * HD, s.stream);
            cudaMemsetAsync(s.v_q[i], 0, size_t(pool_rows_) * HD, s.stream);
            cudaMemsetAsync(s.k_scale[i], 0, size_t(pool_rows_) * (HD / 64) * 2, s.stream);
            cudaMemsetAsync(s.v_scale[i], 0, size_t(pool_rows_) * (HD / 64) * 2, s.stream);
            cudaMemcpyAsync(s.page_table[i], ident.data(), ident.size() * 4, cudaMemcpyHostToDevice, s.stream);
            cudaMemsetAsync(s.idx_tail[i], 0, size_t(3) * ID * 4, s.stream);
            cudaMemsetAsync(s.idx_dead[i], 0, size_t(ID) * 4, s.stream);
            cudaMemsetAsync(s.idx_pooled[i], 0, size_t(pooled_rows) * ID * 4, s.stream);
            cudaMemsetAsync(s.idx_block_pos[i], 0, 4, s.stream);
        }

        // the scratch
        const size_t xq_bytes = strata::kernels::native_q8_1_bytes(2560, int(kMaxTok));
        strata::kernels::QsaShapes sr;
        sr.n_head = qc; sr.n_head_kv = 1; sr.head_dim = HD; sr.n_rot = 64;
        sr.idx_n_head = IQ; sr.idx_dim = ID; sr.idx_block = 4; sr.idx_top_k = 2048; sr.page_size = 4;
        const int64_t attn_scr = strata::kernels::qsa_decode_attn_scratch_floats(cap_, sr);
        if (!ck(cudaMalloc((void**)&s.qfull, size_t(kMaxTok) * qc * 2 * HD * 4), "qfull", err) ||
            !ck(cudaMalloc((void**)&s.qcur, size_t(kMaxTok) * qc * HD * 4), "qcur", err) ||
            !ck(cudaMalloc((void**)&s.kcur, size_t(kMaxTok) * HD * 4), "kcur", err) ||
            !ck(cudaMalloc((void**)&s.vcur, size_t(kMaxTok) * HD * 4), "vcur", err) ||
            !ck(cudaMalloc((void**)&s.qidx, size_t(kMaxTok) * IQ * ID * 4), "qidx", err) ||
            !ck(cudaMalloc((void**)&s.idx_raw, size_t(kMaxTok) * ID * 4), "idx_raw", err) ||
            !ck(cudaMalloc((void**)&s.scores, size_t(kMaxTok) * max_blocks_ * 4), "scores", err) ||
            !ck(cudaMalloc((void**)&s.attn, size_t(kMaxTok) * qc * HD * 4), "attn", err) ||
            !ck(cudaMalloc((void**)&s.attn32, size_t(kMaxTok) * qc * HD * 4), "attn32", err) ||
            !ck(cudaMalloc((void**)&s.sel, size_t(kMaxTok) * cap_ * 4), "sel", err) ||
            !ck(cudaMalloc((void**)&s.scratch_attn, size_t(kMaxTok) * attn_scr * 4), "scratch", err) ||
            !ck(cudaMalloc(&s.xq, xq_bytes), "xq", err) ||
            !ck(cudaMalloc((void**)&s.step_dev, size_t(kMaxTok) * strata::kernels::kStepCount * 4), "step", err) ||
            !ck(cudaMalloc((void**)&s.pos_q_dev, size_t(kMaxTok) * 6 * 4), "pos_q", err) ||
            !ck(cudaMalloc((void**)&s.pos_k_dev, size_t(kMaxTok) * 4), "pos_k", err) ||
            !ck(cudaMalloc((void**)&s.pos_i_dev, size_t(kMaxTok) * IQ * 4), "pos_i", err) ||
            !ck(cudaMalloc((void**)&s.y_partial, size_t(kMaxTok) * H * 4), "y_partial", err) ||
            !ck(cudaMalloc((void**)&s.x_dev, size_t(kMaxTok) * H * 4), "x_dev", err) ||
            !ck(cudaMalloc(&pcounters_[r], 16), "pcounters", err))
            { close(); return false; }
        cudaMemset(pcounters_[r], 0, 16);
        // the inbox device alias
        void* dp = nullptr;
        if (!ck(cudaHostGetDevicePointer(&dp, inbox_, 0), "inbox dev", err)) { close(); return false; }
        inbox_dev_r_[r] = static_cast<uint8_t*>(dp);
        if (!ck(cudaHostGetDevicePointer(&dp, g_abort_, 0), "g_abort dev", err)) { close(); return false; }
        g_abort_dev_r_[r] = static_cast<unsigned*>(dp);
        cudaStreamSynchronize(s.stream);
    }

    // the measurement log's total (the pools dominate: 12 layers x 5 ranks x the
    // single-head int8 K/V pool of the full 64K cells)
    {
        int64_t total = 0;
        for (int r = 0; r < kRanks; ++r)
            for (int64_t q = 0; q < n_qsa_layers_; ++q) {
                total += 2 * int64_t(max_cells_) * HD;   // k_q + v_q
                total += 2 * int64_t((max_cells_ + 3) / 4 + 2) * ID * 4;   // idx_pooled (+ idx_tail/dead are small)
            }
        vram_bytes_ = total;
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

    open_ = true;
    return true;
}

bool Tp5QsaTier::run(int64_t l, const float* x, int64_t n_tok, const int32_t* step,
                     const int32_t* pos_q, const int32_t* pos_k, const int32_t* pos_i,
                     float* y, std::string& err) {
    if (!open_) { err = "tp5 qsa: tier not open"; return false; }
    if (!is_qsa_layer(l)) { err = "tp5 qsa: layer is not a QSA layer"; return false; }
    if (n_tok <= 0 || n_tok > kMaxTok) { err = "tp5 qsa: window out of range"; return false; }
    t_ = Timings{};
    const int64_t qi = l / 4;   // the QSA layer's index in the sequence 3,7,...,47
    if (qi >= n_qsa_layers_) { err = "tp5 qsa: layer index out of range"; return false; }

    ++epoch_;
    const int64_t epoch = int64_t(epoch_);
    const uint32_t bank = uint32_t((epoch - 1) & 1);
    const auto t0 = std::chrono::steady_clock::now();

    // bank reuse guard: the previous use of this bank must have its credit
    if (epoch > 2) {
        const int64_t prev = epoch - 2;
        if (int64_t(credits_[bank].consumed_epoch) < prev) {
            err = "tp5 qsa: bank reuse before credit (protocol)";
            return false;
        }
    }

    const strata::kernels::QsaShapes s_all = strata::kernels::qsa_real_shapes();
    const int64_t n = n_tok;

    for (int r = 0; r < kRanks; ++r) {
        RankState& s = rs_[r];
        On on(gpu_of_rank(r));
        const cudaStream_t st = s.stream;
        const int qc = q_count_[r];
        if (!ck(cudaMemcpyAsync(s.x_dev, x, size_t(n_tok) * H * 4, cudaMemcpyHostToDevice, st),
                "x copy", err))
            return false;
        if (!ck(cudaMemcpyAsync(s.step_dev, step, size_t(n_tok) * strata::kernels::kStepCount * 4,
                                cudaMemcpyHostToDevice, st), "step copy", err))
            return false;
        // the positions expand per head (rope_neox_apply wants one per row):
        // every head of the token sits at the same position, so the host expansion
        // is pos[t] repeated qc (or 1, or IQ) times - a fixed pattern per call
        // the positions expand per head (rope_neox_apply wants one per row):
        // every head of the token sits at the same position, so the host expansion
        // is pos[t] repeated qc (or 1, or IQ) times - a fixed pattern per call
        if (pos_q_exp_.size() < size_t(n_tok * qc)) pos_q_exp_.resize(size_t(n_tok * qc));
        for (int64_t t = 0; t < n_tok; ++t)
            for (int h = 0; h < qc; ++h) pos_q_exp_[size_t(t * qc + h)] = pos_q[t];
        if (pos_i_exp_.size() < size_t(n_tok * IQ)) pos_i_exp_.resize(size_t(n_tok * IQ));
        for (int64_t t = 0; t < n_tok; ++t)
            for (int h = 0; h < IQ; ++h) pos_i_exp_[size_t(t * IQ + h)] = pos_i[t];
        if (!ck(cudaMemcpyAsync(s.pos_q_dev, pos_q_exp_.data(), size_t(n_tok) * qc * 4,
                                cudaMemcpyHostToDevice, st), "pos_q copy", err) ||
            !ck(cudaMemcpyAsync(s.pos_k_dev, pos_k, size_t(n_tok) * 4,
                                cudaMemcpyHostToDevice, st), "pos_k copy", err) ||
            !ck(cudaMemcpyAsync(s.pos_i_dev, pos_i_exp_.data(), size_t(n_tok) * IQ * 4,
                                cudaMemcpyHostToDevice, st), "pos_i copy", err))
            return false;
        if (!ck(cudaMemsetAsync(pcounters_[r] + (epoch & 3), 0, 4, st), "pc zero", err))
            return false;
        if (trace_on_)   // 06 §5.3: the rank's own chain, start (before the quantize)
            strata::kernels::gpu_stamp(trace_dev_[r], 0, st);
        strata::kernels::native_quantize_q8_1(s.x_dev, s.xq, (int) H, (int) n_tok, st);
        // the q projection: the rank's contiguous row span of the 12288 rows
        const uint8_t* wq_l = s.wq + size_t(lay_q_span_[size_t(qi) * kRanks + r]);
        strata::kernels::native_mmvq(lay_q_type_[size_t(qi)], wq_l, s.xq, s.qfull, (int) H,
                                     (int) (qc * 2 * HD), (int) n_tok, st);
        // the q half's norm + rope (HIP: the two-launch path, as the verify window's own)
        {
            // copy the q half out of the q|gate rows (stride 2*HD)
            strata::kernels::copy_rows_strided(s.qcur, s.qfull, int64_t(n_tok * qc), HD, 2 * HD, st);
            strata::kernels::native_qsa_rms_norm_weighted(s.qcur, s.wqn + size_t(qi) * HD, s.qcur, (int) HD,
                                                          int(n_tok * qc), 1e-6f, st);
            strata::kernels::rope_neox_apply(s.qcur, s.qcur, int64_t(n_tok * qc), (int) HD, (int) s_all.n_rot,
                                             s.cos_tab, s.sin_tab, s.pos_q_dev, st);
        }
        // the kv projection: the rank's kv head's 256 rows
        strata::kernels::native_mmvq(lay_k_type_[size_t(qi)], s.wk + size_t(lay_k_span_[size_t(qi) * kRanks + r]), s.xq,
                                     s.kcur, (int) H, (int) HD, (int) n_tok, st);
        strata::kernels::native_mmvq(lay_v_type_[size_t(qi)], s.wv + size_t(lay_v_span_[size_t(qi) * kRanks + r]), s.xq,
                                     s.vcur, (int) H, (int) HD, (int) n_tok, st);
        {
            strata::kernels::native_qsa_rms_norm_weighted(s.kcur, s.wkn + size_t(qi) * HD, s.kcur, (int) HD,
                                                          int(n_tok), 1e-6f, st);
            strata::kernels::rope_neox_apply(s.kcur, s.kcur, int64_t(n_tok), (int) HD, (int) s_all.n_rot,
                                             s.cos_tab, s.sin_tab, s.pos_k_dev, st);
        }
        // the kv append into the rank's own single-head pool
        {
            strata::kernels::QsaShapes sr = s_all;
            sr.n_head = qc;
            sr.n_head_kv = 1;
            strata::kernels::kv_append_q8_steps(s.k_q[qi], s.v_q[qi], s.k_scale[qi], s.v_scale[qi],
                                                s.page_table[qi], s.step_dev, strata::kernels::kStepCount,
                                                s.kcur, s.vcur, (int) HD, int(n_tok), sr, st, nullptr);
        }
        // the indexer query projection (mirrored, BF16) + norm/rope
        strata::kernels::bf16_gemv_fp32_mmvf_multi(s.x_dev, H, s.wiq + size_t(qi) * IQ * ID * 2, s.qidx, int64_t(IQ * ID),
                                                   H, int64_t(IQ * ID), int(n_tok), st);
        {
            strata::kernels::native_qsa_rms_norm_weighted(s.qidx, s.wiqn + size_t(qi) * ID, s.qidx, (int) ID,
                                                          int(n_tok * IQ), 1e-6f, st);
            strata::kernels::rope_neox_apply(s.qidx, s.qidx, int64_t(n_tok * IQ), (int) ID, (int) s_all.n_rot,
                                             s.cos_tab, s.sin_tab, s.pos_i_dev, st);
        }
        // the indexer key projection (BF16) - the raw keys this window's rows produce
        strata::kernels::bf16_gemv_fp32_mmvf_multi(s.x_dev, H, s.wik + size_t(qi) * ID * 2, s.idx_raw, ID,
                                                   H, ID, int(n_tok), st);
        if (r == 0) {   // the commit's raw input: rank 0's rows, copied out before the append reads them
            auto& hr = h_idx_raw_[size_t(qi)];
            hr.resize(size_t(n_tok) * ID);
            if (!ck(cudaMemcpyAsync(hr.data(), s.idx_raw, size_t(n_tok) * ID * 4, cudaMemcpyDeviceToHost, st),
                    "idx_raw out", err))
                return false;
        }
        // the indexer append (the mirror's own state) + scores + topk
        {
            const strata::kernels::QsaIndexerBuffers ib{s.idx_tail[qi], s.idx_dead[qi], s.idx_pooled[qi],
                                                        s.idx_block_pos[qi]};
            strata::kernels::native_qsa_indexer_append_steps(s.idx_raw, s.step_dev + strata::kernels::kStepPos,
                                                             strata::kernels::kStepCount, (int) n_tok, 0,
                                                             s.wikn + size_t(qi) * ID, 1e-6f, ib, s_all, max_cells_,
                                                             strata::kernels::rope_scaling(), st);
            strata::kernels::qsa_block_scores(s.idx_pooled[qi], s.idx_dead[qi], s.qidx, s.step_dev,
                                              n_tok, max_blocks_, s_all, s.scores, st);
            strata::kernels::qsa_block_topk(s.scores, s.step_dev, n_tok, max_blocks_, cap_, s_all,
                                            s.sel, st);
        }
        // the attention over the rank's own single-head pool
        {
            strata::kernels::QsaAttnPools pools;
            pools.k_q = s.k_q[qi];
            pools.v_q = s.v_q[qi];
            pools.k_scale = s.k_scale[qi];
            pools.v_scale = s.v_scale[qi];
            pools.page_table = s.page_table[qi];
            strata::kernels::QsaShapes sr = s_all;
            sr.n_head = qc;
            sr.n_head_kv = 1;
            strata::kernels::qsa_decode_attn_batch(s.qcur, pools, s.sel, s.step_dev, cap_, sr,
                                                   s.scratch_attn, s.attn, n_tok, st);
        }
        // the gate (the rank's heads) and the out projection's partial.  The gate
        // kernel is ROW-LESS (q_full = [n_head][2*HD] of ONE token): the multi-row
        // window calls it once per row (row-independent, bitwise the same as the
        // single-row path - the first wiring passed n_tok*qc as n_head and only row
        // 0's heads were gated, rows 1+ read zeros).
        for (int64_t t = 0; t < n_tok; ++t)
            strata::kernels::native_qsa_gate_apply(s.attn + size_t(t) * qc * HD, s.qfull + size_t(t) * qc * 2 * HD,
                                                   s.attn32 + size_t(t) * qc * HD, qc, (int) HD, st);
        {
            // the out projection reads attn32's q8_1 (n_in = qc*256)
            strata::kernels::native_quantize_q8_1(s.attn32, s.xq, (int) (qc * HD), (int) n_tok, st);
            const uint8_t* wo_l = s.wo + size_t(lay_wo_span_[size_t(qi) * kRanks + r]);
            strata::kernels::native_mmvq(lay_o_type_[size_t(qi)], wo_l, s.xq, s.y_partial,
                                         (int) (qc * HD), (int) H, (int) n_tok, st);
        }
        if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
            err = std::string("tp5 qsa: rank ") + std::to_string(r) + " chain: " + cudaGetErrorString(le);
            return false;
        }
        // publish the partial rows through the RELAY inbox
        {
            const int blocks_y = int((size_t(n_tok) * H + 255) / 256);
            // the counter's target is the TOTAL block count (blocks_y * n_tok): the
            // launch's grid is 2D (x covers the row elements, y one block per row), so
            // a target of blocks_y alone released `ready` after the FIRST row's blocks
            // and the CPU sum read the other rows as zeros (the component gate ran
            // n_tok = 1, whose grid has no y redundancy, and missed it)
            publish_partial_kernel<<<dim3(blocks_y, unsigned(n_tok)), 256, 0, st>>>(
                s.y_partial, inbox_dev_r_[r], strata::kernels::tp5::Inbox::bytes(int(kMaxTok)),
                epoch, bank, r, blocks_y * int(n_tok), pcounters_[r] + (epoch & 3));
            if (trace_on_) {  // 06 §5.3: ... end; slot 2 = the per-stage fold stamp
                strata::kernels::gpu_stamp(trace_dev_[r], 1, st);
                strata::kernels::gpu_stamp(trace_dev_[r], 2, st);
            }
            if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
                err = std::string("tp5 qsa: publish launch (rank ") + std::to_string(r) +
                      "): " + cudaGetErrorString(le);
                return false;
            }
        }
    }
    (void) s_all;

    return relay_tail(epoch, bank, n_tok, y, t0, err);
}

bool Tp5QsaTier::commit(int64_t l, int64_t n_keep, const float* raw, const int32_t* commit_step,
                        std::string& err) {
    if (!open_) { err = "tp5 qsa: tier not open"; return false; }
    if (!is_qsa_layer(l)) { err = "tp5 qsa: layer is not a QSA layer"; return false; }
    if (n_keep < 0 || n_keep > kMaxTok) { err = "tp5 qsa: commit out of range"; return false; }
    const int64_t qi = l / 4;
    // the KV pool is position-addressed: run() already wrote the cells, and a rejected
    // row's cell is rewritten by the next window that owns that position.  The indexer
    // mirror needs the kept prefix's append REPLAYED (the window's run appended every
    // row, including the rejected ones - exactly the engine's own contract, where the
    // commit graph replays the append from the tail snapshot).
    if (n_keep == 0) return true;   // nothing kept: the next window rewrites from pos0
    for (int r = 0; r < kRanks; ++r) {
        RankState& s = rs_[r];
        On on(gpu_of_rank(r));
        const cudaStream_t st = s.stream;
        if (!ck(cudaMemcpyAsync(s.idx_raw, raw, size_t(n_keep) * ID * 4, cudaMemcpyHostToDevice, st),
                "raw copy", err))
            return false;
        if (!ck(cudaMemcpyAsync(s.step_dev, commit_step, size_t(n_keep) * strata::kernels::kStepCount * 4,
                                cudaMemcpyHostToDevice, st), "step copy", err))
            return false;
        const strata::kernels::QsaIndexerBuffers ib{s.idx_tail[qi], s.idx_dead[qi], s.idx_pooled[qi],
                                                    s.idx_block_pos[qi]};
        strata::kernels::native_qsa_indexer_append_steps(s.idx_raw, s.step_dev + strata::kernels::kStepPos,
                                                         strata::kernels::kStepCount, (int) n_keep, 0,
                                                         s.wikn + size_t(qi) * ID, 1e-6f, ib, strata::kernels::qsa_real_shapes(),
                                                         max_cells_, strata::kernels::rope_scaling(), st);
        if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
            err = std::string("tp5 qsa: commit (rank ") + std::to_string(r) + "): " + cudaGetErrorString(le);
            return false;
        }
    }
    return true;
}

bool Tp5QsaTier::relay_tail(int64_t epoch, uint32_t bank, int64_t n_rows, float* y,
                            const std::chrono::steady_clock::time_point& t0,
                            std::string& err) {
    const uint64_t stride = strata::kernels::tp5::Inbox::bytes(int(kMaxTok));
    const auto dl = std::chrono::steady_clock::now() + std::chrono::milliseconds(30000);
    volatile uint64_t* ready[kRanks];
    for (int r = 0; r < kRanks; ++r)
        ready[r] = reinterpret_cast<volatile uint64_t*>(
            inbox_ + (size_t(bank) * kRanks + size_t(r)) * stride +
            strata::kernels::tp5::Inbox::flag_off(int(Tp5QsaTier::kMaxTok)));
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
            err = "tp5 qsa: ready timeout (a rank's partial never landed)";
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
    collect_trace();
    {
        const int64_t nrow = n_rows;
        const float* p[kRanks];
        for (int r = 0; r < kRanks; ++r)
            p[r] = reinterpret_cast<const float*>(
                inbox_ + (size_t(bank) * kRanks + size_t(r)) * stride);
        sum5_avx2(p[0], p[1], p[2], p[3], p[4], y, size_t(nrow) * H);
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

void Tp5QsaTier::collect_trace() {
    // 06 §5.3 (STRATA_TP5_TRACE=1): same readback as the FFN tier's - the ready wait
    // has landed, so every end stamp is written.
    if (!trace_on_) return;
    for (int r = 0; r < kRanks; ++r) {
        On on(gpu_of_rank(r));
        // the end stamp is QUEUED on the rank's stream after the publish; sync the
        // stream the stamp lives on before the read (see the GDN tier's note)
        cudaStreamSynchronize(rs_[r].stream);
        // 06 §5.3: the fold anchor - see the FFN tier's comment (sync -> anchor -> memcpy).
        const long long anchor_ns = now_ns();
        unsigned long long h[3] = {0, 0, 0};
        if (cudaMemcpy(h, trace_dev_[r], 24, cudaMemcpyDeviceToHost) != cudaSuccess) {
            (void) cudaGetLastError();
            tr_.prod_span_ns[r] = 0;
            continue;
        }
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
}

}  // namespace strata::core
