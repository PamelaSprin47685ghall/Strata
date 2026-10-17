// src/core/tp5_ffn.cu - 06-phase4 §4.2.1-4.2.2: the 5-rank partial FFN tier's implementation.
//
// See the header for the contract.  The compute per rank is ONE `native_expert_grouped`
// call on the rank's 1/5 pack slice (the engine's own grouped kernel - the same
// arithmetic the peer tier uses, one call for the window's whole entry list):
//
//   q8      : quantize_q8_1_rows(x_dev, n_tok, H, q8, s)      (the window's rows)
//   partial : native_expert_grouped(L_r, ptr, start, cnt, dst, tok, groups, entries,
//                                    q8, scratch, y_dev, s)
//
// `L_r` is the layer's layout with n_ff = 128 (the rank's slice) and the rank blob's
// own offsets (gu_row = the slice's row bytes; the compact layout is gate|up|down so
// up_off = rank_gu and down_off = 2*rank_gu). The grouped rows are unweighted;
// the publish kernel combines each token's ten entries with its router weights.
//
// Only one weighted row per token crosses each link. The RELAY sums those five
// rows in rank order; the verify graph adds its shared expert without reweighting.
#include "strata/core/tp5_ffn.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "strata/kernels/verify_kernels.hpp" // gpu_stamp: 06 §5.3's STRATA_TP5_TRACE
#include "strata/plan/tp5_plan.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <immintrin.h> // _mm_sfence: the mapped-publish fence
#include <stdexcept>
#include <thread>

namespace strata::core {

namespace {

namespace tp5 = strata::kernels::tp5;
namespace plan = strata::plan::tp5;

constexpr int64_t H = strata::kernels::cpu::H;      // 2560
constexpr int kMaxTok = strata::kernels::cpu::MAXT; // 8
constexpr int kTopK = 10;
constexpr int64_t kEntries = kMaxTok * kTopK; // 80: the window's entry cap

/// The packed plan staging's int offsets (graph mode): start | dst | tok | cnt,
/// non-overlapping (see capture_rank).
constexpr int kPlanDstOff() { return int(kEntries) + 1; }
constexpr int kPlanTokOff() { return 2 * int(kEntries) + 1; }
constexpr int kPlanCntOff() { return 3 * int(kEntries) + 1; }
constexpr size_t kPlanOffset = size_t(kEntries) * sizeof(unsigned long long);
constexpr size_t kStepOffset = (kPlanOffset + (kPlanCntOff() + 4) * sizeof(int32_t) + 15) & ~size_t(15);
constexpr size_t kMetadataBytes = kStepOffset + sizeof(tp5::StepState);

struct On {
    int prev = 0;
    explicit On(int dev) {
        cudaGetDevice(&prev);
        if (prev != dev) cudaSetDevice(dev);
    }
    ~On() {
        int cur = 0;
        cudaGetDevice(&cur);
        if (cur != prev) cudaSetDevice(prev);
    }
};

/// The host clock in ns (06 §5.3: the fold anchor for prod_end_host_ns).
long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

bool ck(cudaError_t e, const char *what, std::string &err) {
    if (e == cudaSuccess) return true;
    err = std::string("tp5 ffn: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

// The same q8_1 block expression as quantize_q8_1_rows, with the small plan,
// weights and completion-counter reset folded into the input read. The mapped
// inputs stay immutable until every rank publishes ready, as with captured H2Ds.
__global__ void stage_rank_kernel(const float *__restrict__ input, const uint4 *__restrict__ metadata,
                                  uint4 *__restrict__ metadata_out, float *__restrict__ weights,
                                  uint8_t *__restrict__ q8, unsigned *__restrict__ counter, int n_tok) {
    static_assert(kMetadataBytes % sizeof(uint4) == 0);
    if (blockIdx.x == 0) {
        for (unsigned i = threadIdx.x; i < kMetadataBytes / sizeof(uint4); i += blockDim.x)
            metadata_out[i] = metadata[i];
        for (int i = int(threadIdx.x); i < n_tok * kTopK; i += int(blockDim.x))
            weights[i] = input[int64_t(n_tok) * H + i];
        if (threadIdx.x == 0) *counter = 0;
    }
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= int64_t(n_tok) * H) return;
    const float xi = input[i];
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = strata::kernels::q8_1_finite(amax / 127.0f);
    uint8_t *block = q8 + (i / 32) * 36;
    reinterpret_cast<int8_t *>(block + sizeof(half2))[i % 32] = strata::kernels::q8_1_quant(xi, d, amax);
    if (i % 32 == 0) *reinterpret_cast<half2 *>(block) = strata::kernels::q8_1_ds(d, sum);
}

/// The RELAY producer: weight and reduce the rank's expert rows, publishing one
/// row per token. The last arriving block fences system-scope and publishes ready.
/// The inbox and its flag are laid out for kMaxTok rows, not kEntries expert rows.
/// In graph mode (M3) `step` is non-null: epoch and bank come from the device StepState
/// the graph's H2D node refreshes per launch - a captured constant would replay the
/// first window's epoch forever (05 §3's rule).  Eager mode passes nullptr and uses the
/// call's own epoch/bank.
__global__ void publish_partial_kernel(const float *__restrict__ y_dev, const float *__restrict__ weights,
                                       uint8_t *__restrict__ inbox_base, uint64_t stride, int64_t epoch, uint32_t bank,
                                       int rank, const tp5::StepState *__restrict__ step,
                                       unsigned int *__restrict__ counter) {
    if (step != nullptr) {
        epoch = (int64_t)step->epoch;
        bank = step->bank;
    }
    uint8_t *ib = inbox_base + (uint64_t(bank) * tp5::kRanks + rank) * stride;
    float *payload = reinterpret_cast<float *>(ib);
    volatile uint64_t *ready = reinterpret_cast<volatile uint64_t *>(ib + tp5::Inbox::flag_off(kMaxTok));
    const int64_t token = blockIdx.y;
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col < H) {
        const float *parts = y_dev + token * kTopK * H + col;
        const float *w = weights + token * kTopK;
        float sum = parts[0] * w[0];
#pragma unroll
        for (int j = 1; j < kTopK; ++j) sum = fmaf(parts[int64_t(j) * H], w[j], sum);
        payload[token * H + col] = sum;
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

__global__ void combine_partial_kernel(const float *__restrict__ routed, const float *__restrict__ shared,
                                       float *__restrict__ output, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = routed[i] + shared[i];
}

/// The RELAY's five-partial sum, rank order, AVX2 (target-attributed: hipcc compiles
/// the .cu without -mavx2; the attribute enables it for this function alone).  The
/// lane order is fixed, so the values are the scalar loop's bit for bit.
__attribute__((target("avx2"))) void sum5_avx2(const float *a, const float *b, const float *c, const float *d,
                                               const float *e, float *y, size_t elems) {
    const size_t vec = elems / 8 * 8;
    for (size_t i = 0; i < vec; i += 8) {
        __m256 acc = _mm256_loadu_ps(a + i);
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(b + i));
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(c + i));
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(d + i));
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(e + i));
        _mm256_storeu_ps(y + i, acc);
    }
    for (size_t i = vec; i < elems; ++i) y[i] = a[i] + b[i] + c[i] + d[i] + e[i];
}

} // namespace

int Tp5FfnTier::gpu_of_rank(int r) const {
    // GPU0 carries the verify window's captured graph; the tier's five partials run on
    // GPUs 1-4 (rank 4 shares GPU1 with rank 0 - both idle from the graph's viewpoint).
    // 2026-10-16 trace: that assumption is wrong - r0 and r4 stretch each other
    // ~1.8x (0.471/0.470 ms/run vs 0.257-0.270 for r1-r3, 178-window aggregate), and
    // during the relay GPU0 runs only the layer's 1-thread wait_flag node (the graph
    // is blocked on the flag WE raise after the sum). STRATA_TP5_FFN_R4_GPU moves
    // rank 4 (default 1 = the old placement); 0 puts it on GPU0's idle CUs.
    return (r < 4) ? (r + 1) : r4_gpu_;
}

Tp5FfnTier::~Tp5FfnTier() { close(); }

void Tp5FfnTier::close() {
    // The inbox is the first allocation; open_ is set only after all ranks load.
    // Disabled tiers still make no runtime calls, while failed opens release owners.
    if (!open_ && inbox_ == nullptr) return;
    if (g_abort_ != nullptr) __atomic_store_n(g_abort_, 1u, __ATOMIC_RELEASE);
    for (int r = 0; r < tp5::kRanks; ++r) {
        if (pstream_[r] == nullptr) continue; // The stream precedes every rank-owned allocation.
        On on(gpu_of_rank(r));
        // Captured H2D nodes may still read the pinned staging on an error path.
        cudaStreamSynchronize(pstream_[r]);
        for (int b = 0; b < int(graph_buckets_); ++b) {
            if (gexecs_[r][size_t(b)]) {
                cudaGraphExecDestroy(gexecs_[r][size_t(b)]);
                gexecs_[r][size_t(b)] = nullptr;
            }
            if (graphs_[r][size_t(b)]) {
                cudaGraphDestroy(graphs_[r][size_t(b)]);
                graphs_[r][size_t(b)] = nullptr;
            }
        }
        if (meta_stage_[r]) {
            cudaFreeHost(meta_stage_[r]);
            meta_stage_[r] = nullptr;
        }
        cudaStreamDestroy(pstream_[r]);
        pstream_[r] = nullptr;
        if (w_dev_[r]) {
            cudaFree(w_dev_[r]);
            w_dev_[r] = nullptr;
        }
        if (x_dev_[r]) {
            cudaFree(x_dev_[r]);
            x_dev_[r] = nullptr;
        }
        if (y_dev_[r]) {
            cudaFree(y_dev_[r]);
            y_dev_[r] = nullptr;
        }
        if (q8_[r]) {
            cudaFree(q8_[r]);
            q8_[r] = nullptr;
        }
        if (scratch_[r]) {
            cudaFree(scratch_[r]);
            scratch_[r] = nullptr;
        }
        if (meta_dev_[r]) {
            cudaFree(meta_dev_[r]);
            meta_dev_[r] = nullptr;
        }
        if (pcounters_[r]) {
            cudaFree(pcounters_[r]);
            pcounters_[r] = nullptr;
        }
        if (trace_dev_[r]) {
            cudaFree(trace_dev_[r]);
            trace_dev_[r] = nullptr;
        }
        ptr_dev_[r] = ptr_stage_[r] = nullptr;
        start_dev_[r] = dst_dev_[r] = tok_dev_[r] = cnt_dev_[r] = nullptr;
        plan_stage_[r] = nullptr;
        step_stage_[r] = nullptr;
        inbox_dev_r_[r] = nullptr;
        meta_mapped_[r] = nullptr;
        x_mapped_[r] = nullptr;
    }
    if (x_stage_) {
        cudaFreeHost(x_stage_);
        x_stage_ = nullptr;
    }
    if (inbox_) cudaFreeHost(inbox_);
    if (credits_) cudaFreeHost(credits_);
    if (g_abort_) cudaFreeHost(g_abort_);
    inbox_ = nullptr;
    credits_ = nullptr;
    g_abort_ = nullptr;
    open_ = false;
}

int64_t Tp5FfnTier::vram_bytes() const {
    if (!open_) return 0;
    int64_t total = 0;
    for (int64_t l = 0; l < n_layers_; ++l) total += slice_bytes(l) * n_expert_;
    total *= tp5::kRanks;
    const int64_t per_rank = (int64_t(kMaxTok) * (H + kTopK) + kEntries * H) * 4 + // x, weights, y
                             int64_t(kMaxTok) * (H / 32) * 36 +                    // q8_1
                             int64_t(strata::kernels::native_expert_scratch_bytes(kEntries, 128)) +
                             int64_t(kMetadataBytes) + 4 * int64_t(sizeof(unsigned)) + (trace_on_ ? 24 : 0);
    return total + per_rank * tp5::kRanks;
}

bool Tp5FfnTier::open(int64_t n_layers, const uint8_t *blobs, int64_t n_expert, std::string &err) {
    close();
    epoch_ = 0;  // fresh inboxes and credits start a new relay lifetime
    r4_gpu_ = 1; // the default placement; STRATA_TP5_FFN_R4_GPU overrides (the trace experiment)
    if (const char *v = std::getenv("STRATA_TP5_FFN_R4_GPU")) {
        const int g = std::atoi(v);
        if (g >= 0 && g <= 4) r4_gpu_ = g;
    }
    if (blobs == nullptr || n_layers <= 0 || n_expert <= 0) {
        err = "tp5 ffn: null blobs or empty layer/expert set";
        return false;
    }
    const auto &lay = strata::kernels::cpu::expert_layout();
    if (!lay.native) {
        err = "tp5 ffn: needs a native pack";
        return false;
    }
    if (n_layers > (int64_t)lay.fmt.size()) {
        err = "tp5 ffn: more layers than the layout";
        return false;
    }
    for (int64_t l = 0; l < n_layers; ++l) {
        const auto &f = lay.fmt[size_t(l)];
        if (f.gu_type != 42 || f.d_type != 42) { // Q2_0/Q2_0 only: the slice arithmetic is Q2_0's
            err = "tp5 ffn: layer " + std::to_string(l) + " is not Q2_0/Q2_0 (the pack reorder is Q2_0's)";
            return false;
        }
        if (f.n_ff != 640) {
            err = "tp5 ffn: layer " + std::to_string(l) + " n_ff != 640 (the 5x128 split)";
            return false;
        }
    }
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    graph_ = graph_mode_default();
    const char *stage = std::getenv("STRATA_TP5_STAGE_KERNEL");
    stage_kernel_ = graph_ && stage && stage[0] == '1';

    // One weighted row per token, per (bank, rank); credits per bank.
    const uint64_t stride = tp5::Inbox::bytes(kMaxTok);
    const size_t total = size_t(stride) * tp5::kBanks * tp5::kRanks;
    unsigned mapped_flags = cudaHostAllocMapped | cudaHostAllocPortable;
#if defined(__HIPCC__)
    mapped_flags |= cudaHostAllocCoherent;
#endif
    if (!ck(cudaHostAlloc((void **)&inbox_, total, mapped_flags), "inbox", err)) {
        close();
        return false;
    }
    std::memset(inbox_, 0, total);
    if (!ck(cudaHostAlloc((void **)&credits_, sizeof(tp5::Credit) * tp5::kBanks, mapped_flags), "credits", err)) {
        close();
        return false;
    }
    std::memset(credits_, 0, sizeof(tp5::Credit) * tp5::kBanks);
    if (!ck(cudaHostAlloc((void **)&g_abort_, sizeof(unsigned), mapped_flags), "g_abort", err)) {
        close();
        return false;
    }
    *g_abort_ = 0;

    // Per-layer slice sizes in the layer-major weight allocation.
    slice_bytes_.assign(size_t(n_layers), 0);
    layer_offs_.assign(size_t(n_layers), 0);
    int64_t layer_acc = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        const auto &f = lay.fmt[size_t(l)];
        layer_offs_[size_t(l)] = layer_acc;
        slice_bytes_[size_t(l)] = int64_t(size_t(128) * size_t(f.gu_row) * 2 + size_t(2560) * (size_t(f.d_row) / 5));
        layer_acc += slice_bytes_[size_t(l)] * n_expert;
    }

    // per-rank device state
    const size_t q8_bytes = size_t(kMaxTok) * size_t(H / 32) * 36;
    const size_t scratch = strata::kernels::native_expert_scratch_bytes(kEntries, 128);
    int64_t w_total = 0;
    for (int64_t l = 0; l < n_layers; ++l) w_total += slice_bytes(l) * n_expert_;
    for (int r = 0; r < tp5::kRanks; ++r) {
        On on(gpu_of_rank(r));
#if defined(STRATA_USE_HIP)
        if (gpu_of_rank(r) == 0) {
            // The verifier spins on our result. Ordinary HIP streams can share
            // its hardware queue after GPU_MAX_HW_QUEUES is exhausted, creating
            // a FIFO dependency cycle even though almost every CU is idle.
            // An all-CU masked stream owns an independent queue, not a CU slice.
            hipDeviceProp_t prop{};
            if (!ck(hipGetDeviceProperties(&prop, 0), "primary device properties", err)) {
                close();
                return false;
            }
            uint32_t mask[32];
            const uint32_t words = uint32_t((prop.multiProcessorCount + 31) / 32);
            if (words == 0 || words > 32) {
                err = "tp5 ffn: primary device CU mask is out of range";
                close();
                return false;
            }
            std::fill_n(mask, words, UINT32_MAX);
            if ((prop.multiProcessorCount & 31) != 0)
                mask[words - 1] = (uint32_t(1) << (prop.multiProcessorCount & 31)) - 1;
            if (!ck(hipExtStreamCreateWithCUMask(&pstream_[r], words, mask), "primary rank stream", err)) {
                close();
                return false;
            }
            // This API is legacy-blocking: relay work stays on pstream_, and
            // NULL-stream transfers must wait until its producer has drained.
        } else
#endif
            if (!ck(cudaStreamCreateWithFlags(&pstream_[r], cudaStreamNonBlocking), "pstream", err)) {
            close();
            return false;
        }
        if (!ck(cudaMalloc((void **)&w_dev_[r], size_t(w_total)), "weights", err)) {
            close();
            return false;
        }
        { // the pack reorder, layer by layer: gate|up slices + the stride-extracted down
            std::vector<uint8_t> rb(size_t(w_total), 0);
            uint8_t *dst = rb.data();
            const uint8_t *src = blobs;
            for (int64_t l = 0; l < n_layers; ++l) {
                const auto &f = lay.fmt[size_t(l)];
                const size_t rank_gu = size_t(128) * size_t(f.gu_row);
                const size_t blk = size_t(f.d_row) / 5;
                for (int64_t e = 0; e < n_expert_; ++e) {
                    const uint8_t *ex = src + size_t(e) * size_t(f.bytes);
                    std::memcpy(dst, ex + size_t(128 * r) * size_t(f.gu_row), rank_gu);
                    dst += rank_gu;
                    std::memcpy(dst, ex + f.up_off + size_t(128 * r) * size_t(f.gu_row), rank_gu);
                    dst += rank_gu;
                    const uint8_t *down = ex + f.down_off;
                    for (int j = 0; j < 2560; ++j)
                        std::memcpy(dst + size_t(j) * blk, down + size_t(j) * size_t(f.d_row) + size_t(2 * r) * 18,
                                    blk);
                    dst += size_t(2560) * blk;
                }
                src += size_t(f.bytes) * size_t(n_expert_);
            }
            if (!ck(cudaMemcpy(w_dev_[r], rb.data(), rb.size(), cudaMemcpyHostToDevice), "weights copy", err)) {
                close();
                return false;
            }
        }
        if (!ck(cudaMalloc((void **)&x_dev_[r], size_t(kMaxTok) * (H + kTopK) * 4), "x", err) ||
            !ck(cudaMalloc((void **)&y_dev_[r], size_t(kEntries) * H * 4), "y", err) ||
            !ck(cudaMalloc((void **)&q8_[r], q8_bytes), "q8", err) ||
            !ck(cudaMalloc(&scratch_[r], scratch), "scratch", err) ||
            !ck(cudaMalloc(&meta_dev_[r], kMetadataBytes), "metadata", err) ||
            !ck(cudaHostAlloc(&meta_stage_[r], kMetadataBytes, stage_kernel_ ? mapped_flags : cudaHostAllocDefault), "metadata stage", err) ||
            !ck(cudaMalloc(&pcounters_[r], 4 * 4), "pcounters", err)) {
            close();
            return false;
        }
        if (stage_kernel_) {
            void *mapped = nullptr;
            if (!ck(cudaHostGetDevicePointer(&mapped, meta_stage_[r], 0), "metadata mapped", err)) {
                close();
                return false;
            }
            meta_mapped_[r] = mapped;
        }
        std::memset(meta_stage_[r], 0, kMetadataBytes);
        auto *md = static_cast<uint8_t *>(meta_dev_[r]);
        auto *mh = static_cast<uint8_t *>(meta_stage_[r]);
        ptr_dev_[r] = reinterpret_cast<unsigned long long *>(md);
        start_dev_[r] = reinterpret_cast<int32_t *>(md + kPlanOffset);
        dst_dev_[r] = start_dev_[r] + kPlanDstOff();
        tok_dev_[r] = start_dev_[r] + kPlanTokOff();
        cnt_dev_[r] = start_dev_[r] + kPlanCntOff();
        ptr_stage_[r] = reinterpret_cast<unsigned long long *>(mh);
        plan_stage_[r] = reinterpret_cast<int32_t *>(mh + kPlanOffset);
        step_stage_[r] = reinterpret_cast<tp5::StepState *>(mh + kStepOffset);
        if (!ck(cudaMemset(pcounters_[r], 0, 16), "pc zero", err)) {
            close();
            return false;
        }
        // the inbox device alias (queried on THIS device - see the probe's rule)
        void *dp = nullptr;
        if (!ck(cudaHostGetDevicePointer(&dp, inbox_, 0), "inbox dev", err)) {
            close();
            return false;
        }
        inbox_dev_r_[r] = static_cast<uint8_t *>(dp);
    }
    // host staging for the grouped plan
    h_start_.assign(size_t(kEntries) + 1, 0);
    h_dst_.assign(size_t(kEntries), 0);
    h_tok_.assign(size_t(kEntries), 0);
    h_cnt_.assign(4, 0);

    // 06 §5.3 (STRATA_TP5_TRACE=1): per-rank stamp buffers.  Off by default: the
    // eager/graph paths issue no stamp kernels and nothing is read back.
    trace_on_ = [] {
        const char *v = std::getenv("STRATA_TP5_TRACE");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (trace_on_) {
        for (int r = 0; r < tp5::kRanks; ++r) {
            On on(gpu_of_rank(r));
            if (!ck(cudaMalloc((void **)&trace_dev_[r], 24), "trace stamps", err)) {
                close();
                return false;
            }
            cudaMemset(trace_dev_[r], 0, 24);
        }
    }

    // Per-token-count graphs read fresh pinned inputs and metadata at each launch.
    { // 06 §6.3 launch proof: the tally is host-side only; the env gate keeps the
        // default path free of even the boolean check's register pressure
        const char *lc = std::getenv("STRATA_TP5_LAUNCH_COUNT");
        launch_count_on_ = lc && (lc[0] == '1' || lc[0] == 'y' || lc[0] == 'Y');
        for (int r = 0; r < tp5::kRanks; ++r) graph_launches_[r] = 0;
        eager_runs_ = 0;
    }
    if (graph_) {
        if (!ck(cudaHostAlloc((void **)&x_stage_, size_t(kMaxTok) * (H + kTopK) * 4, stage_kernel_ ? mapped_flags : cudaHostAllocPortable), "x stage",
                err)) {
            close();
            return false;
        }
        std::memset(x_stage_, 0, size_t(kMaxTok) * (H + kTopK) * 4);
        for (int r = 0; r < tp5::kRanks; ++r) {
            if (stage_kernel_) {
                On on(gpu_of_rank(r));
                void *mapped = nullptr;
                if (!ck(cudaHostGetDevicePointer(&mapped, x_stage_, 0), "x mapped", err)) {
                    close();
                    return false;
                }
                x_mapped_[r] = static_cast<const float *>(mapped);
            }
            std::memset(ptr_stage_[r], 0, 8 * size_t(kEntries));
            std::memset(plan_stage_[r], 0, 4 * (size_t(kPlanCntOff()) + 4));
            *step_stage_[r] = tp5::StepState{0, 0, 0};
        }
        for (int r = 0; r < tp5::kRanks; ++r)
            for (int b = 0; b < int(graph_buckets_); ++b)
                if (!capture_rank(r, b, err)) {
                    close();
                    return false;
                }
    }

    open_ = true;
    return true;
}

bool Tp5FfnTier::run(int64_t layer, const float *x, const int32_t *ids, const float *weights, int64_t n_tok, int64_t k,
                     float *y, std::string &err) {
    if (!open_) {
        err = "tp5 ffn: tier not open";
        return false;
    }
    if (!is_tp5_layer(layer)) {
        err = "tp5 ffn: layer not on the tier";
        return false;
    }
    if (n_tok <= 0 || n_tok > kMaxTok) {
        err = "tp5 ffn: window out of range";
        return false;
    }
    if (k != kTopK) {
        err = "tp5 ffn: expects top-10 routing";
        return false;
    }
    t_ = Timings{};

    ++epoch_;
    const int64_t epoch = int64_t(epoch_);
    const uint32_t bank = uint32_t((epoch - 1) & 1);
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t n = n_tok * k; // entries

    // bank reuse guard: the previous use of this bank must have its credit
    if (epoch > 2) {
        const int64_t prev = epoch - 2;
        if (int64_t(credits_[bank].consumed_epoch) < prev) {
            err = "tp5 ffn: bank reuse before credit (protocol)";
            return false;
        }
    }

    if (graph_) return run_graph(layer, x, ids, weights, n_tok, k, y, err);

    if (launch_count_on_) ++eager_runs_; // 06 §6.3: the eager path is not a launch

    // ---- the grouped plan (host): distinct experts in routing order, entries as rows
    int groups = 0, entries = 0;
    plan_of(ids, n, k, groups, entries);

    // the layer's layout, as the rank's slice sees it: n_ff = 128, the compact blob's
    // offsets (gate at 0, up at rank_gu, down at 2*rank_gu)
    const auto &f = strata::kernels::cpu::expert_layout().fmt[size_t(layer)];
    const size_t rank_gu = size_t(128) * size_t(f.gu_row);
    const dim3 publish_grid(unsigned((H + 255) / 256), unsigned(n_tok));
    const uint64_t stride = tp5::Inbox::bytes(kMaxTok);

    // ---- the five ranks' UNWEIGHTED partials (one grouped call each)
    for (int r = 0; r < tp5::kRanks; ++r) {
        On on(gpu_of_rank(r));
        const cudaStream_t s = pstream_[r];
        if (!ck(cudaMemcpyAsync(x_dev_[r], x, size_t(n_tok) * H * 4, cudaMemcpyHostToDevice, s), "x copy", err) ||
            !ck(cudaMemcpyAsync(x_dev_[r] + n_tok * H, weights, size_t(n) * 4, cudaMemcpyHostToDevice, s),
                "weights copy", err))
            return false;
        if (!ck(cudaMemsetAsync(pcounters_[r] + (epoch & 3), 0, 4, s), "pc zero", err)) return false;
        // the plan: this rank's slice addresses.  The layers are contiguous in w_dev_
        // (layer-major), so layer l's expert e sits at layer_off(l) + e*slice_bytes(l) -
        // WITHOUT the layer offset every layer > 0 read layer 0's experts (the 48-layer
        // run's n=1 empty-text signature)
        const uint8_t *layer_base = w_dev_[r] + size_t(layer_off(layer));
        for (int g = 0; g < groups; ++g) {
            // which expert the group is: recompute from ids (the group's first entry)
            const int32_t e = ids[h_dst_[size_t(h_start_[size_t(g)])]];
            ptr_stage_[r][size_t(g)] = (unsigned long long)(layer_base + size_t(e) * size_t(slice_bytes(layer)));
        }
        std::memcpy(plan_stage_[r], h_start_.data(), 4 * size_t(groups + 1));
        std::memcpy(plan_stage_[r] + kPlanDstOff(), h_dst_.data(), 4 * size_t(entries));
        std::memcpy(plan_stage_[r] + kPlanTokOff(), h_tok_.data(), 4 * size_t(entries));
        std::memcpy(plan_stage_[r] + kPlanCntOff(), h_cnt_.data(), 16);
        if (!ck(cudaMemcpyAsync(meta_dev_[r], meta_stage_[r], kMetadataBytes, cudaMemcpyHostToDevice, s),
                "metadata copy", err))
            return false;
        if (trace_on_) // 06 §5.3: the rank's own chain, start (before the quantize)
            strata::kernels::gpu_stamp(trace_dev_[r], 0, s);
        strata::kernels::quantize_q8_1_rows(x_dev_[r], n_tok, H, q8_[r], s);
        strata::kernels::NativeExpertLayout L;
        L.gu_type = f.gu_type;
        L.d_type = f.d_type;
        L.n_embd = H;
        L.n_ff = 128;          // the rank's slice of the 640 channels
        L.gu_row = f.gu_row;   // the slice's rows keep the full row bytes (n_embd wide)
        L.d_row = f.d_row / 5; // 36: the rank's 2 Q2_0 blocks per down row
        L.up_off = rank_gu;    // the compact blob: gate | up | down
        L.down_off = 2 * rank_gu;
        L.bytes = size_t(slice_bytes(layer));
        strata::kernels::native_expert_grouped(L, ptr_dev_[r], start_dev_[r], cnt_dev_[r], dst_dev_[r], tok_dev_[r],
                                               groups, entries, q8_[r], scratch_[r], y_dev_[r], s);
        if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
            err = std::string("tp5 ffn: grouped launch (rank ") + std::to_string(r) + "): " + cudaGetErrorString(le);
            return false;
        }
        // publish the partial rows through the RELAY inbox
        publish_partial_kernel<<<publish_grid, 256, 0, s>>>(y_dev_[r], x_dev_[r] + n_tok * H, inbox_dev_r_[r], stride,
                                                            epoch, bank, r, nullptr, pcounters_[r] + (epoch & 3));
        if (trace_on_) { // 06 §5.3: ... end (after the publish launch); slot 2 is the
            // per-stage calibration stamp: the host reads it right after the stream
            // sync in collect_trace, so (host_now - slot2) folds the rank's GPU clock
            // onto the host clock for THIS stage (E_RELAY's consumer side)
            strata::kernels::gpu_stamp(trace_dev_[r], 1, s);
            strata::kernels::gpu_stamp(trace_dev_[r], 2, s);
        }
        if (const cudaError_t le = cudaGetLastError(); le != cudaSuccess) {
            err = std::string("tp5 ffn: publish launch (rank ") + std::to_string(r) + "): " + cudaGetErrorString(le);
            return false;
        }
    }

    // ---- the CPU relay: shared with the graph path
    return relay_tail(epoch, bank, n_tok, y, t0, err);
}

void Tp5FfnTier::combine(const float *routed, const float *shared, float *output, int64_t n_tok, void *stream) const {
    const int64_t n = n_tok * H;
    combine_partial_kernel<<<unsigned((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(routed, shared,
                                                                                                     output, n);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

bool Tp5FfnTier::relay_tail(int64_t epoch, uint32_t bank, int64_t n, float *y,
                            const std::chrono::steady_clock::time_point &t0, std::string &err) {
    const uint64_t stride = tp5::Inbox::bytes(kMaxTok);
    // wait ready (bounded, WIDE - the ranks share their GPUs with nothing else, but the
    // first window's cold caches can take seconds)
    const auto dl = std::chrono::steady_clock::now() + std::chrono::milliseconds(30000);
    volatile uint64_t *ready[tp5::kRanks];
    for (int r = 0; r < tp5::kRanks; ++r)
        ready[r] = reinterpret_cast<volatile uint64_t *>(inbox_ + (size_t(bank) * tp5::kRanks + size_t(r)) * stride +
                                                         tp5::Inbox::flag_off(kMaxTok));
    if (trace_on_) // 06 §5.3 E_RELAY: per-rank first-seen host stamps (no fold)
        for (int r = 0; r < tp5::kRanks; ++r) tr_.ready_seen_ns[r] = 0;
    int wait_rank = 0;
    while (true) {
        bool all = true;
        for (int r = 0; r < tp5::kRanks; ++r)
            if (*ready[r] >= uint64_t(epoch)) {
                if (trace_on_ && tr_.ready_seen_ns[r] == 0)
                    tr_.ready_seen_ns[r] = double(now_ns()); // the rank's ready_first
            } else {
                if (all) wait_rank = r;
                all = false;
            }
        if (all) break;
        if (std::chrono::steady_clock::now() > dl) {
            // 06 §4.2.3: WHICH rank never landed, and what its stream is doing -
            // the intermittent hang's only clue so far (the QSA tier beside it)
            char which[64] = {};
            for (int r = 0; r < tp5::kRanks; ++r)
                if (*ready[r] < uint64_t(epoch))
                    std::snprintf(which + std::strlen(which), sizeof(which) - std::strlen(which), "%s%d",
                                  r > 0 ? "," : "", r);
            std::fprintf(stderr, "tp5 ffn: ready timeout epoch=%lld ranks=[%s] ready=[", (long long)epoch, which);
            for (int r = 0; r < tp5::kRanks; ++r)
                std::fprintf(stderr, "%s%llu", r > 0 ? "," : "", (unsigned long long)*ready[r]);
            std::fprintf(stderr, "]\n");
            err = "tp5 ffn: ready timeout (a rank's partial never landed)";
            __atomic_store_n(g_abort_, 1u, __ATOMIC_RELEASE);
            return false;
        }
        // busy-wait with a pause: yield() parked the thread long enough to dominate
        // the per-layer ready latency (48 layers x ~0.5 ms = the engine's 27 ms/window
        // measured by the cost split); the deadline check every 4096 spins keeps the
        // timeout honest
        for (int spin = 0; spin < 4096; ++spin) {
            if (*ready[wait_rank] >= uint64_t(epoch)) break;
            _mm_pause();
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (trace_on_) { // 06 §5.3 E_RELAY: the host-clock endpoints, host ns
        t_.t0_host_ns = std::chrono::duration<double, std::nano>(t0.time_since_epoch()).count();
        t_.ready_exit_host_ns = std::chrono::duration<double, std::nano>(t1.time_since_epoch()).count();
    }
    collect_trace();
    {
        // the sum in rank order (FP32, fixed order), straight into the caller's rows.
        // AVX2 (the target-attributed helper): the scalar loop was 8.5 ms/window in the
        // engine - 48 layers x 80 rows x 2560 x 5 ranks - the RELAY's own cost split
        // measured it (06 §5.3).  The lane order is fixed, so the values are the
        // scalar loop's.
        const int64_t nrow = n;
        const float *p[tp5::kRanks];
        for (int r = 0; r < tp5::kRanks; ++r)
            p[r] = reinterpret_cast<const float *>(inbox_ + (size_t(bank) * tp5::kRanks + size_t(r)) * stride);
        sum5_avx2(p[0], p[1], p[2], p[3], p[4], y, size_t(nrow) * H);
    }
    const auto t2 = std::chrono::steady_clock::now();
    _mm_sfence();
    __atomic_store_n(&credits_[bank].consumed_epoch, uint64_t(epoch), __ATOMIC_RELEASE);
    t_.ready_all_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    t_.sum_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    t_.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

void Tp5FfnTier::collect_trace() {
    // 06 §5.3 (STRATA_TP5_TRACE=1): the ready wait has landed, so every rank's end
    // stamp is written (the publish kernel precedes it on the same stream).  A SYNC
    // read per rank - trace runs are for decomposition, never for speed.
    if (!trace_on_) return;
    for (int r = 0; r < tp5::kRanks; ++r) {
        On on(gpu_of_rank(r));
        // the end stamp is QUEUED on the rank's stream after the publish; a blocking
        // cudaMemcpy on the default stream does NOT wait for it.  Sync the stream the
        // stamp lives on first (the GDN tier's zero-span race - same shape here).
        cudaStreamSynchronize(pstream_[r]);
        // 06 §5.3: the fold anchor - the FIRST host moment that is at or after
        // slot 2's GPU completion.  Taking it before the memcpy (not after the
        // readback) keeps the sync+copy latency OUT of the fold; what remains is
        // only the sync's own wake-up latency, the same for every rank.
        const long long anchor_ns = now_ns();
        unsigned long long h[3] = {0, 0, 0};
        if (cudaMemcpy(h, trace_dev_[r], 24, cudaMemcpyDeviceToHost) != cudaSuccess) {
            (void)cudaGetLastError();
            tr_.prod_span_ns[r] = 0;
            continue;
        }
        if (h[1] >= h[0] && h[0] != 0) tr_.prod_span_ns[r] = double(h[1] - h[0]);
        else tr_.prod_span_ns[r] = 0; // a missing/out-of-order stamp contributes nothing
        // E_RELAY's consumer side: slot 2 stamped right after slot 1 on the same
        // stream, and this point is after the stream sync, so the host clock at
        // `now` corresponds to the GPU clock at h[2].  The fold is per stage (the
        // GPU counters drift; a startup-only calibration would not see it):
        //   prod_end_host_ns[r] = now + (h[1] - h[2])   (both GPU-clock differences)
        // The sync's own latency (a few hundred us) shifts BOTH the fold anchor and
        // the host's ready stamp equally for every rank, so E_RELAY differences
        // between ranks stay exact.
        if (h[1] != 0 && h[2] >= h[1]) tr_.prod_end_host_ns[r] = double(anchor_ns) + double(h[1]) - double(h[2]);
        else tr_.prod_end_host_ns[r] = 0;
    }
}

bool Tp5FfnTier::graph_mode_default() {
    const char *v = std::getenv("STRATA_TP5_GRAPH");
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

/// Builds the grouped plan (the distinct-experts / rows-per-entry layout) into the
/// host staging `h_*` arrays - the eager path and run_graph share it.
void Tp5FfnTier::plan_of(const int32_t *ids, int64_t n, int64_t k, int &groups, int &entries) {
    groups = 0;
    entries = 0;
    for (int64_t i = 0; i < n; ++i) {
        bool seen = false;
        for (int64_t j = 0; j < i; ++j)
            if (ids[j] == ids[i]) {
                seen = true;
                break;
            }
        if (seen) continue;
        h_start_[size_t(groups)] = entries;
        for (int64_t j = i; j < n; ++j)
            if (ids[j] == ids[i]) {
                h_dst_[size_t(entries)] = int32_t(j);
                h_tok_[size_t(entries)] = int32_t(j / k);
                ++entries;
            }
        ++groups;
    }
    h_start_[size_t(groups)] = entries;
    h_cnt_[0] = groups;
    h_cnt_[1] = entries;
}

bool Tp5FfnTier::capture_rank(int r, int bucket, std::string &err) {
    // Bucket-sized quantize -> grouped -> weighted publish. The real plan counts
    // and StepState are copied together; only one output row per token is published.
    On on(gpu_of_rank(r));
    const cudaStream_t s = pstream_[r];
    const auto &f = strata::kernels::cpu::expert_layout().fmt[0];
    const size_t rank_gu = size_t(128) * size_t(f.gu_row);
    strata::kernels::NativeExpertLayout L;
    L.gu_type = f.gu_type;
    L.d_type = f.d_type;
    L.n_embd = H;
    L.n_ff = 128;
    L.gu_row = f.gu_row;
    L.d_row = f.d_row / 5;
    L.up_off = rank_gu;
    L.down_off = 2 * rank_gu;
    L.bytes = size_t(slice_bytes(0));
    // a valid plan for the capture run: one group, one entry, expert 0
    ptr_stage_[r][0] = (unsigned long long)w_dev_[r];
    h_start_[0] = 0;
    h_start_[1] = 1;
    h_dst_[0] = 0;
    h_tok_[0] = 0;
    h_cnt_[0] = 1;
    h_cnt_[1] = 1;
    // the bucket's caps: n_tok = bucket+1 rows -> at most n_tok*10 groups/entries.  The
    // graph's work is sized EXACTLY to the bucket (the CAP-sized capture was the engine's
    // -1.5%: 8x the x H2D and the swiglu/quantize grids at 1-row windows).  grid_groups=0
    // keeps the eager call's kernel selection (em0 = exp_mode(); a strided grid_groups
    // would switch to the CUDA-layout fallback kernel).
    const int64_t bucket_cap = int64_t(bucket + 1) * kTopK;
    const int bucket_rows = bucket + 1;
    const dim3 publish_grid(unsigned((H + 255) / 256), unsigned(bucket_rows));
    *step_stage_[r] = tp5::StepState{1, 0, 0};
    // the packed staging layout (NON-overlapping - the first cut put dst/tok/cnt at int
    // offsets +2/+3/+4, ON TOP of start's 81 ints; the fills overwrote each other and the
    // graph replayed garbage start/ent_tok values - a memory fault at n_tok=5):
    //   [0, kE] start | [kE+1, 2kE] dst | [2kE+1, 3kE] tok | [3kE+1, 3kE+4] cnt
    std::memcpy(plan_stage_[r] + 0, h_start_.data(), 4 * (size_t(kEntries) + 1));
    std::memset(plan_stage_[r] + kPlanDstOff(), 0, 4 * size_t(kEntries));
    std::memset(plan_stage_[r] + kPlanTokOff(), 0, 4 * size_t(kEntries));
    std::memcpy(plan_stage_[r] + kPlanCntOff(), h_cnt_.data(), 16);

    if (!ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin cap", err)) return false;
    // the counter slot this epoch uses (epoch 0 in the capture; every real epoch's slot
    // is zeroed by the graph's first node - the slot is unique per epoch, so the memset
    // races with nothing)
    if (stage_kernel_) {
        stage_rank_kernel<<<unsigned(bucket_rows * H / 256), 256, 0, s>>>(
            x_mapped_[r], static_cast<const uint4 *>(meta_mapped_[r]), static_cast<uint4 *>(meta_dev_[r]),
            x_dev_[r] + int64_t(bucket_rows) * H, static_cast<uint8_t *>(q8_[r]), pcounters_[r], bucket_rows);
    } else {
        ck(cudaMemsetAsync(pcounters_[r], 0, 4, s), "cap pc zero", err);
        ck(cudaMemcpyAsync(x_dev_[r], x_stage_, size_t(bucket_rows) * (H + kTopK) * 4, cudaMemcpyHostToDevice, s), "cap x",
           err);
        ck(cudaMemcpyAsync(meta_dev_[r], meta_stage_[r], kMetadataBytes, cudaMemcpyHostToDevice, s), "cap metadata", err);
        strata::kernels::quantize_q8_1_rows(x_dev_[r], bucket_rows, H, q8_[r], s);
    }
    // grid_groups=0 (default): gy = cap_groups = the bucket's max, the same kernel
    // selection as the eager call; the stride loops cover any real count below it
    strata::kernels::native_expert_grouped(L, ptr_dev_[r], start_dev_[r], cnt_dev_[r], dst_dev_[r], tok_dev_[r],
                                           bucket_cap, bucket_cap, q8_[r], scratch_[r], y_dev_[r], s);
    publish_partial_kernel<<<publish_grid, 256, 0, s>>>(
        y_dev_[r], x_dev_[r] + int64_t(bucket_rows) * H, inbox_dev_r_[r], tp5::Inbox::bytes(kMaxTok), 0, 0, r,
        reinterpret_cast<const tp5::StepState *>(static_cast<const uint8_t *>(meta_dev_[r]) + kStepOffset),
        pcounters_[r]);
    cudaError_t e = cudaStreamEndCapture(s, &graphs_[r][size_t(bucket)]);
    if (e != cudaSuccess) {
        err = std::string("tp5 ffn: end cap (rank ") + std::to_string(r) + ", bucket " + std::to_string(bucket) +
              "): " + cudaGetErrorString(e);
        return false;
    }
    e = cudaGraphInstantiate(&gexecs_[r][size_t(bucket)], graphs_[r][size_t(bucket)], nullptr, nullptr, 0);
    if (e != cudaSuccess) {
        err = std::string("tp5 ffn: instantiate (rank ") + std::to_string(r) + ", bucket " + std::to_string(bucket) +
              "): " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool Tp5FfnTier::run_graph(int64_t layer, const float *x, const int32_t *ids, const float *weights, int64_t n_tok,
                           int64_t k, float *y, std::string &err) {
    const int64_t n = n_tok * k;
    int groups = 0, entries = 0;
    plan_of(ids, n, k, groups, entries);
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t epoch = int64_t(epoch_); // run() already bumped it
    const uint32_t bank = uint32_t((epoch - 1) & 1);

    // All five H2Ds finish before relay_tail returns; the next run may then rewrite
    // this portable pinned input once, rather than copying it into five owners.
    std::memcpy(x_stage_, x, size_t(n_tok) * H * 4);
    std::memcpy(x_stage_ + n_tok * H, weights, size_t(n) * 4);

    // the layer's slice base (layer-major in w_dev_)
    const uint8_t *layer_base = nullptr;

    // fill every rank's staging (host memory - no device context needed), then launch
    // all five graphs in one pass: the fill loop's 5 device toggles per layer were
    // 4 driver calls each, ~0.2 ms of the 0.57 ms per-layer ready (48 layers x 5)
    const size_t sb = size_t(slice_bytes(layer));
    const size_t loff = size_t(layer_off(layer));
    for (int r = 0; r < tp5::kRanks; ++r) {
        layer_base = w_dev_[r] + loff;
        for (int g2 = 0; g2 < groups; ++g2) {
            const int32_t e = ids[h_dst_[size_t(h_start_[size_t(g2)])]];
            ptr_stage_[r][size_t(g2)] = (unsigned long long)(layer_base + size_t(e) * sb);
        }
        std::memcpy(plan_stage_[r] + 0, h_start_.data(), 4 * size_t(groups + 1));
        std::memcpy(plan_stage_[r] + kPlanDstOff(), h_dst_.data(), 4 * size_t(entries));
        std::memcpy(plan_stage_[r] + kPlanTokOff(), h_tok_.data(), 4 * size_t(entries));
        std::memcpy(plan_stage_[r] + kPlanCntOff(), h_cnt_.data(), 16);
        *step_stage_[r] = tp5::StepState{uint64_t(epoch), bank, 0};
        // the publish counter: the graph's captured memset node zeroes slot 0 (the slot
        // the graph's publish kernel uses) at every replay - no host-side zeroing needed
    }
    const int bucket = int(n_tok) - 1;
    // Start the primary-GPU rank first; it also shares bandwidth with the
    // verifier's shared expert. Keep one restore scope for the entire batch,
    // rather than returning to the caller's device between every rank.
    const int first = r4_gpu_ == 0 ? 4 : 0;
    {
    On on(gpu_of_rank(first));
    int current = gpu_of_rank(first);
    for (int i = 0; i < tp5::kRanks; ++i) {
        const int r = (first + i) % tp5::kRanks;
        const int device = gpu_of_rank(r);
        if (device != current) {
            if (!ck(cudaSetDevice(device), "rank device", err)) return false;
            current = device;
        }
        if (trace_on_) // 06 §5.3: the graph's own span - stamps OUTSIDE the graph
            strata::kernels::gpu_stamp(trace_dev_[r], 0, pstream_[r]);
        if (launch_count_on_) ++graph_launches_[r]; // 06 §6.3: per-rank tally
        if (!ck(cudaGraphLaunch(gexecs_[r][size_t(bucket)], pstream_[r]), "graph launch", err)) return false;
        if (trace_on_) {
            strata::kernels::gpu_stamp(trace_dev_[r], 1, pstream_[r]);
            strata::kernels::gpu_stamp(trace_dev_[r], 2, pstream_[r]); // the per-stage fold stamp
        }
    }
    }
    t_.stage_launch_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                             .count(); // 06 shift-6: the staging + 5-launch host cost

    // the same CPU relay tail as the eager path (wait ready, sum, credit) - shared below
    return relay_tail(epoch, bank, n_tok, y, t0, err);
}

} // namespace strata::core
