// tests/hip/tp5_ffn_test.cpp - 06-phase4 §4.2.1's quality gate for the 5-rank partial FFN.
//
// Two independent references: the unsplit GPU operator shares Q8_1 activation
// quantization, allowing a strict element-wise FP32 reduction-order bound. A
// dequantized CPU FP32 operator also bounds the activation-quantization error in
// relative L2, so tiny outputs cannot hide missing or twice-weighted contributions.
//
//   usage: tp5_ffn_test --pack DIR [--layer N] [--windows K] [--seed S] [--csv FILE]
//   needs the five GPUs; exit 0 = pass, 77 = fewer than five devices, 1 = mismatch
#include "strata/core/expert_source.hpp"
#include "strata/core/tp5_ffn.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/plan/tp5_plan.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <exception>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace tp5 = strata::kernels::tp5;
namespace plan = strata::plan::tp5;
using strata::core::Tp5FfnTier;

namespace {

constexpr int64_t H = strata::kernels::cpu::H; // 2560
constexpr int kTopK = 10;

// Independent full-width operator: no rank slicing, no relay, and one group per
// routing entry rather than the tier's deduplicated expert groups. It uses the
// same activation quantization, so only FP32 summation order is allowed to differ.
class UnsplitGpuOracle {
    static constexpr int kEntries = Tp5FfnTier::kWindowCap * kTopK;
    struct Plan {
        unsigned long long ptr[kEntries];
        int32_t start[kEntries + 1], dst[kEntries], tok[kEntries], groups;
    } plan_{};
    Plan *plan_d_ = nullptr;
    uint8_t *blobs_ = nullptr;
    float *x_ = nullptr, *weights_ = nullptr, *parts_ = nullptr, *shared_ = nullptr, *out_ = nullptr;
    void *xq_ = nullptr, *scratch_ = nullptr;
    strata::kernels::NativeExpertLayout layout_;
    cudaStream_t stream_ = nullptr;
    std::array<void *, 9> owned_{};
    size_t owned_count_ = 0, blob_bytes_ = 0;
    std::vector<float> result_;

    static bool ck(cudaError_t e, const char *op, std::string &err) {
        if (e == cudaSuccess) return true;
        err = std::string("unsplit oracle: ") + op + ": " + cudaGetErrorString(e);
        return false;
    }
    template <class T> bool allocate(T *&p, size_t bytes, std::string &err) {
        if (owned_count_ == owned_.size()) {
            err = "unsplit oracle: allocation table full";
            return false;
        }
        if (!ck(cudaMalloc((void **)&p, bytes), "allocate", err)) return false;
        owned_[owned_count_++] = p;
        return true;
    }

  public:
    UnsplitGpuOracle() = default;
    UnsplitGpuOracle(const UnsplitGpuOracle &) = delete;
    UnsplitGpuOracle &operator=(const UnsplitGpuOracle &) = delete;
    ~UnsplitGpuOracle() {
        if (stream_) cudaStreamSynchronize(stream_);
        for (size_t i = 0; i < owned_count_; ++i) cudaFree(owned_[i]);
        if (stream_) cudaStreamDestroy(stream_);
    }
    bool open(const std::vector<uint8_t> &blobs, const strata::kernels::cpu::NativeFmt &f, std::string &err) {
        using namespace strata::kernels;
        layout_ = {f.gu_type, f.d_type, f.n_embd, f.n_ff, f.gu_row, f.d_row, f.up_off, f.down_off, f.bytes};
        blob_bytes_ = f.bytes;
        result_.resize(size_t(Tp5FfnTier::kWindowCap) * H);
        if (!ck(cudaSetDevice(0), "device", err) ||
            !ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err) ||
            !allocate(blobs_, blobs.size(), err) || !allocate(x_, result_.size() * 4, err) ||
            !allocate(weights_, size_t(kEntries) * 4, err) ||
            !allocate(xq_, size_t(Tp5FfnTier::kWindowCap) * (H / 32) * 36, err) ||
            !allocate(scratch_, native_expert_scratch_bytes(kEntries, f.n_ff), err) ||
            !allocate(parts_, size_t(kEntries) * H * 4, err) || !allocate(shared_, result_.size() * 4, err) ||
            !allocate(out_, result_.size() * 4, err) || !allocate(plan_d_, sizeof(Plan), err))
            return false;
        for (int e = 0; e < kEntries; ++e) {
            plan_.start[e] = e;
            plan_.dst[e] = e;
            plan_.tok[e] = e / kTopK;
        }
        plan_.start[kEntries] = kEntries;
        return ck(cudaMemcpyAsync(blobs_, blobs.data(), blobs.size(), cudaMemcpyHostToDevice, stream_), "blobs", err) &&
               ck(cudaMemsetAsync(shared_, 0, result_.size() * 4, stream_), "shared zero", err) &&
               ck(cudaStreamSynchronize(stream_), "load sync", err);
    }
    bool run(const float *x, const int32_t *ids, const float *weights, int n_tok, std::string &err) {
        using namespace strata::kernels;
        const int entries = n_tok * kTopK;
        plan_.groups = entries;
        for (int e = 0; e < entries; ++e) plan_.ptr[e] = (unsigned long long)(blobs_ + size_t(ids[e]) * blob_bytes_);
        if (!ck(cudaMemcpyAsync(x_, x, size_t(n_tok) * H * 4, cudaMemcpyHostToDevice, stream_), "x", err) ||
            !ck(cudaMemcpyAsync(weights_, weights, size_t(entries) * 4, cudaMemcpyHostToDevice, stream_), "weights",
                err) ||
            !ck(cudaMemcpyAsync(plan_d_, &plan_, sizeof(Plan), cudaMemcpyHostToDevice, stream_), "plan", err))
            return false;
        try {
            quantize_q8_1_rows(x_, n_tok, H, xq_, stream_);
            native_expert_grouped(layout_, plan_d_->ptr, plan_d_->start, &plan_d_->groups, plan_d_->dst, plan_d_->tok,
                                  entries, entries, xq_, scratch_, parts_, stream_);
            native_moe_combine_multi(parts_, weights_, shared_, out_, H, kTopK, n_tok, stream_);
        } catch (const std::exception &e) {
            err = "unsplit oracle: " + std::string(e.what());
            return false;
        }
        return ck(cudaGetLastError(), "kernels", err) &&
               ck(cudaMemcpyAsync(result_.data(), out_, size_t(n_tok) * H * 4, cudaMemcpyDeviceToHost, stream_),
                  "output", err) &&
               ck(cudaStreamSynchronize(stream_), "run sync", err);
    }
    const float *result() const { return result_.data(); }
};

/// The UNSPLIT CPU reference: one expert's full down(silu(gate·up)) row for x (H floats),
/// from the pack blob's raw Q2_0 blocks (the artifact's own decoder, so the reference and
/// the GPU agree on what the bits mean).  The blob is [gate @ 0, up @ up_off, down @ down_off]
/// - the LAYOUT's offsets, which for native packs are NOT the fixed 460800 split.
void expert_row_cpu(const uint8_t *blob, const float *x, int64_t n_ff, size_t up_off, size_t down_off, size_t d_row,
                    float *out) {
    // gate: n_ff rows x H cols; up: same; down: H rows x n_ff cols
    std::vector<float> gate(n_ff), up(n_ff);
    for (int64_t r = 0; r < n_ff; ++r) {
        float g = 0.f, u = 0.f;
        for (int64_t c = 0; c < H; c += 64) {
            float w[64];
            plan::q2_0_decode(blob + (size_t(r) * H + c) / 64 * 18, w); // gate row r, col block
            for (int i = 0; i < 64; ++i) g += w[i] * x[c + i];
            plan::q2_0_decode(blob + up_off + (size_t(r) * H + c) / 64 * 18, w);
            for (int i = 0; i < 64; ++i) u += w[i] * x[c + i];
        }
        gate[r] = g;
        up[r] = u;
    }
    std::vector<float> act(n_ff);
    for (int64_t r = 0; r < n_ff; ++r) {
        const float g = gate[r];
        act[r] = g / (1.f + std::exp(-g)) * up[r]; // silu(gate) * up
    }
    for (int64_t r = 0; r < H; ++r) { // down: H rows x n_ff
        float acc = 0.f;
        const uint8_t *down = blob + down_off;
        for (int64_t c = 0; c < n_ff; c += 64) {
            float w[64];
            plan::q2_0_decode(down + size_t(r) * d_row + (size_t(c) / 64) * 18, w);
            for (int i = 0; i < 64; ++i) acc += w[i] * act[c + i];
        }
        out[r] = acc;
    }
}

} // namespace

int main(int argc, char **argv) {
    std::string pack, csv;
    int layer = 0, windows = 4;
    unsigned seed = 42;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--pack") pack = val();
        else if (a == "--layer") layer = std::atoi(val());
        else if (a == "--windows") windows = std::atoi(val());
        else if (a == "--seed") seed = unsigned(std::atoi(val()));
        else if (a == "--csv") csv = val();
        else {
            std::fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (const char *p = std::getenv("STRATA_TP5_PACK")) pack = p;
    if (pack.empty()) {
        std::printf("tp5_ffn_test: no --pack and STRATA_TP5_PACK unset - SKIP "
                    "(the test needs the real expert blobs)\n");
        return 77;
    }
    { // a stale ctest ENVIRONMENT (another machine's pack dir) must skip, not crash:
        // expert_layout_load() treats a missing native_experts.txt as the canonical
        // layout (fmt empty) and lay.fmt[layer] would read out of bounds below
        std::error_code ec;
        if (!std::filesystem::is_directory(pack, ec)) {
            std::printf("tp5_ffn_test: pack dir not present (%s) - SKIP\n", pack.c_str());
            return 77;
        }
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 5) {
        std::printf("tp5_ffn_test: %d devices, need 5 - SKIP\n", ndev);
        return 77;
    }

    // the layer's expert blobs straight from the pack file
    strata::core::FileExpertSource src;
    std::string err;
    // the layout must be LOADED first (generate.cpp does this at startup): without it
    // expert_layout() is the canonical default and fmt[] is empty for native packs
    const int64_t n_expert = 512;
    if (!strata::kernels::cpu::expert_layout_load(pack, 48, n_expert, err)) {
        // a stale pack path (another machine's ctest ENVIRONMENT) is a skip, not a failure:
        // the pack is generated model data, not part of the repo
        std::fprintf(stderr, "tp5_ffn_test: layout load: %s - SKIP (the pack dir is not present)\n", err.c_str());
        return 77;
    }
    const auto &lay = strata::kernels::cpu::expert_layout();
    const int64_t n_layers = (int64_t)lay.fmt.size();
    const int64_t n_ff = lay.fmt[size_t(layer)].n_ff;
    // native packs keep the blobs in the GGUF shards (native_experts.txt names them);
    // a canonical pack has experts.bin.  Both work through FileExpertSource.
    if (const char *g = std::getenv("STRATA_TP5_GGUF")) src.set_gguf(g);
    if (!src.open(pack, n_layers, n_expert, err)) {
        std::fprintf(stderr, "tp5_ffn_test: open pack: %s - SKIP (the pack/GGUF is not present)\n", err.c_str());
        return 77;
    }
    // the tier: ONE layer by default (the reference compares one layer); STRATA_TP5_LAYERS
    // opens more (the 4.2.2 shape) and the reference still checks layer 0
    Tp5FfnTier tier;
    int n_tp5_layers = 1;
    if (const char *v = std::getenv("STRATA_TP5_LAYERS"))
        n_tp5_layers = std::max(1, std::min(int(n_layers), std::atoi(v)));
    int64_t span = 0;
    for (int64_t l = 0; l < n_tp5_layers; ++l) span += lay.blob_bytes(l) * n_expert;
    std::vector<uint8_t> all_blobs(size_t(span), 0);
    {
        size_t off = 0;
        for (int64_t l = 0; l < n_tp5_layers; ++l) {
            const int64_t bb = int64_t(lay.blob_bytes(l));
            for (int64_t e = 0; e < n_expert; ++e) {
                // Native GGUF blobs are assembled into a recycled staging pool.
                // Copy immediately: retaining all 512 pointers aliases old experts
                // once the source reuses a buffer after 256 assemblies.
                const uint8_t *blob = src.blob(l, e);
                if (blob == nullptr) {
                    std::fprintf(stderr, "tp5_ffn_test: layer %lld blob %lld is null\n", (long long)l, (long long)e);
                    return 1;
                }
                std::memcpy(all_blobs.data() + off, blob, size_t(bb));
                off += size_t(bb);
            }
        }
    }
    if (!tier.open(n_tp5_layers, all_blobs.data(), n_expert, err)) {
        std::fprintf(stderr, "tp5_ffn_test: tier open: %s\n", err.c_str());
        return 1;
    }
    const auto &fm = lay.fmt[size_t(layer)];
    const size_t blob_bytes = size_t(fm.bytes);
    std::vector<uint8_t> layer_blobs(blob_bytes * size_t(n_expert));
    for (int64_t e = 0; e < n_expert; ++e)
        std::memcpy(layer_blobs.data() + size_t(e) * blob_bytes, src.blob(layer, e), blob_bytes);
    std::printf("tier open: %d layer%s, %lld experts each, %.2f GiB VRAM total\n", n_tp5_layers,
                n_tp5_layers == 1 ? "" : "s", (long long)n_expert, double(tier.vram_bytes()) / 1073741824.0);
    UnsplitGpuOracle gpu_reference;
    if (!gpu_reference.open(layer_blobs, fm, err)) {
        std::fprintf(stderr, "tp5_ffn_test: %s\n", err.c_str());
        return 1;
    }

    // random activations, ids and weights; several windows of several row counts
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> fx(-0.5f, 0.5f);
    std::uniform_int_distribution<int> fe(0, int(n_expert) - 1);
    std::uniform_real_distribution<float> fw(0.05f, 0.3f);

    FILE *cf = csv.empty() ? nullptr : std::fopen(csv.c_str(), "w");
    if (cf) std::fprintf(cf, "window,n_tok,ready_all_us,cpu_sum_us,total_ms,gpu_max_abs,cpu_max_abs,cpu_rel_l2\n");

    int fails = 0;
    for (int w_i = 0; w_i < windows; ++w_i) {
        if (w_i > 0 && w_i == windows / 2) {
            tier.close();
            if (!tier.open(n_tp5_layers, all_blobs.data(), n_expert, err)) {
                std::fprintf(stderr, "tp5_ffn_test: reopen: %s\n", err.c_str());
                return 1;
            }
        }
        for (int n_tok : {1, 2, 3, 4, 5, 6, 7, 8}) {
            std::vector<float> x(size_t(n_tok) * H);
            std::vector<int32_t> ids(size_t(n_tok) * kTopK);
            std::vector<float> w(size_t(n_tok) * kTopK);
            for (auto &v : x) v = fx(rng);
            for (int t = 0; t < n_tok; ++t) {
                auto first = ids.begin() + t * kTopK;
                *first = t == 0 ? fe(rng) : ids[0]; // Shared expert across distinct activation rows.
                for (int j = 1; j < kTopK; ++j) {
                    int e;
                    do {
                        e = fe(rng);
                    } while (std::find(first, first + j, e) != first + j);
                    first[j] = e;
                }
            }
            for (auto &v : w) v = fw(rng);
            // normalize the weights per token like the router does
            for (int t = 0; t < n_tok; ++t) {
                float s = 0;
                for (int j = 0; j < kTopK; ++j) s += w[size_t(t) * kTopK + j];
                for (int j = 0; j < kTopK; ++j) w[size_t(t) * kTopK + j] /= (s > 0 ? s : 1);
            }
            // The tier applies router weights exactly once and returns one row per token.
            std::vector<float> y(size_t(n_tok) * H, 0.0f);
            if (!tier.run(layer, x.data(), ids.data(), w.data(), n_tok, kTopK, y.data(), err)) {
                std::fprintf(stderr, "tp5_ffn_test: run: %s\n", err.c_str());
                return 1;
            }
            if (!gpu_reference.run(x.data(), ids.data(), w.data(), n_tok, err)) {
                std::fprintf(stderr, "tp5_ffn_test: %s\n", err.c_str());
                return 1;
            }
            double gpu_max_abs = 0, max_abs = 0;
            int elem_fails = 0;
            for (size_t i = 0; i < y.size(); ++i) {
                const double expected = gpu_reference.result()[i];
                const double d = std::fabs(double(y[i]) - expected);
                const double tol = 1e-6 + 1e-5 * std::fabs(expected);
                gpu_max_abs = std::max(gpu_max_abs, d);
                if (!std::isfinite(y[i]) || !std::isfinite(expected) || d > tol) {
                    if (elem_fails < 8)
                        std::fprintf(stderr,
                                     "tp5_ffn_test: window %d n_tok %d elem %zu got %.9g "
                                     "GPU %.9g (d %.2e > tol %.2e)\n",
                                     w_i, n_tok, i, double(y[i]), expected, d, tol);
                    ++elem_fails;
                }
            }
            double squared_error = 0, reference_energy = 0;
            for (int t = 0; t < n_tok; ++t) {
                std::vector<float> ref(H, 0.f), row(H);
                for (int j = 0; j < kTopK; ++j) {
                    const int32_t e = ids[size_t(t) * kTopK + j];
                    expert_row_cpu(layer_blobs.data() + size_t(e) * blob_bytes, x.data() + size_t(t) * H, n_ff,
                                   fm.up_off, fm.down_off, fm.d_row, row.data());
                    const float wj = w[size_t(t) * kTopK + j];
                    for (int64_t i = 0; i < H; ++i) ref[i] += wj * row[i];
                }
                for (int64_t i = 0; i < H; ++i) {
                    const double d = std::fabs(double(y[size_t(t) * H + i]) - double(ref[i]));
                    squared_error += d * d;
                    reference_energy += double(ref[i]) * double(ref[i]);
                    max_abs = std::max(max_abs, d);
                }
            }
            // This is only the CPU/GPU activation-quantization budget. The GPU
            // oracle above separately catches individual missing/stale output rows.
            const double relative_l2 = reference_energy > 0 ? std::sqrt(squared_error / reference_energy)
                                                            : (squared_error == 0 ? 0 : INFINITY);
            const bool ok_window = elem_fails == 0 && std::isfinite(relative_l2) && relative_l2 <= 0.05;
            fails += ok_window ? 0 : 1;
            const auto &t = tier.timings();
            std::printf("window %d n_tok %d: gpu_max_abs %.3e cpu_max_abs %.3e rel_l2 %.3e  ready %.0f us  "
                        "sum %.0f us  %.2f ms  %s\n",
                        w_i, n_tok, gpu_max_abs, max_abs, relative_l2, t.ready_all_us, t.sum_us, t.total_ms,
                        ok_window ? "ok" : "FAIL");
            if (cf)
                std::fprintf(cf, "%d,%d,%.1f,%.1f,%.3f,%.3e,%.3e,%.3e\n", w_i, n_tok, t.ready_all_us, t.sum_us,
                             t.total_ms, gpu_max_abs, max_abs, relative_l2);
        }
    }
    if (cf) std::fclose(cf);
    const bool ok = fails == 0;
    std::printf("tp5_ffn_test: %s (%d failures)\n", ok ? "PASS" : "FAIL", fails);
    return ok ? 0 : 1;
}
