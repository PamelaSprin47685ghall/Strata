// include/strata/core/tp5_ffn.hpp - 06-phase4 §4.2.1-4.2.2: the 5-rank partial FFN tier.
//
// The routed-expert computation split five ways on the F axis (the plan's ffn[r] = 128
// channels): every rank holds its 1/5 slice of every expert blob of EVERY TP5 LAYER (the
// pack reorder of tp5_plan.hpp, using the layout's own up_off/down_off), computes the
// partial `down_r(silu(gate_r·up_r))` rows with `native_expert_grouped`
// (one call for the whole window's entries - the engine's own grouped kernel, the same
// arithmetic the peer tier uses). Each rank reduces its ten expert rows with the
// router weights before publishing ONE row per token. The TP5 RELAY sums these five
// weighted partials (bank alternation, GPU-written ready, bounded waits). The verifier
// adds only its shared expert; it must NOT apply the router weights a second time.
//
// Default placement: ranks 0..3 run on GPUs 1..4, rank 4 shares GPU1. GPU0 carries
// the verifier; putting a rank there needs both VRAM headroom and an independent
// hardware queue, because the verifier waits for that rank's result.
#pragma once

#include "strata/kernels/tp5_relay.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// The 5-rank partial FFN tier (06 §4.2.1/§4.2.2).  Owns the TP5 LAYERS' routed experts
/// split five ways; `open` takes the layers' blobs (contiguous, `n_layers * n_expert`
/// experts of `lay.blob_bytes(l)` each - the layout's per-layer sizes).
class Tp5FfnTier {
  public:
    Tp5FfnTier() = default;
    ~Tp5FfnTier();
    Tp5FfnTier(const Tp5FfnTier &) = delete;
    Tp5FfnTier &operator=(const Tp5FfnTier &) = delete;

    /// Opens the tier for layers [0, n_layers) on the five GPUs.  `blobs` is the layers'
    /// routed expert blobs, layer-major: layer l's expert e at
    /// `blobs + (Σ_{l'<l} n_expert*bytes(l')) + e*bytes(l)`.  Returns false with `err` on
    /// any failure - a tier that cannot open is a startup error, never silent.
    bool open(int64_t n_layers, const uint8_t *blobs, int64_t n_expert, std::string &err);
    void close();
    bool valid() const { return open_; }
    int64_t n_layers() const { return n_layers_; }
    /// Whether `layer` runs on the tier (all of them once opened).
    bool is_tp5_layer(int64_t layer) const { return open_ && layer >= 0 && layer < n_layers_; }

    /// One window's WEIGHTED routed rows for `layer`, split five ways and summed:
    ///   x : n_tok rows of 2560 floats (the verifier's activation rows)
    ///   ids : n_tok*k routing ids; k = 10
    ///   weights : n_tok*k router weights, applied once on each rank
    ///   y : OUTPUT, n_tok rows of 2560 floats; shared expert is NOT included
    /// Blocks until the five partials are summed (bounded 30 s; a timeout is a hard
    /// error, never a partial row).
    bool run(int64_t layer, const float *x, const int32_t *ids, const float *weights, int64_t n_tok, int64_t k,
             float *y, std::string &err);

    /// Capturable final combine: read the mapped routed rows and add the already
    /// gated shared expert. TP5 does not use the all-resident LFUSE path.
    void combine(const float *routed, const float *shared, float *output, int64_t n_tok, void *stream) const;

    /// The last run's timings (for the measurement log).
    struct Timings {
        double ready_all_us = 0, sum_us = 0,
               total_ms = 0; // 06 §5.3: the two host-ns stamps below are the E_RELAY endpoints
        double t0_host_ns = 0, ready_exit_host_ns = 0;
        double stage_launch_us =
            0; // 06 shift-6: staging + the 5 graph launches (inside ready_all_us; the flag wait is the rest)
    };
    /// 06 §6.3 launch proof: the per-rank graph-launch tally since open().  The eager
    /// path's runs are NOT launches - `eager_runs()` reports them separately.
    long long graph_launches(int r) const { return graph_launches_[r]; }
    long long eager_runs() const { return eager_runs_; }
    const Timings &timings() const { return t_; }

    /// 06 §5.3 (STRATA_TP5_TRACE=1): the last run's per-rank producer GPU span, in ns
    /// (gpu_stamp before/after the rank's chain on ITS OWN stream; the stamps are
    /// read back after the relay tail's ready wait, so the span is complete).  The
    /// slowest rank of the run is `prod_span_ns[argmax]`.
    ///
    /// E_RELAY's consumer side: a third stamp (slot 2) lands on the rank's stream
    /// right after slot 1, so the host reads it back at a known wall time and folds
    /// the rank's GPU clock onto the host clock for THIS stage (a per-stage
    /// calibration - the GPU counters drift, a startup-only one would not see it).
    /// `prod_end_host_ns[r]` is then slot1's GPU time expressed on the host clock:
    /// the decomposition's `max_r(producer_end)` in the same units as the host's
    /// ready/sum stamps.  All zero when the trace is off (the default: no stamp
    /// kernels, no readback).
    struct Trace {
        double prod_span_ns[strata::kernels::tp5::kRanks] = {};
        double prod_end_host_ns[strata::kernels::tp5::kRanks] = {};
        /// 06 §5.3: the host ns each rank's ready flag was first seen >= epoch in
        /// this run's wait loop (the RELAY protocol's ready_first/ready_all, host
        /// clock - no fold needed).  E_RELAY's exposed wait = ready_exit - min_r.
        double ready_seen_ns[strata::kernels::tp5::kRanks] = {};
    };
    const Trace &trace() const { return tr_; }

    /// VRAM the tier holds across all ranks (for the §4.2.2 ledger check).
    int64_t vram_bytes() const;

    static constexpr int64_t kEmb = 2560; ///< n_embd
    static constexpr int kWindowCap = 8;  ///< MAXT: the verify window's cap

    /// STRATA_TP5_GRAPH=1 records each rank's layer call (quantize -> grouped ->
    /// weighted publish), with one graph per token-count bucket. The real group
    /// count is device-read, so unused groups do no work. Two captured H2D nodes
    /// carry the activations+weights and the packed plan+epoch/bank metadata.
    /// The varying inputs ride pinned staging buffers
    /// whose copies are CAPTURED as graph nodes - the values are read at launch time,
    /// never captured constants. Default OFF; eager uses the same weighted contract.
    static bool graph_mode_default();

  private:
    /// The GPU rank r's partial runs on (GPU0 stays with the verify graph; see the .cu).
    int gpu_of_rank(int r) const;

    /// Rank 4's GPU (STRATA_TP5_FFN_R4_GPU, cached at open()): 1 = the historical
    /// placement (shared with rank 0 on GPU1), 0 = GPU0 (the verify graph blocks on
    /// the tier's own flag there, so its CUs are idle during the relay).
    int r4_gpu_ = 1;

    bool open_ = false;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    uint64_t epoch_ = 0;
    Timings t_{};
    Trace tr_{};

    // 06 §5.3 (STRATA_TP5_TRACE=1): per-rank stamp buffers (2 u64 each: start/end)
    // and the env flag, cached at open().  Off by default: run() issues no stamp
    // kernels and relay_tail() reads nothing back.
    bool trace_on_ = false;
    unsigned long long *trace_dev_[strata::kernels::tp5::kRanks] = {};

    /// Reads the per-rank stamps back (after the ready wait) into tr_.  Trace only.
    void collect_trace();

    // per-rank device state (rank r lives on gpu_of_rank(r))
    uint8_t *w_dev_[strata::kernels::tp5::kRanks] = {};              // the rank's 1/5 blobs, all layers
    float *x_dev_[strata::kernels::tp5::kRanks] = {};                // window activations followed by router weights
    float *y_dev_[strata::kernels::tp5::kRanks] = {};                // the UNWEIGHTED partial rows (n_tok*k x H)
    void *q8_[strata::kernels::tp5::kRanks] = {};                    // q8_1 of the window's rows
    void *scratch_[strata::kernels::tp5::kRanks] = {};               // native_expert_grouped scratch
    void *meta_dev_[strata::kernels::tp5::kRanks] = {};              // pointers | grouped plan | step (one H2D)
    void *meta_stage_[strata::kernels::tp5::kRanks] = {};            // pinned mirror of meta_dev_
    const void *meta_mapped_[strata::kernels::tp5::kRanks] = {};     // device aliases for kernel staging
    unsigned long long *ptr_dev_[strata::kernels::tp5::kRanks] = {}; // group blob addrs (device)
    int32_t *start_dev_[strata::kernels::tp5::kRanks] = {};
    int32_t *dst_dev_[strata::kernels::tp5::kRanks] = {};
    int32_t *tok_dev_[strata::kernels::tp5::kRanks] = {};
    int32_t *cnt_dev_[strata::kernels::tp5::kRanks] = {};
    cudaStream_t pstream_[strata::kernels::tp5::kRanks] = {};

    // graph mode (M3): per-rank PER-BUCKET graphs (bucket = n_tok-1, 1..kWindowCap) + the
    // pinned staging the graph's H2D nodes read.  The bucket's grid is sized for its real
    // max groups; the plan counts are device-read and the publish grid fixes its
    // token count, so one graph serves every window of that row count.
    static constexpr int graph_buckets_ = kWindowCap;
    bool graph_ = false;
    bool stage_kernel_ = false; // STRATA_TP5_STAGE_KERNEL=1: one mapped-input staging/quantize node
    // 06 §6.3 launch proof (STRATA_TP5_LAUNCH_COUNT=1): per-rank cudaGraphLaunch tally.
    // Host-side ++ at the launch site only - no GPU work, no stream traffic.
    bool launch_count_on_ = false;
    long long graph_launches_[strata::kernels::tp5::kRanks] = {};
    long long eager_runs_ = 0; // the eager path's per-layer runs (not graphs)
    cudaGraph_t graphs_[strata::kernels::tp5::kRanks][size_t(graph_buckets_)] = {};
    cudaGraphExec_t gexecs_[strata::kernels::tp5::kRanks][size_t(graph_buckets_)] = {};
    float *x_stage_ = nullptr; // shared MAXT*(H+10) floats (rows followed by router weights)
    const float *x_mapped_[strata::kernels::tp5::kRanks] = {};
    unsigned long long *ptr_stage_[strata::kernels::tp5::kRanks] = {}; // aliases meta_stage_
    int32_t *plan_stage_[strata::kernels::tp5::kRanks] = {};           // start|dst|tok|cnt in meta_stage_
    strata::kernels::tp5::StepState *step_stage_[strata::kernels::tp5::kRanks] = {};
    /// Captures rank r's bucket-`bucket` graph (open() calls it when graph_ is set).
    bool capture_rank(int r, int bucket, std::string &err);
    /// The graph-mode run: fills every rank's staging (x, the plan, the step's
    /// epoch/bank), launches the five graphs, then the shared CPU relay tail.
    bool run_graph(int64_t layer, const float *x, const int32_t *ids, const float *weights, int64_t n_tok, int64_t k,
                   float *y, std::string &err);
    /// The CPU relay tail both paths share: bounded ready wait, the rank-order FP32 sum,
    /// the credit publish, the timings.
    bool relay_tail(int64_t epoch, uint32_t bank, int64_t n, float *y, const std::chrono::steady_clock::time_point &t0,
                    std::string &err);
    /// Builds the grouped plan into the h_* staging arrays (distinct experts in routing
    /// order, one row per entry).
    void plan_of(const int32_t *ids, int64_t n, int64_t k, int &groups, int &entries);

    // the RELAY state (host-mapped)
    uint8_t *inbox_ = nullptr;
    strata::kernels::tp5::Credit *credits_ = nullptr;
    unsigned *g_abort_ = nullptr;
    // per-rank device aliases
    uint8_t *inbox_dev_r_[strata::kernels::tp5::kRanks] = {};
    // host staging for the grouped plan (rebuilt per run)
    std::vector<int32_t> h_start_, h_dst_, h_tok_, h_cnt_;
    // per-layer slice sizes (the layout's own)
    std::vector<int64_t> slice_bytes_;
    std::vector<int64_t> layer_offs_;
    int64_t slice_bytes(int64_t layer) const { return slice_bytes_[size_t(layer)]; }
    int64_t layer_off(int64_t layer) const { return layer_offs_[size_t(layer)]; }

    // the publish kernel's grid-wide completion counters (4 slots, epoch & 3)
    unsigned *pcounters_[strata::kernels::tp5::kRanks] = {};
};

} // namespace strata::core
