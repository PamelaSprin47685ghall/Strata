// include/strata/core/tp5_gdn.hpp - 06-phase4 §4.2.3: the 5-rank partial GDN tier.
//
// The GDN layers (36 of 48, l % 4 != 3) split five ways by the V-head table
// [10,10,10,10,8]: rank r owns V heads [v_start, v_start+v_count) and the qk
// heads those V heads reference (`qh = v_head % 16`, a GATHER - the qk set is
// contiguous only for ranks 0 and 2).  Per layer, per rank:
//
//   qkv projection  (attn_qkv [2560, 10240]): the rank's output channels -
//                   the q rows of its qk heads, the k rows of the same heads,
//                   the v rows of its V heads - gathered to a compact blob.
//                   The q/k rows are gathered in V-HEAD ORDER (local slot i
//                   holds the qk head of V head i), so the recurrence kernel's
//                   `qh = head % h_k` with h_k = v_count is the identity: the
//                   SHARED kernel needs no change.
//   conv            ssm_conv1d's channel slice (same channels as the qkv
//                   output), the conv history [C_r x 3] per rank; the L2 norm's
//                   qk_heads argument is v_count (the local q|k head count)
//   a/b             ssm_alpha/beta's column slice (V heads), dt/ssm_a's V-head
//                   rows - gate/beta are [n_tok, v_count]
//   z               attn_gate's column slice (V heads): [n_tok, v_count*128]
//   recurrence      gdn_step_norm_multi over the rank's OWN state
//                   ([v_count x 128 x 128] per layer), h_k = v_count
//   publish         the rank's RECURRENCE-OUTPUT slice (V heads
//                   [v_start, v_start+v_count) of the full value dim) through
//                   the RELAY inbox; the CPU gathers the five DISJOINT slices
//                   into the caller's full y rows.
//
// The OUT projection is NOT the tier's: it runs ONCE on GPU0 over the gathered
// full y (the engine's own quantize+dot bit path).  A rank-side split-sum out
// projection's ~1e-3 per-window error accumulated in the recurrence state and
// degenerated the text - the ranks never project bo.
//
// The engine wiring is the QSA tier's own doorbell shape: the graph's GDN mixer
// segment publishes x and rings m_seqG_, waits m_flagG_ and copies the gathered
// y rows.  `commit`/`commit_light` advance the ranks' own state for the kept
// prefix (the recurrence from the untouched state - the engine's own commit
// replay), and `export_state_span` writes that state back into the engine's
// canonical arena when the tier is the state's authority
// (STRATA_TP5_GDN_AUTHORITATIVE).
#pragma once

#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/tp5_relay.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// The 5-rank partial GDN tier (06 §4.2.3).  Owns the 36 GDN layers' projections
/// split by the V-head table, the per-rank conv history and SSM state.
class Tp5GdnTier {
public:
    Tp5GdnTier() = default;
    ~Tp5GdnTier();
    Tp5GdnTier(const Tp5GdnTier&) = delete;
    Tp5GdnTier& operator=(const Tp5GdnTier&) = delete;

    /// Opens the tier.  `gguf_shards` is the model's shard list IN SPLIT ORDER;
    /// STRATA_TP5_GDN_FIXTURE=dir takes the weights from the fixture directory
    /// (one .bin per tensor of GDN layer 1, blk_1_<name>.bin) instead - see the
    /// QSA tier's note on the openat quirk.
    bool open(const std::vector<std::string>& gguf_shards, int64_t n_embd, std::string& err);
    void close();
    bool valid() const { return open_; }

    /// Whether model layer `l` is one of this tier's GDN layers (l % 4 != 3).
    bool is_gdn_layer(int64_t l) const { return open_ && l >= 0 && l < 48 && (l % 4) != 3; }

    /// One window's mixer rows for GDN layer `l` (the MODEL layer number):
    ///   x     : n_tok rows of 2560 floats (the verifier's activation rows, mapped host)
    ///   y     : OUTPUT, n_tok rows of 6144 floats (the recurrence's y - the FULL
    ///            value dim; the graph quantizes and projects it on GPU0, the
    ///            engine's own bit path)
    /// `state0` (TEST ONLY): the recurrence's starting state, [48*128*128 +
    /// 10240*3] floats per layer - the engine starts from zero; the numeric gate
    /// starts from a random one.  Null = the tier's own current state.
    /// `commit_n`: how many of the window's rows to KEEP (the engine passes 0
    /// here and commits separately; the gate passes n to test the state advance).
    bool run(int64_t l, const float* x, int64_t n_tok, float* y, std::string& err,
             const float* state0 = nullptr, int64_t commit_n = 0, int64_t pos0 = -1);

    /// Sync the ranks' state from the ENGINE's own state after the prefill (the
    /// prefill runs its own GDN path - the tier's state stays zero while the
    /// engine's advances; without this the first decode window reads a zero
    /// state and every GDN layer's output is wrong from the first token).
    /// `engine_state` is the engine's per-layer state on GPU0 ([48*128*128 +
    /// 10240*3] floats, the [row][head][col] layout the kernel uses), one call
    /// per GDN layer.
    bool sync_state(int64_t l, const float* engine_state, std::string& err);

    /// The batched form the engine's per-request sync uses: ONE drain of the
    /// ranks' streams and one H2D per (layer, rank) instead of the per-layer
    /// form's 180 device syncs + 23k tiny copies (8.7 s per request on the
    /// 5x RX 6800 rig - measured as the whole "stage" bucket of the GDN=1
    /// decode split, 2026-10-16). `estate` points at the first block of the
    /// session's canonical gdn_state arena; `gdn_floats` is the per-layer stride (the
    /// caller's own constant - the engine's [48 heads x 128x128 + 10240 x 3]).
    bool sync_state_span(int64_t l0, int64_t l1, const float* estate, uint64_t gdn_floats,
                         std::string& err);

    /// The REVERSE of sync_state_span: write the RANKS' OWN state (the
    /// tier-authoritative one - what commit/commit_light/commit_span advanced)
    /// back into the ENGINE's canonical per-layer blocks on GPU0.  `estate`
    /// points at the arena's layer-0 block and `gdn_floats` is the per-layer
    /// stride - the SAME convention as sync_state_span's read side (layer l's
    /// block is estate + lay_of(l)*gdn_floats).  Only the GDN layers of
    /// [l0, l1) are written, ONE H2D per layer of the full block (state + conv
    /// tail, fully covered by the ranks' head/channel maps - no read-modify-
    /// write of the engine's block); layers outside the span and QSA layers
    /// keep whatever the engine left in them.
    ///
    /// PRECONDITION: call AFTER the queued tier commits (commit/
    /// commit_span/commit_light) AND after the caller waited the engine's own
    /// commit (wait_commit), with no verifier graph in flight on GPU0 - the
    /// export drains every RANK device once with a checked cudaDeviceSynchronize
    /// (a failed rank kernel surfaces here; a shared physical device is drained
    /// once), but never device-syncs GPU0 itself (ranks live on GPUs 1..4):
    /// an engine graph spinning on a host flag would deadlock that sync, and the
    /// GPU0 copies must not be assumed to cross an active graph.  GPU0's ordering
    /// is the caller's wait.
    ///
    /// The permutation is exactly the inverse of sync_state_span's gather:
    /// engine (rr, v_start + i) <- rank (rr, i) per row, and the conv channels
    /// back through the rank's channel map.  The staging is tier-owned (one
    /// pair of pinned buffers at the first export, kept to close() - no per-layer
    /// allocation) and the caller's device context is restored on success and
    /// every early error.  On failure `err` is set and nothing is left waiting.
    bool export_state_span(int64_t l0, int64_t l1, float* estate, uint64_t gdn_floats,
                           std::string& err);

    /// The LIGHT self-commit for a one-token window: advance the state with the
    /// window's OWN intermediates (the run's qkv/hb/gate/beta/z are still in the
    /// rank's scratch) - one recurrence kernel per rank, no projections.  The
    /// full-chain commit starved the shared GPU (36 layers x 5 ranks of GEMMs).
    bool commit_light(int64_t l, std::string& err);

    /// Advance the rank's own state for the kept prefix: the recurrence over
    /// tokens [0, n_keep) from the UNTOUCHED state (the engine's own commit
    /// contract - the window's run advanced a scratch copy, not the state) with
    /// the same x rows.  `x` is the window's own rows (mapped host).
    bool commit(int64_t l, int64_t n_keep, const float* x, std::string& err, int64_t pos0 = -1);

    /// The batched form of commit() for the verifier's post-window loop: the
    /// SAME per-layer chain, submitted rank-major (5 device toggles, not
    /// 36x5) so one rank's 36 chains queue behind each other while the host
    /// enqueues the next rank.  `x_base` is the doorbell snapshot
    /// [layer][rows] (host), layer stride `rows` floats; the span [l0, l1)
    /// holds only GDN layers (the caller skips QSA layers itself).
    bool commit_span(int64_t l0, int64_t l1, int64_t n_keep, const float* x_base,
                     size_t rows, std::string& err, int64_t pos0 = -1);

    /// TEST ONLY: the last run's intermediate buffers of rank 0, device pointers
    /// (qkv, hb, gate, beta, z, y) and the rank's state - for the
    /// numeric gate's stage-by-stage comparison.  Null before the first run.
    struct Debug {
        const float *qkv, *hb, *gate, *beta, *z, *y;
        int v_count, c_count;
    };
    Debug debug(int rank) const {
        const RankState& s = rs_[rank];
        return Debug{s.qkv, s.hb, s.gate, s.beta, s.z, s.y, v_count_[rank], c_count_[rank]};
    }
    cudaStream_t debug_stream(int rank) const { return rs_[rank].stream; }
    int debug_gpu(int rank) const { return gpu_of_rank(rank); }
    /// TEST ONLY: the rank's state pointer (layer `l`'s), device
    float* debug_state(int rank, int64_t l) const { return rs_[rank].state[size_t(lay_of(l))]; }
    float* debug_conv(int rank, int64_t l) const { return rs_[rank].conv[size_t(lay_of(l))]; }
    /// TEST ONLY: the run's scratch state right after the state0 injection (rank r)
    const float* debug_state_scratch(int rank) const { return rs_[rank].state_scratch; }

    /// Total device bytes the tier owns (the measurement log).
    int64_t vram_bytes() const { return vram_bytes_; }

    /// The last run's timings (the measurement log).
    struct Timings {
        double ready_us = 0, sum_us = 0, total_ms = 0;  // 06 §5.3: the two host-ns stamps below are the E_RELAY endpoints
        double t0_host_ns = 0, ready_exit_host_ns = 0;
    };
    const Timings& timings() const { return t_; }

    /// 06 §5.3 (STRATA_TP5_TRACE=1): the last run's per-rank producer GPU span, in ns
    /// (gpu_stamp before/after the rank's chain on ITS OWN stream; read back after the
    /// relay tail's ready wait).  All zero when the trace is off (the default).
    struct Trace {
        double prod_span_ns[strata::kernels::tp5::kRanks] = {};
        double prod_end_host_ns[strata::kernels::tp5::kRanks] = {};
        /// 06 §5.3: the host ns each rank's ready flag was first seen >= epoch in
        /// this run's wait loop (the RELAY protocol's ready_first/ready_all, host
        /// clock - no fold needed).  E_RELAY's exposed wait = ready_exit - min_r.
        double ready_seen_ns[strata::kernels::tp5::kRanks] = {};   // 06 §5.3 E_RELAY consumer side
    };
    const Trace& trace() const { return tr_; }

    static constexpr int64_t kEmb = 2560;             ///< n_embd
    static constexpr int kWindowCap = 8;              ///< MAXT: the verify window's cap

private:
    /// The GPU rank r's partial runs on.  Rank 4 shares GPU2 (the QSA rank 4's
    /// own slot - with the QSA tier off, GPU2 carries the FFN rank 1 and the GDN
    /// ranks 1/4): every placement of the GDN's rank 4 starved some FFN rank's
    /// publish when the GDN stream saturates a card - GPU2 with the QSA off is
    /// the least loaded.  With all three tiers
    /// on, the GDN rank 4 should move again (a placement table, not a constant).
    int gpu_of_rank(int r) const { return (r < 4) ? (r + 1) : r4_gpu_; }

    /// GPU for the GDN rank 4 (1..4; default 2). GPU0 is reserved for the
    /// verifier and the FFN tier's single primary CU-masked queue: a second
    /// GPU0 tier stream plus NULL-stream state syncs stalls the first window.
    int r4_gpu_ = 2;

    struct RankState {
        // the split projections (device)
        uint8_t *wqkv = nullptr, *wg = nullptr;
        uint16_t *wa = nullptr, *wb = nullptr;       // alpha/beta (BF16, column slice)
        const float *wc = nullptr;                    // conv1d [C_r x 4] (channel slice)
        const float *wdt = nullptr, *wsa = nullptr;   // dt/ssm_a [v_count]
        const float *wnm = nullptr;                   // norm (mirrored, [128])
        // the rank's own state, per GDN layer: [v_count*128*128] then conv [C_r*3]
        float **state = nullptr, **conv = nullptr;
        float* state_scratch = nullptr;               // the window's recurrence copy
        float* conv_scratch = nullptr;
        // scratch (device)
        float *qkv = nullptr;                         // n_tok x C_r (the projected rows)
        float *hb = nullptr;                          // n_tok x C_r (the conv output)
        float *gate = nullptr, *beta = nullptr;       // n_tok x v_count
        float *z = nullptr;                           // n_tok x ZV_r
        float *y = nullptr;                           // n_tok x ZV_r (the recurrence output)
        float *x_dev = nullptr;                       // n_tok x 2560 input rows
        void *xq = nullptr;                           // q8_1 of the window's rows
        cudaStream_t stream = nullptr;
    };
    RankState rs_[strata::kernels::tp5::kRanks];

    // the plan's V-head table
    int v_count_[strata::kernels::tp5::kRanks] = {10, 10, 10, 10, 8};
    int v_start_[strata::kernels::tp5::kRanks] = {0, 10, 20, 30, 40};
    // the rank's qk heads, in V-head order (slot i = the qk head of V head i):
    // qk = (v_start + i) % 16
    int c_count_[strata::kernels::tp5::kRanks] = {};  // 2*v_count*128 + v_count*128
    // the rank's channel map (per rank, layer-independent): local channel c ->
    // the model's qkv output channel.  q|k slots hold the qk head of V head i
    // ((v_start+i)%16) in V-head order; v slots hold V head v_start+i's rows.
    int chan_[strata::kernels::tp5::kRanks][3 * 1280] = {};   // 3*S*10 = the max C (rank with v_count 10)

    bool open_ = false;
    int64_t n_gdn_layers_ = 0;
    // per-layer GGML types (the projections mix Q2_0/Q3_K/Q4_K/IQ4_XS by layer)
    int lay_qkv_type_[36] = {}, lay_g_type_[36] = {};
    // per-rank per-layer byte spans of the packed blobs
    int64_t lay_qkv_span_[180] = {}, lay_g_span_[180] = {};   // [layer*5+rank]
    int64_t vram_bytes_ = 0;
    uint64_t epoch_ = 0;
    Timings t_{};
    Trace tr_{};

    // 06 §5.3 (STRATA_TP5_TRACE=1): per-rank stamp buffers (2 u64 each: start/end)
    // and the env flag, cached at open().  Off by default: run() issues no stamp
    // kernels and relay_tail() reads nothing back.
    bool trace_on_ = false;
    unsigned long long* trace_dev_[strata::kernels::tp5::kRanks] = {};

    /// Reads the per-rank stamps back on their own streams after the ready wait.
    bool collect_trace(std::string& err);

    // the RELAY state (host-mapped)
    uint8_t* inbox_ = nullptr;
    strata::kernels::tp5::Credit* credits_ = nullptr;
    unsigned* g_abort_ = nullptr;
    uint8_t* inbox_dev_r_[strata::kernels::tp5::kRanks] = {};
    unsigned* g_abort_dev_r_[strata::kernels::tp5::kRanks] = {};
    unsigned* pcounters_[strata::kernels::tp5::kRanks] = {};
    int32_t* n_keep_dev_[strata::kernels::tp5::kRanks] = {};   // the commit's n_keep (device), one per rank
    int32_t* n_keep_h_ = nullptr;   // portable pinned [kRanks], retained until all rank streams drain

    // 06 §4.2/M3-for-GDN (STRATA_TP5_GDN_GRAPH=1): the per-(layer, bucket) graph mode.
    // The eager path above is untouched; graph_ gates the whole block.  The GDN
    // chain's shapes are FIXED given (layer, n_tok) - only x, epoch and bank vary,
    // so unlike the FFN tier (one graph per bucket, expert pointers staged) each
    // (gi, bucket) pair gets its own graph with the layer's pointers baked in:
    // 36 layers x 5 buckets x 5 ranks = 900 small graphs, ~10 nodes each.  The
    // launch-count tally shares the FFN tier's env (STRATA_TP5_LAUNCH_COUNT).
    static bool graph_mode_default();
    bool capture_gdn_rank(int r, int64_t gi, int bucket, std::string& err);
    bool run_graph(int64_t l, const float* x, int64_t n_tok, float* y, std::string& err,
                   int64_t epoch, uint32_t bank, int64_t gi,
                   const std::chrono::steady_clock::time_point& t0);
    bool graph_ = false;
    static constexpr int graph_buckets_ = 5;   // spec 4 -> T <= 5; 6..8 fall back to eager
    cudaGraph_t graphs_[strata::kernels::tp5::kRanks][36 * graph_buckets_] = {};
    cudaGraphExec_t gexecs_[strata::kernels::tp5::kRanks][36 * graph_buckets_] = {};
    float* x_stage_ = nullptr;   // shared portable pinned input for all ranks, retained until streams drain
    strata::kernels::tp5::StepState* step_stage_ = nullptr;
    strata::kernels::tp5::StepState* step_dev_[strata::kernels::tp5::kRanks] = {};
    unsigned graph_launches_[strata::kernels::tp5::kRanks] = {};
    unsigned graph_eager_fallbacks_ = 0;
    bool launch_count_on_ = false;

    // export_state_span's staging: allocated ONCE at the first export (pinned,
    // kept to close()) - one canonical layer block (the H2D source) and one
    // rank's state+conv (the D2H target).  No per-layer allocation.
    float* exp_stage_ = nullptr;
    float* exp_rank_ = nullptr;

public:
    /// 06 §6.3: the graph-mode launch tally (STRATA_TP5_LAUNCH_COUNT=1), per rank.
    unsigned graph_launch_count(int r) const { return graph_launches_[r]; }
    unsigned graph_eager_fallback_count() const { return graph_eager_fallbacks_; }

private:

    int64_t lay_of(int64_t l) const {   // the GDN layer's index in the sequence 0,1,2,4,5,...
        int64_t n = 0;
        for (int64_t m = 0; m < l; ++m)
            if (m % 4 != 3) ++n;
        return n;
    }

    /// The shared CPU relay tail (bounded ready wait, rank-order FP32 sum, credit).
    bool relay_tail(int64_t epoch, uint32_t bank, int64_t n_rows, float* y,
                    const std::chrono::steady_clock::time_point& t0, std::string& err);
};

}  // namespace strata::core
