// include/strata/core/tp5_qsa.hpp - 06-phase4 §4.2.3: the 5-rank partial QSA tier.
//
// The QSA layers (12 of 48, l % 4 == 3) split five ways by the plan's GQA-aligned
// head table [6,6,4,4,4]: rank r owns query heads [q_start, q_start+q_count) and
// exactly ONE kv head.  Per layer, per rank:
//
//   q projection   (attn_q, rows [q_start*512, +q_count*512) of the 12288 rows -
//                   q and its gate half interleave per head, so the rank's rows are
//                   ONE contiguous span)      -> qfull_r (n_tok x q_count x 2*HD)
//   norm/rope      on the q half (the rank's own pos rows)
//   kv projection  (attn_k/attn_v, rows [kv_head*256, +256)) -> kcur/vcur_r
//   norm/rope      on k (v is not normalised)
//   kv append      into the rank's OWN single-head int8 pool (the pool row formula
//                  (page*kv_heads + h)*page_size with kv_heads=1 IS the compact
//                  single-head pool - the same reorder the G-templated attention's
//                  bit-exact gate proved)
//   indexer        MIRRORED (all 4 heads): q_idx projection + block scores + topk
//                  against the rank's own idx_tail/idx_pooled mirror - the selection
//                  is per-rank state, no reduction needed (it feeds only this rank's
//                  attention)
//   attention      qsa_decode_attn_batch with {n_head=q_count, n_head_kv=1} (the
//                  G-templated kernel, bit-exact per head)
//   gate           native_qsa_gate_apply on the rank's heads
//   out projection attn_output's CONTRACTION slice (per output row the input columns
//                  [q_start*256, +q_count*256), stride-extracted to a compact
//                  n_in = q_count*256) -> a PARTIAL bo row (n_tok x 2560)
//   publish        the partial through the RELAY inbox; the CPU sums the five
//                  partials in rank order into the caller's rows.
//
// The verify window's captured graph replaces its whole QSA mixer segment with
// wait_flag_ge(flagA) + copy_from_mapped(bo) - the same doorbell contract the FFN
// tier uses.  The state (the rank's KV pool and indexer mirror) advances on the
// rank itself: `commit` replays the indexer append for the kept prefix exactly as
// the engine's own commit graph replays it, and the KV pool is position-addressed
// so a rejected row's cell is simply rewritten by the next window.
#pragma once

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/tp5_relay.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// The 5-rank partial QSA tier (06 §4.2.3).  Owns the 12 QSA layers' projections
/// split by the GQA-aligned head table, the per-rank single-head KV pools and
/// indexer mirrors.
class Tp5QsaTier {
public:
    Tp5QsaTier() = default;
    ~Tp5QsaTier();
    Tp5QsaTier(const Tp5QsaTier&) = delete;
    Tp5QsaTier& operator=(const Tp5QsaTier&) = delete;

    /// Opens the tier.  `gguf_shards` is the model's shard list IN SPLIT ORDER
    /// (strata::gguf_split_paths of any shard - the caller resolves it, because a
    /// test may reach the files through short symlinks; see the .cu's note on the
    /// openat quirk).  Returns false with `err` on any failure.
    ///
    /// TEST MODE (STRATA_TP5_QSA_FIXTURE=dir): the weights come from the fixture
    /// directory the gate script writes (one .bin per tensor of QSA layer 3,
    /// named blk_3_<tensor>.bin) instead of the GGUF - a test binary's openat of
    /// the >32 GiB shard 1 is unreliable on this stack (see the .cu), and the
    /// fixture is 22 MB.  `gguf_shards` may then be empty.
    bool open(const std::vector<std::string>& gguf_shards, int64_t n_embd, std::string& err);
    void close();
    bool valid() const { return open_; }

    /// Whether model layer `l` is one of this tier's QSA layers (l % 4 == 3).
    bool is_qsa_layer(int64_t l) const { return open_ && l >= 0 && (l % 4) == 3; }

    /// One window's mixer rows for QSA layer `l` (the MODEL layer number):
    ///   x      : n_tok rows of 2560 floats (the verifier's activation rows, mapped host)
    ///   step   : kStepCount int32 per token (mapped host, the verifier's own h_step_)
    ///   pos_q  : ONE position per token (every head of every rank sits at the same
    ///            position - the verifier's own h_pos_ rows are all the token's pos)
    ///   pos_k  : one position per token
    ///   pos_i  : one position per token
    ///   y      : OUTPUT, n_tok rows of 2560 floats (the verifier's bo_ rows, mapped)
    /// Blocks until the five partials are summed (bounded 30 s).
    bool run(int64_t l, const float* x, int64_t n_tok, const int32_t* step, const int32_t* pos_q,
             const int32_t* pos_k, const int32_t* pos_i, float* y, std::string& err);

    /// Advance the rank's own state for the kept prefix (the commit graph's replay):
    /// the indexer append for tokens [0, n_keep) from `raw` (n_keep rows of idx_dim
    /// floats) and the KV pool's cells are position-addressed (already written by
    /// run).  `raw` defaults to the rows run() copied out (the engine's own idx_raw
    /// rows are the graph's, which the tier mode does not produce); `n_commit_step`
    /// is the commit's step record (kStepCount int32, mapped).
    bool commit(int64_t l, int64_t n_keep, const float* raw, const int32_t* commit_step, std::string& err);
    /// The rows run() copied out of rank 0's indexer projection (n_tok rows of
    /// idx_dim floats, host) - the commit's `raw` input.
    const float* last_idx_raw(int64_t l) const { return h_idx_raw_[size_t(l / 4)].data(); }

    /// TEST ONLY: the last run's intermediate buffers of rank 0, device pointers
    /// (qfull, qcur, kcur, vcur, qidx, attn, attn32, y_partial) - for the numeric
    /// gate's stage-by-stage comparison.  Null before the first run.
    struct Debug {
        const float *qfull, *qcur, *kcur, *vcur, *qidx, *attn, *attn32, *y_partial;
        int q_count;
    };
    Debug debug(int rank) const {
        const RankState& s = rs_[rank];
        return Debug{s.qfull, s.qcur, s.kcur, s.vcur, s.qidx, s.attn, s.attn32, s.y_partial, q_count_[rank]};
    }
    cudaStream_t debug_stream(int rank) const { return rs_[rank].stream; }
    int debug_gpu(int rank) const { return gpu_of_rank(rank); }
    const float* debug_gamma(int rank) const { return rs_[rank].wqn; }          // layer 0's q norm
    const float* debug_cos(int rank, int64_t pos) const { return rs_[rank].cos_tab + size_t(pos) * 32; }
    const int32_t* debug_pos(int rank) const { return rs_[rank].pos_q_dev; }
    /// TEST ONLY: the rank's selection ids (device), for the numeric gate
    const int32_t* debug_sel(int rank) const { return rs_[rank].sel; }
    const int32_t* debug_step(int rank) const { return rs_[rank].step_dev; }
    const int8_t* debug_kq(int rank, int64_t qsa_layer) const { return rs_[rank].k_q[qsa_layer]; }
    const float* debug_kcur(int rank) const { return rs_[rank].kcur; }
    const uint16_t* debug_kscale(int rank, int64_t qsa_layer) const { return rs_[rank].k_scale[qsa_layer]; }
    const float* debug_vcur(int rank) const { return rs_[rank].vcur; }
    const void* debug_xq(int rank) const { return rs_[rank].xq; }   // the q8_1 of the window's rows
    const float* debug_x(int rank) const { return rs_[rank].x_dev; }

    /// Total device bytes the tier owns (the measurement log).
    int64_t vram_bytes() const;

    /// The last run's timings (the measurement log).
    struct Timings {
        double ready_us = 0, sum_us = 0, total_ms = 0;  // 06 §5.3: the two host-ns stamps below are the E_RELAY endpoints
        double t0_host_ns = 0, ready_exit_host_ns = 0;
    };
    const Timings& timings() const { return t_; }

    /// 06 §5.3 (STRATA_TP5_TRACE=1): the last run's per-rank producer GPU span, in ns
    /// (gpu_stamp before/after the rank's chain on ITS OWN stream; read back after the
    /// relay tail's ready wait).  `prod_end_host_ns` folds the rank's GPU clock onto
    /// the host clock via a third per-stage calibration stamp (E_RELAY's consumer
    /// side).  All zero when the trace is off (the default).
    struct Trace {
        double prod_span_ns[strata::kernels::tp5::kRanks] = {};
        double prod_end_host_ns[strata::kernels::tp5::kRanks] = {};
        /// 06 §5.3: the host ns each rank's ready flag was first seen >= epoch in
        /// this run's wait loop (the RELAY protocol's ready_first/ready_all, host
        /// clock - no fold needed).  E_RELAY's exposed wait = ready_exit - min_r.
        double ready_seen_ns[strata::kernels::tp5::kRanks] = {};
    };
    const Trace& trace() const { return tr_; }

    static constexpr int64_t kEmb = 2560;             ///< n_embd
    static constexpr int kWindowCap = 8;              ///< MAXT: the verify window's cap
    static constexpr int64_t kMaxTok = kWindowCap;

    // run()'s copied-out indexer rows, per QSA layer (the commit's raw input)
    std::vector<float> h_idx_raw_[12];
    int64_t vram_bytes_ = 0;
    // per-rank PER-LAYER spans (the blob offsets are layer_index * single-layer span;
    // the q_span_ etc below are the TOTALS the blob sizes were checked against)
    int64_t lay_q_span_[60] = {}, lay_k_span_[60] = {}, lay_v_span_[60] = {}, lay_wo_span_[60] = {};   // [layer*5+rank]

private:
    /// The GPU rank r's partial runs on (GPU0 keeps the verify graph).
    /// The GPU rank r's partial runs on.  Rank 4 shares GPU2 (NOT the FFN tier's
    /// GPU1: with the QSA tier on, GPU1 already carries the QSA rank 0 and the FFN
    /// ranks 0/4 - four concurrent tier streams there starved the FFN rank's publish
    /// on long windows; GPU2 carries the QSA ranks 1/4 and the FFN rank 1).
    int gpu_of_rank(int r) const { return (r < 4) ? (r + 1) : 2; }

    struct RankState {
        // the split projections (device)
        uint8_t *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
        uint16_t *wiq = nullptr, *wik = nullptr;      // indexer q/k (BF16, mirrored)
        const float *wqn = nullptr, *wkn = nullptr, *wiqn = nullptr, *wikn = nullptr;  // norms (fp16->fp32, device)
        // the rank's own single-head int8 KV pool, per QSA layer
        int8_t **k_q = nullptr, **v_q = nullptr;      // [n_qsa_layers]
        uint16_t **k_scale = nullptr, **v_scale = nullptr;
        int32_t** page_table = nullptr;               // identity, per layer
        // the indexer mirror, per QSA layer
        float **idx_tail = nullptr, **idx_dead = nullptr, **idx_pooled = nullptr;
        int32_t** idx_block_pos = nullptr;
        // scratch (device)
        float *qfull = nullptr, *qcur = nullptr, *kcur = nullptr, *vcur = nullptr;
        float *qidx = nullptr, *idx_raw = nullptr, *scores = nullptr, *attn = nullptr;
        float* attn32 = nullptr;
        int32_t* sel = nullptr;
        float* scratch_attn = nullptr;
        void* xq = nullptr;                           // q8_1 of the window's rows
        void* xq_q = nullptr;                         // q8_1 for the q projection's input (shared)
        int32_t* step_dev = nullptr;
        int32_t* pos_q_dev = nullptr, *pos_k_dev = nullptr, *pos_i_dev = nullptr;
        float* y_partial = nullptr;                   // n_tok x 2560 (the out-proj partial)
        float* x_dev = nullptr;                       // n_tok x 2560 input rows
        float* cos_tab = nullptr;                     // the rope table (max_cells x n_rot/2)
        float* sin_tab = nullptr;
        cudaStream_t stream = nullptr;
    };
    RankState rs_[strata::kernels::tp5::kRanks];

    // the plan's head table (GQA-aligned)
    int q_count_[strata::kernels::tp5::kRanks] = {6, 6, 4, 4, 4};
    int q_start_[strata::kernels::tp5::kRanks] = {0, 6, 12, 16, 20};
    int kv_head_[strata::kernels::tp5::kRanks] = {0, 0, 1, 1, 1};

    bool open_ = false;
    int64_t n_qsa_layers_ = 0;
    // per-layer GGML types (the projections mix Q2_0/Q3_K/IQ4_XS/... by layer) and the
    // per-layer byte spans inside the rank's packed blobs
    int lay_q_type_[12] = {}, lay_k_type_[12] = {}, lay_v_type_[12] = {}, lay_o_type_[12] = {};
    /// per-rank per-layer byte spans of the packed blobs (q_count differs by rank)
    int64_t q_span_[strata::kernels::tp5::kRanks] = {};
    int64_t k_span_[strata::kernels::tp5::kRanks] = {};    // per-layer span in the packed k blob
    int64_t v_span_[strata::kernels::tp5::kRanks] = {};    // ... and in the v blob (types differ by layer)
    int64_t wo_span_[strata::kernels::tp5::kRanks] = {};
    int64_t max_cells_ = 0;
    int64_t pages_ = 0;
    int64_t cap_ = 0;          // the selection width cap (2051 at 64K, the shapes' own)
    int64_t max_blocks_ = 0;
    uint64_t epoch_ = 0;
    Timings t_{};
    Trace tr_{};

    // 06 §5.3 (STRATA_TP5_TRACE=1): per-rank stamp buffers (2 u64 each: start/end)
    // and the env flag, cached at open().  Off by default: run() issues no stamp
    // kernels and relay_tail() reads nothing back.
    bool trace_on_ = false;
    unsigned long long* trace_dev_[strata::kernels::tp5::kRanks] = {};

    /// Reads the per-rank stamps back (after the ready wait) into tr_.  Trace only.
    void collect_trace();

    int64_t pool_rows_ = 0;    // pages * page_size (single head)

    // the RELAY state (host-mapped), shared shape with the FFN tier's
    uint8_t* inbox_ = nullptr;
    strata::kernels::tp5::Credit* credits_ = nullptr;
    unsigned* g_abort_ = nullptr;
    uint8_t* inbox_dev_r_[strata::kernels::tp5::kRanks] = {};
    unsigned* g_abort_dev_r_[strata::kernels::tp5::kRanks] = {};

    // the publish kernel's grid-wide completion counters (4 slots, epoch & 3)
    unsigned* pcounters_[strata::kernels::tp5::kRanks] = {};

    // the per-head position expansion (host scratch, reused per call)
    std::vector<int32_t> pos_q_exp_, pos_i_exp_;

    /// The shared CPU relay tail (bounded ready wait, rank-order FP32 sum, credit).
    bool relay_tail(int64_t epoch, uint32_t bank, int64_t n_rows, float* y,
                    const std::chrono::steady_clock::time_point& t0, std::string& err);
};

}  // namespace strata::core
