// include/strata/plan/tp5_plan.hpp - TP5 (5-way tensor parallel) sharding plan.
//
// Phase 1 of the TP5-ROCm plan (tp5-rocm/04-phase1-plan.md).  This header is the arithmetic that
// decides WHICH slice of every tensor family lives on which rank, plus the pack-view reordering
// that makes a rank's slice look like an ordinary matrix to `native_q2_0_mmvq`.  It is deliberately
// CPU-only and GPU-free: the plan must be checkable by hand before any engine code trusts it.
//
// DESIGN CONSTRAINTS (from 02-answers-hw.md Q10 / 04 §1.2):
//   * QSA head split is GQA-ALIGNED: every rank owns exactly ONE kv head, so `q / 12` (the
//     integer-division q->kv map the attention kernel already uses) stays correct per rank with
//     no bridge rank and no kernel change.  The balanced [5,5,5,5,4] table is NOT used: it puts
//     rank 2 across two kv groups and was the 83.7% slowest-rank signature in the Vulkan ledger.
//   * GDN qk heads are replicated per rank following the kernel's own `qh = v_head % h_k`.
//   * Every interval must be block-aligned for the tensor it slices; a misaligned slice is
//     REFUSED, not rounded.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "strata/artifact/dequant.hpp"
#include "strata/plan/plan.hpp"

namespace strata::plan::tp5 {

inline constexpr int kRanks = 5;

/// Half-open interval [begin, end).
struct Interval {
    int64_t begin = 0;
    int64_t end = 0;
    int64_t size() const { return end - begin; }
    bool operator==(const Interval& o) const { return begin == o.begin && end == o.end; }
};

/// QSA query-head split, GQA-aligned.  rank r owns q heads [q_start[r], q_start[r]+q_count[r])
/// and kv head kv_head[r]; every q head h in that range must satisfy h / (nq/nkv) == kv_head[r].
struct QsaMap {
    int q_count[kRanks] = {};
    int q_start[kRanks] = {};
    int kv_head[kRanks] = {};
};

/// GDN value-head split.  rank r owns V heads [v_start[r], v_start[r]+v_count[r]) and replicates
/// the qk heads those V heads reference: qk_heads[r] = { v % 16 : v in range }.
struct GdnMap {
    int v_count[kRanks] = {};
    int v_start[kRanks] = {};
    std::vector<int> qk_heads[kRanks];
};

struct Plan {
    Interval ffn[kRanks];       // F-axis channel interval (gate/up rows AND down contraction)
    QsaMap qsa;
    GdnMap gdn;
    Interval vocab[kRanks];     // output.weight row split (vocabulary axis)
    int mtp_rank = 2;           // default: a rank that owns neither head-full mirror nor 2 kv heads
};

/// A tensor's slice on one rank: a row range or a contraction-axis range, in BLOCKS of the
/// tensor's own quantization.  A slice that is not a whole number of blocks is refused.
struct Slice {
    bool contraction_axis = false;
    Interval range;             // element range on the split axis
    int64_t block_elems = 0;    // elements per quant block on that axis
};

/// Tensor family: what it is and how it splits.  Families the plan does not shard are `Mirror`.
enum class Family {
    RoutedGateUp,   // ffn_{gate,up}_exps: rows on F
    RoutedDown,     // ffn_down_exps: contraction on F
    SharedGateUp,   // ffn_{gate,up}_shexp: rows on F
    SharedDown,     // ffn_down_shexp: contraction on F
    QsaQ,           // attn_q: rows on Q heads (head-aligned)
    QsaKv,          // attn_k/attn_v: rows on the rank's kv head
    QsaOut,         // attn_output: contraction on Q heads
    GdnOut,         // ssm_out: contraction on V heads
    Vocab,          // output.weight: rows on vocab
    Mirror,         // HC, router, norms, embeddings: replicated
};

/// A named tensor as the ledger sees it (name fragment, family, quant block elements on the
/// split axis).  The ledger is data-driven so the same plan code checks the real GGUF.
struct TensorDesc {
    const char* fragment = nullptr;   // matched by substring, e.g. "ffn_gate_exps"
    Family family = Family::Mirror;
    int64_t ne_split = 0;             // extent on the split axis (rows or contraction elems)
    int64_t ne_other = 0;             // extent on the other axis (for byte accounting)
    int64_t block_elems = 1;          // quant block size on the split axis (1 for row-split F32)
    bool per_expert = false;          // routed experts: 512 copies
    int n_experts = 512;
};

/// head_dim on the QSA split axes (256 in this artifact; a constant because the plan's slice
/// arithmetic is per-head and the geometry carries it as `head_dim`).
inline constexpr int64_t kQsaHeadDim = 256;

/// Build the plan from the model geometry.  Returns false and fills `err` when any invariant
/// fails; a plan that cannot be honoured is worse than no plan.
inline bool make_plan(const Geometry& g, Plan& p, std::string& err) {
    // ---- FFN F-axis: 5 x 128 of 640 --------------------------------------
    // 640 = 5 * 128, and 128 elements = 2 Q2_0 blocks (64/blk) on the row side and 2 blocks on
    // the down side.  Every family that slices F must therefore be block-aligned at 128.
    for (int r = 0; r < kRanks; ++r) {
        p.ffn[r] = Interval{int64_t(128 * r), int64_t(128 * (r + 1))};
    }

    // ---- QSA query heads: GQA-aligned [6,6,4,4,4] over 24 q heads, 2 kv heads.
    const int nq = g.qsa_n_head, nkv = g.n_kv_heads;
    if (nkv != 2) { err = "TP5 plan expects exactly 2 kv heads, got " + std::to_string(nkv); return false; }
    if (nq % nkv != 0) { err = "nq not a multiple of nkv"; return false; }
    const int q_per_group = nq / nkv;   // 12
    // rank0/1 -> kv0 (6 q heads each), rank2/3/4 -> kv1 (4 each): 6+6+4+4+4 = 24.
    const int qcnt[5] = {6, 6, 4, 4, 4};
    int qh = 0;
    for (int r = 0; r < kRanks; ++r) {
        p.qsa.kv_head[r] = (r <= 1) ? 0 : 1;
        if (qh + qcnt[r] > nq) { err = "q head table overruns nq"; return false; }
        p.qsa.q_start[r] = qh;
        p.qsa.q_count[r] = qcnt[r];
        // GQA alignment invariant: every q head of this rank maps to kv_head[r] under integer div.
        for (int h = qh; h < qh + qcnt[r]; ++h)
            if (h / q_per_group != p.qsa.kv_head[r]) {
                err = "q head " + std::to_string(h) + " crosses rank " + std::to_string(r) + "'s kv group";
                return false;
            }
        qh += qcnt[r];
    }

    // ---- GDN value heads: [10,10,10,10,8] over 48, qk replication by v % 16.
    const int nv = g.ssm_v_heads;
    if (nv != 48) { err = "TP5 plan expects 48 GDN v heads, got " + std::to_string(nv); return false; }
    const int vcnt[5] = {10, 10, 10, 10, 8};
    int vh = 0;
    for (int r = 0; r < kRanks; ++r) {
        p.gdn.v_start[r] = vh;
        p.gdn.v_count[r] = vcnt[r];
        p.gdn.qk_heads[r].clear();
        for (int v = vh; v < vh + vcnt[r]; ++v) {
            const int qk = v % g.gdn_q_heads;
            if (std::find(p.gdn.qk_heads[r].begin(), p.gdn.qk_heads[r].end(), qk) ==
                p.gdn.qk_heads[r].end())
                p.gdn.qk_heads[r].push_back(qk);
        }
        std::sort(p.gdn.qk_heads[r].begin(), p.gdn.qk_heads[r].end());
        vh += vcnt[r];
    }

    // ---- vocabulary: 248320 = 5 x 49664, Q5_K blocks of 256: 49664 = 194 blocks exactly.
    const int64_t vocab = g.vocab_size;
    const int64_t per = vocab / kRanks;
    if (per * kRanks != vocab || per % 256 != 0) {
        err = "vocab " + std::to_string(vocab) + " does not split into 5 Q5_K-aligned parts";
        return false;
    }
    for (int r = 0; r < kRanks; ++r) p.vocab[r] = Interval{per * r, per * (r + 1)};

    p.mtp_rank = 2;
    return true;
}

/// The slice of tensor `t` on `rank`.  Refuses (returns false) when the interval is not
/// block-aligned for this tensor's quantization, or the family/axis does not apply.
inline bool slice_of(const Plan& p, const TensorDesc& t, int rank, Slice& s, std::string& err) {
    if (rank < 0 || rank >= kRanks) { err = "rank out of range"; return false; }
    auto set = [&](Interval iv, bool contraction) {
        if (iv.begin < 0 || iv.end > t.ne_split || iv.size() <= 0) {
            err = "interval [" + std::to_string(iv.begin) + "," + std::to_string(iv.end) +
                  ") outside split axis extent " + std::to_string(t.ne_split);
            return false;
        }
        if (iv.size() % t.block_elems != 0 || iv.begin % t.block_elems != 0) {
            err = std::string("interval of ") + t.fragment + " not aligned to " +
                  std::to_string(t.block_elems) + "-element blocks on rank " + std::to_string(rank);
            return false;
        }
        s.contraction_axis = contraction;
        s.range = iv;
        s.block_elems = t.block_elems;
        return true;
    };
    const int64_t hd = kQsaHeadDim;
    switch (t.family) {
        case Family::RoutedGateUp:
        case Family::SharedGateUp:
            return set(p.ffn[rank], false);
        case Family::RoutedDown:
        case Family::SharedDown:
            return set(p.ffn[rank], true);
        case Family::QsaQ: {
            // rows on the rank's q heads: head_dim rows per q head
            const Interval iv{int64_t(p.qsa.q_start[rank]) * hd,
                              int64_t(p.qsa.q_start[rank] + p.qsa.q_count[rank]) * hd};
            return set(iv, false);
        }
        case Family::QsaKv: {
            // one kv head = head_dim rows
            const Interval iv{int64_t(p.qsa.kv_head[rank]) * hd, int64_t(p.qsa.kv_head[rank] + 1) * hd};
            return set(iv, false);
        }
        case Family::QsaOut: {
            // contraction on q heads: head_dim elems per head
            const Interval iv{int64_t(p.qsa.q_start[rank]) * hd,
                              int64_t(p.qsa.q_start[rank] + p.qsa.q_count[rank]) * hd};
            return set(iv, true);
        }
        case Family::GdnOut: {
            // ssm_out contraction on V heads: ds=128 elems per head
            const Interval iv{int64_t(p.gdn.v_start[rank]) * 128,
                              int64_t(p.gdn.v_start[rank] + p.gdn.v_count[rank]) * 128};
            return set(iv, true);
        }
        case Family::Vocab:
            return set(p.vocab[rank], false);
        case Family::Mirror:
            s.contraction_axis = false;
            s.range = Interval{0, t.ne_split};
            s.block_elems = t.block_elems;
            return true;
    }
    err = "unreachable family";
    return false;
}

// ---- pack view: the per-rank expert blob reorder (4 §1.3) -------------------
//
// The Strata expert blob layout (tools/iq_pack.py) is, per expert, 3 tensors x 460,800 B:
//   gate rows  [0, 460800)      : 640 rows of 720 B (40 Q2_0 blocks of 18 B)
//   up rows    [460800, 921600) : same
//   down rows  [921600, 1382400): 2560 rows of 180 B (10 blocks)
// A rank's slice:
//   gate/up: rows [128r, 128(r+1))  -> CONTIGUOUS 92,160 B at offset 128r*720
//   down:    per output row, blocks [2r, 2r+2) of 10  -> stride 180 B, 36 B per row
// After the reorder each rank holds a compact per-rank blob of 276,480 B where the down part is
// [2560 rows x 36 B] contiguous - which is exactly a Q2_0 matrix with n_in=128 (2 blocks/row).

inline constexpr int64_t kExpertBlob = 1382400;      // bytes per expert, all three tensors
inline constexpr int64_t kGuTensorBytes = 460800;    // gate (or up, or down) part per expert
inline constexpr int64_t kGuRowBytes = 720;          // 40 blocks x 18 B
inline constexpr int64_t kDownRowBytes = 180;        // 10 blocks x 18 B
inline constexpr int64_t kGuRowsPerExpert = 640;
inline constexpr int64_t kDownRowsPerExpert = 2560;
inline constexpr int64_t kRankGuBytes = 128 * 720;   // 92,160
inline constexpr int64_t kRankDownBytes = 2560 * 36; // 92,160

/// Source byte range of part `part` (0=gate, 1=up) of expert `e` for rank `r` in a pack blob.
inline void gate_up_src(int rank, int64_t expert, int part, int64_t& off, int64_t& len) {
    off = expert * kExpertBlob + int64_t(part) * kGuTensorBytes + int64_t(128 * rank) * kGuRowBytes;
    len = kRankGuBytes;
}

/// Copy one expert's down part for rank `r` from the pack blob into a compact per-rank layout.
/// `dst` must hold kRankDownBytes bytes.  Stride-extract: row j, blocks [2r, 2r+2).
inline void down_pack_rank(const uint8_t* src_expert, int rank, uint8_t* dst) {
    const uint8_t* down = src_expert + 2 * kGuTensorBytes;
    for (int j = 0; j < kDownRowsPerExpert; ++j) {
        const uint8_t* row = down + int64_t(j) * kDownRowBytes + int64_t(2 * rank) * 18;
        std::memcpy(dst + int64_t(j) * 36, row, 36);
    }
}

/// Inverse of down_pack_rank: rebuild a full expert's down part from 5 rank blobs.  Used by the
/// round-trip test to prove the reorder is lossless.
inline void down_unpack_rank(const uint8_t* const* rank_srcs, uint8_t* dst_expert_down) {
    for (int j = 0; j < kDownRowsPerExpert; ++j) {
        for (int r = 0; r < kRanks; ++r) {
            std::memcpy(dst_expert_down + int64_t(j) * kDownRowBytes + int64_t(2 * r) * 18,
                        rank_srcs[r] + int64_t(j) * 36, 36);
        }
    }
}

/// Q2_0 block decode: the artifact header's own reference (same bits, subnormals included).
inline void q2_0_decode(const uint8_t* block, float* out64) {
    strata::dequantize_q2_0(block, out64);
}

} // namespace strata::plan::tp5
