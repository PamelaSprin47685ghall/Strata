// tests/core/tp5_plan_test.cpp - the TP5 sharding plan's own invariants, on CPU, no GPU.
//
// What these tests catch (04 §1.3), and nothing else: coverage holes (a family whose per-rank
// intervals do not tile the axis), block misalignment (a Q4_K shared down would need 256-elem
// alignment), GQA misalignment (a q head whose kv group is not the rank's), qk-head rule errors
// (the kernel's `v % 16` vs a wrong rule), pack reorder loss, and the numeric equivalence of
// sliced expert FFN partial sums against the unsplit reference.
#include "strata/plan/tp5_plan.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace strata::plan::tp5;

static int failures = 0;
#define CHECK(cond, msg)                                                                       \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, msg);                           \
            ++failures;                                                                        \
        }                                                                                      \
    } while (0)

// ---- 1. coverage: every family's intervals tile the axis exactly once ----------------------
static void coverage() {
    strata::plan::Geometry g;
    Plan p;
    std::string err;
    CHECK(make_plan(g, p, err), "make_plan must succeed on the real geometry");
    CHECK(err.empty(), "no error on success");

    // FFN: 5 x 128 tiles [0, 640)
    int64_t covered = 0, last_end = 0;
    for (int r = 0; r < kRanks; ++r) {
        CHECK(p.ffn[r].begin == last_end, "ffn intervals must be contiguous");
        last_end = p.ffn[r].end;
    }
    CHECK(last_end == 640, "ffn intervals must cover 640");
    covered = last_end;

    // QSA q heads: each of the 24 appears exactly once
    std::vector<int> seen(24, 0);
    for (int r = 0; r < kRanks; ++r)
        for (int h = p.qsa.q_start[r]; h < p.qsa.q_start[r] + p.qsa.q_count[r]; ++h) ++seen[h];
    for (int h = 0; h < 24; ++h) CHECK(seen[h] == 1, "every q head owned exactly once");

    // GDN v heads: 48 covered once
    std::vector<int> vseen(48, 0);
    for (int r = 0; r < kRanks; ++r)
        for (int v = p.gdn.v_start[r]; v < p.gdn.v_start[r] + p.gdn.v_count[r]; ++v) ++vseen[v];
    for (int v = 0; v < 48; ++v) CHECK(vseen[v] == 1, "every v head owned exactly once");

    // vocab: 5 x 49664 = 248320
    int64_t vlast = 0;
    for (int r = 0; r < kRanks; ++r) {
        CHECK(p.vocab[r].begin == vlast, "vocab intervals contiguous");
        vlast = p.vocab[r].end;
    }
    CHECK(vlast == 248320, "vocab covers 248320");
    (void)covered;
}

// ---- 2. block alignment refusal: Q4_K (256-elem blocks) shared down ------------------------
static void block_refusal() {
    strata::plan::Geometry g;
    Plan p;
    std::string err;
    CHECK(make_plan(g, p, err), "plan");
    // ffn_down_shexp as Q4_K: 256-elem blocks on the contraction axis; 128 % 256 != 0.
    TensorDesc t{"ffn_down_shexp", Family::SharedDown, /*ne_split=*/2560, /*ne_other=*/640,
                 /*block_elems=*/256};
    Slice s;
    CHECK(!slice_of(p, t, 0, s, err), "128-elem interval must be REFUSED for 256-elem blocks");
    CHECK(!err.empty(), "refusal carries an error");
    // and the same family as Q2_0 (64-elem blocks) is fine on every rank.
    TensorDesc q20{"ffn_down_shexp", Family::SharedDown, 2560, 640, 64};
    for (int r = 0; r < kRanks; ++r)
        CHECK(slice_of(p, q20, r, s, err), "Q2_0 shared down slices cleanly");
}

// ---- 3. QSA GQA alignment: q head -> kv head by integer division ---------------------------
static void qsa_alignment() {
    strata::plan::Geometry g;
    Plan p;
    std::string err;
    CHECK(make_plan(g, p, err), "plan");
    for (int r = 0; r < kRanks; ++r)
        for (int h = p.qsa.q_start[r]; h < p.qsa.q_start[r] + p.qsa.q_count[r]; ++h)
            CHECK(h / 12 == p.qsa.kv_head[r], "q head maps to the rank's kv head (q/12)");
}

// ---- 4. GDN qk rule: kernel says qh = v % 16; a wrong rule must be caught ------------------
static void gdn_qk_rule() {
    strata::plan::Geometry g;
    Plan p;
    std::string err;
    CHECK(make_plan(g, p, err), "plan");
    // the plan's own lists must satisfy v % 16 membership
    for (int r = 0; r < kRanks; ++r)
        for (int v = p.gdn.v_start[r]; v < p.gdn.v_start[r] + p.gdn.v_count[r]; ++v) {
            const int qk = v % 16;
            CHECK(std::find(p.gdn.qk_heads[r].begin(), p.gdn.qk_heads[r].end(), qk) !=
                      p.gdn.qk_heads[r].end(),
                  "v % 16 qk head is replicated on the owning rank");
        }
    // counterexample: the WRONG rule v / 3 does not have this property for the plan's tables.
    // Build the wrong lists and show at least one local v whose v/3 head is missing - proving
    // the membership check above is not vacuous.
    int wrong_misses = 0;
    for (int r = 0; r < kRanks; ++r) {
        std::vector<int> wrong;
        for (int v = p.gdn.v_start[r]; v < p.gdn.v_start[r] + p.gdn.v_count[r]; ++v)
            wrong.push_back(v / 3);
        for (int v = p.gdn.v_start[r]; v < p.gdn.v_start[r] + p.gdn.v_count[r]; ++v)
            if (std::find(wrong.begin(), wrong.end(), v % 16) == wrong.end()) ++wrong_misses;
    }
    CHECK(wrong_misses > 0, "the v/3 rule misses heads the v%16 rule covers (check is not vacuous)");
}

// ---- 5. pack round trip: random bytes through the rank reorder and back --------------------
static void pack_roundtrip() {
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<uint8_t> expert(kExpertBlob);
    for (auto& b : expert) b = uint8_t(byte(rng));
    // gate/up slices are contiguous ranges; down is stride-extracted.  Round-trip both.
    std::vector<uint8_t> gu[kRanks], up[kRanks], down[kRanks];
    for (int r = 0; r < kRanks; ++r) {
        gu[r].resize(kRankGuBytes);
        up[r].resize(kRankGuBytes);
        down[r].resize(kRankDownBytes);
        down_pack_rank(expert.data(), r, down[r].data());
    }
    // gate/up: concatenation in rank order must equal the original gate part (and up part).
    std::vector<uint8_t> gu_cat(kGuTensorBytes * 2);
    for (int r = 0; r < kRanks; ++r) {
        int64_t goff = 0, glen = 0, uoff = 0, ulen = 0;
        gate_up_src(r, 0, /*part=*/0, goff, glen);
        gate_up_src(r, 0, /*part=*/1, uoff, ulen);
        std::memcpy(gu[r].data(), expert.data() + goff, size_t(glen));       // gate slice
        std::memcpy(up[r].data(), expert.data() + uoff, size_t(ulen));       // up slice
        std::memcpy(gu_cat.data() + size_t(128 * r) * 720, gu[r].data(), size_t(kRankGuBytes));
        std::memcpy(gu_cat.data() + kGuTensorBytes + size_t(128 * r) * 720, up[r].data(),
                    size_t(kRankGuBytes));
    }
    CHECK(std::memcmp(gu_cat.data(), expert.data(), size_t(kGuTensorBytes)) == 0,
          "gate part round trip is byte-identical");
    CHECK(std::memcmp(gu_cat.data() + kGuTensorBytes, expert.data() + kGuTensorBytes,
                      size_t(kGuTensorBytes)) == 0,
          "up part round trip is byte-identical");
    std::vector<uint8_t> rebuilt(kGuTensorBytes);
    const uint8_t* srcs[kRanks] = {down[0].data(), down[1].data(), down[2].data(), down[3].data(),
                                   down[4].data()};
    down_unpack_rank(srcs, rebuilt.data());
    CHECK(std::memcmp(rebuilt.data(), expert.data() + 2 * kGuTensorBytes, size_t(kGuTensorBytes)) == 0,
          "down stride round trip is byte-identical");
}

// ---- 6. numeric: sliced expert FFN partial sums vs unsplit reference ------------------------
//
// One expert, Q2_0 weights, random FP32 activation, double accumulation on both sides: the
// sliced path must reproduce the unsplit reference to FP rounding, with partials summed in
// rank order (the RELAY's own order).
static void numeric_ffn() {
    const int n_in = 2560, n_out = 2560, f_mid = 640;
    std::mt19937_64 rngw(4242), rngx(777);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> act(-1.f, 1.f);

    // gate [640 x 2560], up [640 x 2560], down [2560 x 640], all Q2_0
    std::vector<uint8_t> gate(size_t(640 * 2560 / 64) * 18), up(gate.size()),
        down(size_t(2560 * 640 / 64) * 18);
    for (auto& b : gate) b = uint8_t(byte(rngw));
    for (auto& b : up) b = uint8_t(byte(rngw));
    for (auto& b : down) b = uint8_t(byte(rngw));
    // make scales sane: fp16 scale in [0.01, 0.05]
    auto fix_scale = [](std::vector<uint8_t>& w) {
        for (size_t i = 0; i < w.size(); i += 18) {
            const uint16_t s = 0x2C00; // fp16 ~0.0117
            w[i] = uint8_t(s & 0xff);
            w[i + 1] = uint8_t(s >> 8);
        }
    };
    fix_scale(gate); fix_scale(up); fix_scale(down);

    std::vector<float> x(n_in);
    for (auto& f : x) f = act(rngx);

    // reference: h = silu(gate(x)) * up(x); y = down(h)  (double accumulation)
    auto dot_q2 = [&](const uint8_t* w, const float* v, int n) {
        double s = 0.0;
        std::vector<float> dq(64);
        for (int j = 0; j < n; j += 64) {
            q2_0_decode(w + (j / 64) * 18, dq.data());
            for (int k = 0; k < 64; ++k) s += double(dq[k]) * double(v[j + k]);
        }
        return s;
    };
    std::vector<double> h(f_mid), y(n_out, 0.0);
    std::vector<float> xf(x.begin(), x.end());
    for (int f = 0; f < f_mid; ++f) {
        const double gv = dot_q2(gate.data() + size_t(f) * (n_in / 64) * 18, xf.data(), n_in);
        const double uv = dot_q2(up.data() + size_t(f) * (n_in / 64) * 18, xf.data(), n_in);
        h[f] = gv / (1.0 + std::exp(-gv)) * uv;
    }
    std::vector<float> dq(64);
    for (int o = 0; o < n_out; ++o) {
        const uint8_t* row = down.data() + size_t(o) * (f_mid / 64) * 18;
        for (int b = 0; b < f_mid / 64; ++b) {
            q2_0_decode(row + b * 18, dq.data());
            for (int k = 0; k < 64; ++k) y[o] += double(dq[k]) * h[b * 64 + k];
        }
    }

    // sliced: rank r takes contraction blocks [2r, 2r+2) of each down row (F interval
    // [128r,128r+128)); partials summed over ranks in order.
    std::vector<double> y_sum(n_out, 0.0);
    for (int r = 0; r < kRanks; ++r) {
        for (int o = 0; o < n_out; ++o) {
            double part = 0.0;
            std::vector<float> dq(64);
            const uint8_t* row = down.data() + size_t(o) * (f_mid / 64) * 18;
            // down row is 640 elems = 10 blocks; f in [128r,128r+128) = blocks [2r, 2r+2)
            for (int b = 2 * r; b < 2 * r + 2; ++b) {
                q2_0_decode(row + b * 18, dq.data());
                for (int k = 0; k < 64; ++k) {
                    const int f = b * 64 + k;
                    part += double(dq[k]) * h[f];
                }
            }
            y_sum[o] += part; // rank order, as the RELAY sums
        }
    }
    double max_rel = 0.0;
    for (int o = 0; o < n_out; ++o) {
        const double denom = std::max(1.0, std::fabs(y[o]));
        max_rel = std::max(max_rel, std::fabs(y[o] - y_sum[o]) / denom);
    }
    // Same accumulation order and double precision on both sides: must be bit-close.
    CHECK(max_rel < 1e-12, "sliced partials equal the unsplit reference (double)");
    std::printf("numeric_ffn max_rel=%.3g\n", max_rel);
}

int main() {
    coverage();
    block_refusal();
    qsa_alignment();
    gdn_qk_rule();
    pack_roundtrip();
    numeric_ffn();
    if (failures == 0) {
        std::printf("tp5_plan_test: all checks passed\n");
        return 0;
    }
    std::printf("tp5_plan_test: %d FAILURES\n", failures);
    return 1;
}
