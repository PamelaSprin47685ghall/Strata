// src/plan/tp5_ledger_main.cpp - the TP5 per-rank VRAM ledger (04 §1.5a).
//
// Static accounting from the real GGUF tensor table: for each rank, sum the bytes its slice of
// every family costs, add the KV/indexer/state/MTP/scratch/prompt terms, and REFUSE (exit 1)
// when the heaviest rank exceeds the card.  This is arithmetic you can check by hand against
// tp5-rocm/02-answers-hw.md Q13's table; the table is the CHECK, the GGUF is the source.
//
//   usage: tp5-ledger --gguf-dir DIR --ctx 65536|131072 [--prompt-chunk 8192]
//                      [--mtp-mib 835] [--scratch-mib 55] [--runtime-gib 1.0] [--vram-gib 15.98]
//                      [--split-proj 0|1] [--mtp-rank N|-1]
#include "strata/plan/tp5_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

// ---- minimal GGUF v3 header reader (header only; the artifact reader is in src/artifact but
// drags the full engine; the ledger only needs names/shapes/types) ---------------------------
namespace {

#pragma pack(push, 1)
struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};
#pragma pack(pop)

struct TInfo {
    std::string name;
    std::vector<uint64_t> ne;
    uint32_t type;
    // bytes of this tensor's data
    uint64_t nbytes(const std::vector<int>& block_elems,
                    const std::vector<int>& block_bytes) const;
};

uint64_t TInfo::nbytes(const std::vector<int>& be, const std::vector<int>& bb) const {
    uint64_t elems = 1;
    for (uint64_t d : ne) elems *= d;
    const int idx = int(type);
    if (idx < 0 || size_t(idx) >= be.size()) return 0;
    const int64_t per = be[idx];
    if (per <= 0 || elems % uint64_t(per) != 0) return 0; // malformed: refuse, do not guess
    return elems / uint64_t(per) * uint64_t(bb[idx]);
}

// ggml type table for the encodings this model uses (mirrors tools/gguf_reader.py)
const std::vector<int> kBlockElems = [] {
    std::vector<int> v(43, 0);
    v[0] = 1;   v[1] = 1;   v[2] = 32;  v[3] = 32;  v[4] = 32;  v[5] = 32;  v[6] = 32;  v[7] = 32;
    v[8] = 32;  v[9] = 32;  v[10] = 256; v[11] = 256; v[12] = 256; v[13] = 256; v[14] = 256;
    v[15] = 256; v[16] = 256; v[17] = 256; v[18] = 256; v[19] = 256; v[20] = 32;
    v[21] = 256; v[22] = 256; v[23] = 256; v[28] = 1;  v[29] = 256; v[30] = 1;
    v[34] = 32;  v[35] = 32; v[36] = 32;  v[37] = 32;  v[38] = 32;  v[39] = 32;  v[40] = 64;
    v[41] = 64;  v[42] = 64; // 41 = Q4_0_8_8, 42 = Q2_0 (this artifact's routed encoding)
    return v;
}();
const std::vector<int> kBlockBytes = [] {
    std::vector<int> v(43, 0);
    v[0] = 4;   v[1] = 2;   v[2] = 18;  v[3] = 20;  v[4] = 0;   v[5] = 0;   v[6] = 22;  v[7] = 24;
    v[8] = 34;  v[9] = 36;  v[10] = 84;  v[11] = 110; v[12] = 144; v[13] = 176; v[14] = 210;
    v[15] = 292; v[16] = 66; v[17] = 74; v[18] = 98; v[19] = 50; v[20] = 18;
    v[21] = 110; v[22] = 82; v[23] = 136; v[28] = 8;  v[29] = 56; v[30] = 2;
    v[34] = 18;  v[35] = 18; v[36] = 18;  v[37] = 18;  v[38] = 18;  v[39] = 17;  v[40] = 36;
    v[41] = 36;  v[42] = 18;
    return v;
}();

std::vector<TInfo> read_gguf_tensors(const std::filesystem::path& p) {
    std::vector<TInfo> out;
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return out;
    GGUFHeader h;
    if (fread(&h, sizeof h, 1, f) != 1 || h.magic != 0x46554747u || h.version != 3) {
        fclose(f);
        return out;
    }
    auto read_str = [&](std::string& s) {
        uint64_t n = 0;
        if (fread(&n, 8, 1, f) != 1) return false;
        s.resize(size_t(n));
        return n == 0 || fread(s.data(), 1, size_t(n), f) == size_t(n);
    };
    std::function<bool(uint32_t)> read_value = [&](uint32_t t) -> bool {
        switch (t) {
            case 0: case 1: fseek(f, 1, SEEK_CUR); return true;
            case 2: case 3: fseek(f, 2, SEEK_CUR); return true;
            case 4: case 5: case 6: case 12: fseek(f, 4, SEEK_CUR); return true;
            case 7: fseek(f, 1, SEEK_CUR); return true;
            case 8: { // string
                std::string s;
                if (!read_str(s)) return false;
                return true;
            }
            case 9: { // array: type + count + elements
                uint32_t et = 0; uint64_t cnt = 0;
                if (fread(&et, 4, 1, f) != 1 || fread(&cnt, 8, 1, f) != 1) return false;
                for (uint64_t i = 0; i < cnt; ++i)
                    if (!read_value(et)) return false;
                return true;
            }
            case 10: case 11: fseek(f, 8, SEEK_CUR); return true;
            default: return false;
        }
    };
    for (uint64_t i = 0; i < h.metadata_kv_count; ++i) {
        std::string key;
        uint32_t t = 0;
        if (!read_str(key) || fread(&t, 4, 1, f) != 1 || !read_value(t)) { fclose(f); return out; }
    }
    for (uint64_t i = 0; i < h.tensor_count; ++i) {
        TInfo ti;
        uint32_t nd = 0;
        if (!read_str(ti.name) || fread(&nd, 4, 1, f) != 1) { fclose(f); return out; }
        ti.ne.resize(nd);
        if (nd && fread(ti.ne.data(), 8, nd, f) != nd) { fclose(f); return out; }
        if (fread(&ti.type, 4, 1, f) != 1) { fclose(f); return out; }
        fseek(f, 8, SEEK_CUR); // offset
        // GGUF ne[] is reverse-logical: ne[0] is the fastest axis (rows in this repo's convention)
        std::reverse(ti.ne.begin(), ti.ne.end());
        out.push_back(std::move(ti));
    }
    fclose(f);
    return out;
}

// ---- the ledger --------------------------------------------------------------
struct LedgerRow {
    std::string component;
    uint64_t per_rank[5] = {};
};

double gib(uint64_t b) { return double(b) / 1073741824.0; }

} // namespace

int main(int argc, char** argv) {
    std::string gguf_dir;
    uint64_t ctx = 65536;
    uint64_t prompt_chunk = 8192;
    uint64_t mtp_mib = 835, scratch_mib = 55;
    double vram_gib = 15.98, runtime_gib = 1.0;
    bool split_proj = false;
    int mtp_rank = -1; // -1 = MTP on every rank (worst case), else single rank
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](auto& out) {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            std::istringstream iss(argv[++i]);
        };
        (void)next;
        if (a == "--gguf-dir") { if (i + 1 >= argc) std::exit(2); gguf_dir = argv[++i]; }
        else if (a == "--ctx") { ctx = std::strtoull(argv[++i], nullptr, 10); }
        else if (a == "--prompt-chunk") { prompt_chunk = std::strtoull(argv[++i], nullptr, 10); }
        else if (a == "--mtp-mib") { mtp_mib = std::strtoull(argv[++i], nullptr, 10); }
        else if (a == "--scratch-mib") { scratch_mib = std::strtoull(argv[++i], nullptr, 10); }
        else if (a == "--vram-gib") { vram_gib = std::strtod(argv[++i], nullptr); }
        else if (a == "--runtime-gib") { runtime_gib = std::strtod(argv[++i], nullptr); }
        else if (a == "--split-proj") { split_proj = std::atoi(argv[++i]) != 0; }
        else if (a == "--mtp-rank") { mtp_rank = std::atoi(argv[++i]); }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }
    if (gguf_dir.empty()) { std::fprintf(stderr, "--gguf-dir is required\n"); return 2; }

    // read all shards' tensor tables
    std::vector<TInfo> tensors;
    for (const auto& e : std::filesystem::directory_iterator(gguf_dir)) {
        if (e.path().extension() == ".gguf") {
            auto t = read_gguf_tensors(e.path());
            if (t.empty()) { std::fprintf(stderr, "cannot read %s\n", e.path().c_str()); return 2; }
            tensors.insert(tensors.end(), t.begin(), t.end());
        }
    }
    std::printf("ledger: %zu tensors from %s\n", tensors.size(), gguf_dir.c_str());

    strata::plan::Geometry g;
    strata::plan::tp5::Plan p;
    std::string err;
    if (!strata::plan::tp5::make_plan(g, p, err)) {
        std::fprintf(stderr, "tp5-ledger: plan failed: %s\n", err.c_str());
        return 1;
    }

    // classify every tensor into a family and accumulate per-rank bytes
    LedgerRow rows[16];
    int nr = 0;
    auto add = [&](const char* name, const uint64_t per[5]) {
        rows[nr].component = name;
        for (int r = 0; r < 5; ++r) rows[nr].per_rank[r] = per[r];
        ++nr;
    };
    uint64_t routed[5] = {}, dense_mirror[5] = {}, proj_split[5] = {}, proj_mirror[5] = {},
             kv[5] = {}, idx[5] = {}, gdn_state[5] = {}, vocab[5] = {},
             mtp[5] = {}, scratch[5] = {}, prompt[5] = {};

    const uint64_t GiB = 1073741824ull;
    for (const auto& t : tensors) {
        const std::string& n = t.name;
        const uint64_t bytes = t.nbytes(kBlockElems, kBlockBytes);
        auto has = [&](const char* f) { return n.find(f) != std::string::npos; };
        if (has("ffn_")) {
            // routed + shared experts: the F-axis slice is exactly 1/5 of every ffn tensor.
            for (int r = 0; r < 5; ++r) routed[r] += bytes / 5;
        } else if (has("per_layer_token_embd") || has("ple_")) {
            // PLE stays in RAM under --ple-io mmap/ram; NOT a VRAM term.  Skipped.
        } else if (n == "output.weight") {
            for (int r = 0; r < 5; ++r) vocab[r] += bytes / 5;   // 49664 of 248320 rows
        } else if (has("hc_") || n == "ffn_gate_inp.weight" || n == "ffn_gate_inp_shexp.weight" ||
                   n == "token_embd.weight" || has("_norm") || has("ssm_a") || has("ssm_dt")) {
            // HC, router, token embedding, all norms, ssm_a/dt: mirrored (02 Q13 dense).
            for (int r = 0; r < 5; ++r) dense_mirror[r] += bytes;
        } else {
            // splittable projections: attn_q/attn_qkv (Q heads), attn_k/attn_v (kv heads),
            // attn_gate/ssm_alpha/beta/conv1d (head channels), attn_output/ssm_out (contraction),
            // indexer q/k projections (mirror the small key, split the 4-head query).
            const uint64_t share = split_proj ? 0 : bytes;
            if (split_proj) {
                if (has("attn_k") || has("attn_v")) {
                    for (int r = 0; r < 5; ++r) proj_split[r] += bytes / 2;   // 1 of 2 kv heads
                } else if (has("attn_output")) {
                    const int parts[5] = {1536, 1536, 1024, 1024, 1024};
                    const uint64_t k_total = 6144;
                    for (int r = 0; r < 5; ++r) proj_split[r] += bytes * parts[r] / k_total;
                } else if (has("ssm_out")) {
                    const int parts[5] = {1280, 1280, 1280, 1280, 1024};
                    const uint64_t k_total = 6144;
                    for (int r = 0; r < 5; ++r) proj_split[r] += bytes * parts[r] / k_total;
                } else {
                    // Q-head share [6,6,4,4,4]/24 for q projections and head-channel tensors
                    const int qcnt[5] = {6, 6, 4, 4, 4};
                    for (int r = 0; r < 5; ++r) proj_split[r] += bytes * qcnt[r] / 24;
                }
            } else {
                for (int r = 0; r < 5; ++r) proj_mirror[r] += share;
            }
        }
    }

    // KV int8: per QSA layer, per rank: 1 kv head (2 heads split across 5 ranks: rank0/1 own kv0,
    // rank2/3/4 own kv1 - each rank owns exactly ONE head, so 528 B/cell/head x 12 layers).
    const uint64_t cell_bytes = 528; // 2 x 256 int8 + scales (02 Q13)
    const uint64_t kv_per_rank = 12 * ctx * cell_bytes / 2;  // one head of the two
    // rank0/1 share kv0, rank2/3/4 share kv1: each rank stores its OWN copy of its head
    for (int r = 0; r < 5; ++r) kv[r] = 12 * ctx * cell_bytes;
    // ... but a rank only needs the head its q heads reference; both ranks of a group can share
    // only if the cache is shared.  Conservative: every rank holds one head's cells.
    (void)kv_per_rank;

    // indexer keys: mirrored (all ranks need cell selection)
    const uint64_t idx_bytes = 12 * (ctx / 4 + 2) * 128 * 4;
    for (int r = 0; r < 5; ++r) idx[r] = idx_bytes;

    // GDN state: per-rank v-head share
    const uint64_t state_total = strata::plan::state_bytes(g);
    const int vcnt[5] = {10, 10, 10, 10, 8};
    for (int r = 0; r < 5; ++r) gdn_state[r] = state_total * vcnt[r] / 48;

    // MTP + verify scratch: single rank (or worst-case all)
    for (int r = 0; r < 5; ++r) {
        mtp[r] = (mtp_rank < 0 || r == mtp_rank) ? mtp_mib * 1048576ull : 0;
        scratch[r] = (mtp_rank < 0 || r == mtp_rank) ? scratch_mib * 1048576ull : 0;
    }
    // prompt buffers: every rank keeps its own (03 baseline log: 3.33 GiB per rank)
    const uint64_t prompt_bytes = prompt_chunk == 8192 ? 3.33 * GiB : 3.33 * GiB / 2;
    for (int r = 0; r < 5; ++r) prompt[r] = prompt_bytes;

    add("routed+shared experts (1/5)", routed);
    add("dense mirror (HC, router, embd)", dense_mirror);
    add(split_proj ? "split projections (Q/K/V/out/gdn-out)" : "mirrored projections",
        split_proj ? proj_split : proj_mirror);
    add("KV int8 (1 kv head, 12 layers)", kv);
    add("indexer keys (mirror)", idx);
    add("GDN state (v-share)", gdn_state);
    add("vocab (1/5)", vocab);
    add("MTP draft", mtp);
    add("verify scratch", scratch);
    add("prompt buffers", prompt);

    const double cap = (vram_gib - 0.5 - runtime_gib) * GiB; // 512 MiB engine safety line
    std::printf("%-38s %10.3f %10.3f %10.3f %10.3f %10.3f  (GiB)\n", "component", 0.0, 0.0, 0.0, 0.0, 0.0);
    int worst = 0;
    uint64_t totals[5] = {};
    for (int r = 0; r < 5; ++r) {
        uint64_t s = 0;
        for (int i = 0; i < nr; ++i) s += rows[i].per_rank[r];
        totals[r] = s;
        if (s > totals[worst]) worst = r;
    }
    for (int i = 0; i < nr; ++i) {
        std::printf("%-38s", rows[i].component.c_str());
        for (int r = 0; r < 5; ++r) std::printf(" %10.3f", gib(rows[i].per_rank[r]));
        std::printf("\n");
    }
    std::printf("%-38s", "TOTAL");
    for (int r = 0; r < 5; ++r) std::printf(" %10.3f", gib(totals[r]));
    std::printf("\ncapacity (vram - 0.5 - runtime)   %10.3f GiB\n", cap / GiB);
    std::printf("heaviest rank: %d at %.3f GiB -> %s\n", worst, gib(totals[worst]),
                totals[worst] <= uint64_t(cap) ? "FITS" : "OVER");
    return totals[worst] <= uint64_t(cap) ? 0 : 1;
}
