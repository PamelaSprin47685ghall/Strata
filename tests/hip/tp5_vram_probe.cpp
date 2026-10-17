// tests/hip/tp5_vram_probe.cpp - empty-allocation VRAM probe (04 §1.5b).
//
// NOT a performance test: it allocates, on each card, blocks the same size as the ledger's
// static account (weights, KV, scratch, 2-bank mailbox), creates one 96-node hipGraph, and
// reports "ledger bytes vs actually consumed" per card.  That difference replaces the ledger's
// 1.0 GiB runtime estimate.  Skips (exit 77) with fewer than 5 HIP devices.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#define HIP_CHECK(x, what)                                                                     \
    do {                                                                                       \
        const hipError_t e = (x);                                                              \
        if (e != hipSuccess) {                                                                 \
            std::fprintf(stderr, "HIP error %s at %s\n", hipGetErrorString(e), what);          \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

static uint64_t vram_used_sys(int hip_dev) {
    // HIP device ordinals follow KFD order; cardN = card1..card5 for HIP 0..4 (serve/server.py).
    const int card = hip_dev + 1;
    char path[128];
    std::snprintf(path, sizeof path, "/sys/class/drm/card%d/device/mem_info_vram_used", card);
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    uint64_t v = 0;
    if (fscanf(f, "%lu", &v) != 1) v = 0;
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    const char* dir = nullptr;
    uint64_t ctx = 65536;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--gguf-dir") && i + 1 < argc) dir = argv[++i];
        else if (!std::strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = std::strtoull(argv[++i], nullptr, 10);
        else { std::fprintf(stderr, "usage: %s --gguf-dir DIR [--ctx N]\n", argv[0]); return 77; }
    }
    // the ctest run reads STRATA_TP5_GGUF_DIR at RUN time ($ENV{...} in add_test would
    // expand at CONFIGURE time and freeze an empty string into the test command)
    if (!dir) dir = std::getenv("STRATA_TP5_GGUF_DIR");
    if (!dir || !*dir) {
        // the ctest registration passes $ENV{STRATA_TP5_GGUF_DIR}; empty = the model dir is
        // not set on this machine -> SKIP (77), the probe is a measurement not a unit test
        std::printf("hip_tp5_vram_probe: no --gguf-dir (set STRATA_TP5_GGUF_DIR) - SKIP\n");
        return 77;
    }
    {   // the dir must actually be there: a stale ctest ENVIRONMENT pointing at another
        // machine's model dir would otherwise run the allocation on a box without the model
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) {
            std::printf("hip_tp5_vram_probe: gguf dir not present (%s) - SKIP\n", dir);
            return 77;
        }
    }

    int n = 0;
    HIP_CHECK(hipGetDeviceCount(&n), "hipGetDeviceCount");
    if (n < 5) {
        std::printf("tp5_vram_probe: only %d HIP devices, need 5 - SKIP\n", n);
        return 77;
    }

    // Ledger numbers from 1.5a (ctx=65536, split-proj=0, mtp-rank=2, the heaviest config that
    // still fits): per-rank bytes by component.  We allocate them as real blocks.
    struct Block { const char* name; uint64_t bytes; };
    const uint64_t GiB = 1073741824ull;
    const uint64_t MiB = 1048576ull;
    const std::vector<Block> rank_blocks = {
        {"weights (routed 1/5 + dense + proj + vocab)", uint64_t(6.513 * GiB) + uint64_t(0.866 * GiB) +
                                                          uint64_t(1.288 * GiB) + uint64_t(0.081 * GiB)},
        {"KV int8", 12 * ctx * 528},
        {"indexer keys", 12 * (ctx / 4 + 2) * 128 * 4},
        {"GDN state", uint64_t(0.023 * GiB)},
        {"MTP + scratch (rank 2 only, others 0)", 835 * MiB + 55 * MiB}, // added to rank 2 below
        {"prompt buffer", uint64_t(3.330 * GiB)},
        {"mailbox 2 banks", 2 * 5 * (10240 + 128) * 2},
    };
    uint64_t ledger[5] = {};
    for (int r = 0; r < 5; ++r) {
        for (const auto& b : rank_blocks) ledger[r] += b.bytes;
        if (r == 2) ledger[r] += 835 * MiB + 55 * MiB; // MTP+scratch on the mtp rank
        ledger[r] -= 835 * MiB + 55 * MiB; // (the shared row above is rank-2-only; recompute below)
    }
    // recompute cleanly: the shared MTP row is only for rank 2
    for (int r = 0; r < 5; ++r) {
        uint64_t s = 0;
        for (size_t i = 0; i < rank_blocks.size(); ++i) {
            if (i == 4) { if (r == 2) s += rank_blocks[i].bytes; }
            else s += rank_blocks[i].bytes;
        }
        ledger[r] = s;
    }

    std::printf("%-6s %12s %14s %14s %12s\n", "rank", "ledger B", "sys before B", "sys after B",
                "delta GiB");
    for (int r = 0; r < 5; ++r) {
        HIP_CHECK(hipSetDevice(r), "hipSetDevice");
        const uint64_t before = vram_used_sys(r);
        size_t free_h = 0, total_h = 0;
        HIP_CHECK(hipMemGetInfo(&free_h, &total_h), "hipMemGetInfo");
        // allocate the ledger's blocks
        std::vector<void*> ptrs;
        bool ok = true;
        for (const auto& b : rank_blocks) {
            const bool is_mtp = std::strstr(b.name, "MTP");
            const uint64_t want = is_mtp && r != 2 ? 0 : b.bytes;
            if (want == 0) continue;
            void* p = nullptr;
            if (hipMalloc(&p, want) != hipSuccess) { ok = false; break; }
            ptrs.push_back(p);
        }
        // one 96-node hipGraph: capture 96 trivial kernel launches on a stream
        hipStream_t stream;
        HIP_CHECK(hipStreamCreate(&stream), "hipStreamCreate");
        hipGraph_t graph = nullptr;
        hipGraphExec_t exec = nullptr;
        if (ok) {
            if (hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal) == hipSuccess) {
                for (int i = 0; i < 96; ++i) {
                    // a no-op kernel: hipMemsetAsync of one word is a graph node
                    if (hipMemsetAsync(ptrs.empty() ? nullptr : ptrs[0], 0, 4, stream) != hipSuccess) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    if (hipStreamEndCapture(stream, &graph) != hipSuccess || !graph) ok = false;
                    else if (hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != hipSuccess) ok = false;
                }
            } else {
                ok = false;
            }
        }
        const uint64_t after = vram_used_sys(r);
        size_t free2 = 0, total2 = 0;
        HIP_CHECK(hipMemGetInfo(&free2, &total2), "hipMemGetInfo2");
        std::printf("%-6d %12lu %14lu %14lu %12.3f%s\n", r, ledger[r], before, after,
                    double(int64_t(after) - int64_t(before)) / 1073741824.0, ok ? "" : " ALLOC-FAIL");
        std::printf("       hipMemGetInfo: free %zu -> %zu MiB (total %zu MiB)\n", free_h >> 20,
                    free2 >> 20, total2 >> 20);
        if (exec) hipGraphExecDestroy(exec);
        if (graph) hipGraphDestroy(graph);
        HIP_CHECK(hipStreamDestroy(stream), "hipStreamDestroy");
        for (void* p : ptrs) hipFree(p);
    }
    return 0;
}
