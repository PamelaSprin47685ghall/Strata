// tests/hip/tp5_relay.cpp - the TP5 RELAY probe (05-phase2 §2.1-2.7 + §3).
//
// A model-free, five-rank, 96-stage, varying-payload closed loop with a real consumer, the
// fault matrix (§2.4), the four broadcast placements (§2.5), per-stage CSV timing (§2.6), the
// send-wall ladder (§2.7) and the hipGraph mode (§3: one launch per rank per token).
//
//   usage: hip_tp5_relay [--tokens N] [--rows R] [--bcast fine|uncached|hsa-coarse|host]
//                        [--mode eager|persistent|graph] [--csv FILE] [--fault LIST]
//                        [--active-ranks N] [--stages S]
//   faults: withhold (gen never published), partial (4/5 gen), truncate (ready before
//           payload, producer launch skipped), wrong-bank (consumer reads the other bank)
//   exit 0 = pass (fault runs: fault DETECTED), 1 = fail, 77 = fewer than 5 HIP devices
//
// Timing model (05 §2.6): GPU intervals come from wall_clock64 ticks converted with the
// per-rank calibrated ns/tick; CPU intervals from steady_clock.  The two clocks are never
// subtracted from each other.
#include "strata/kernels/tp5_relay.hpp"

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <atomic>
#include <immintrin.h>   // _mm_sfence: the relay gen publish fence
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define HIP_CHECK(x, what)                                                                     \
    do {                                                                                       \
        const hipError_t e = (x);                                                              \
        if (e != hipSuccess) {                                                                 \
            std::fprintf(stderr, "HIP error %s at %s\n", hipGetErrorString(e), what);          \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

using namespace strata::kernels::tp5;
using clk = std::chrono::steady_clock;
namespace chrono = std::chrono;

namespace {

constexpr int kDefaultStages = 96;
constexpr int kTimeoutMs = 2000; // GGML_TP5_RELAY_HANDOFF_TIMEOUT_MS default (05 §2.3)

double us(chrono::steady_clock::duration d) {
    return chrono::duration<double, std::micro>(d).count();
}
double ms(chrono::steady_clock::duration d) {
    return chrono::duration<double, std::milli>(d).count();
}

struct Args {
    int tokens = 1000;
    int rows = 1;
    int stages = kDefaultStages;
    std::string bcast = "host"; // fine is REFUSED on gfx1030 (CPU->GPU invisible)
    std::string mode = "eager";
    std::string csv;
    std::string faults;
    int active_ranks = 5;
};

void hsa_check(hsa_status_t s, const char* what) {
    if (s != HSA_STATUS_SUCCESS) {
        std::fprintf(stderr, "HSA error %d at %s\n", int(s), what);
        std::exit(1);
    }
}

// hsa_amd_pointer_info for the evidence trail (05 §2.5): type, owner, bases, flags.
void print_pointer_info(const char* tag, const void* p) {
    hsa_amd_pointer_info_t info{};
    info.size = sizeof(info);
    if (hsa_amd_pointer_info(const_cast<void*>(p), &info, nullptr, nullptr, nullptr) !=
        HSA_STATUS_SUCCESS) {
        std::printf("  %s: hsa_amd_pointer_info failed\n", tag);
        return;
    }
    const char* type = "?";
    switch (info.type) {
        case HSA_EXT_POINTER_TYPE_UNKNOWN: type = "unknown"; break;
        case HSA_EXT_POINTER_TYPE_HSA: type = "HSA"; break;
        case HSA_EXT_POINTER_TYPE_LOCKED: type = "locked"; break;
        case HSA_EXT_POINTER_TYPE_GRAPHICS: type = "graphics"; break;
        default: break;
    }
    std::printf("  %s: type=%s agentOwner=0x%llx hostBase=%p agentBase=0x%llx size=%zu "
                "global_flags=0x%x\n",
                tag, type, (unsigned long long)(info.agentOwner.handle), info.hostBaseAddress,
                (unsigned long long)(info.agentBaseAddress), size_t(info.sizeInBytes),
                unsigned(info.global_flags));
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (s == "--tokens") a.tokens = std::atoi(val());
        else if (s == "--rows") a.rows = std::atoi(val());
        else if (s == "--stages") a.stages = std::atoi(val());
        else if (s == "--bcast") a.bcast = val();
        else if (s == "--mode") a.mode = val();
        else if (s == "--csv") a.csv = val();
        else if (s == "--fault") a.faults = val();
        else if (s == "--active-ranks") a.active_ranks = std::atoi(val());
        else { std::fprintf(stderr, "unknown arg %s\n", s.c_str()); return 2; }
    }
    if (a.tokens < 1 || a.rows < 1 || a.stages < 1) {
        std::fprintf(stderr, "--tokens/--rows/--stages must be >= 1\n");
        return 2;
    }
    if (a.mode != "eager" && a.mode != "persistent" && a.mode != "graph") {
        std::fprintf(stderr, "unknown --mode %s\n", a.mode.c_str());
        return 2;
    }
    if (a.active_ranks < 1 || a.active_ranks > kRanks) {
        std::fprintf(stderr, "--active-ranks must be 1..%d\n", kRanks);
        return 2;
    }

    int ndev = 0;
    HIP_CHECK(hipGetDeviceCount(&ndev), "hipGetDeviceCount");
    if (ndev < kRanks) {
        std::printf("hip_tp5_relay: only %d HIP devices, need %d - SKIP\n", ndev, kRanks);
        return 77;
    }

    const int n = a.rows * 2560;
    const int stages = a.stages;
    const uint64_t epochs_total = uint64_t(a.tokens) * uint64_t(stages);
    const uint64_t inbox_stride = Inbox::bytes(a.rows);
    const uint64_t bcast_stride = Bcast::bytes(a.rows);
    const size_t bcast_bytes = size_t(bcast_stride) * kBanks * kRanks;
    const bool fault_run = !a.faults.empty();

    // ---- fine placement: REFUSED on this stack (05 §2.5: report, never fall back) ----
    if (a.bcast == "fine") {
        std::printf(
            "placement fine: REFUSED on this gfx1030 stack.\n"
            "  hipExtMallocWithFlags(hipDeviceMallocFinegrained) allocates and the CPU can\n"
            "  write+read the BAR alias (/tmp/diag_fine.cu: 42/43 readback OK), but a CPU\n"
            "  store is NOT visible to GPU polling: fine MTYPE is cacheable and the CPU write\n"
            "  does not invalidate the GPU cache.  The relay's gen publish runs CPU->GPU, so\n"
            "  the protocol cannot run on this placement.  This is the measured answer to\n"
            "  02-Q9's broadcast (A) candidate: unusable in the CPU->GPU direction here.\n");
        return 1;
    }

    // ---- inbox + credit: mapped host memory ----------------------------------------
    const size_t inbox_total = size_t(inbox_stride) * kBanks * kRanks;
    uint8_t* inbox_base = nullptr;
    HIP_CHECK(hipHostMalloc(&inbox_base, inbox_total,
                            hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent),
              "hipHostMalloc inbox");
    Credit* credits = nullptr;
    HIP_CHECK(hipHostMalloc(&credits, sizeof(Credit) * kBanks * kRanks,
                            hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent),
              "hipHostMalloc credit");
    std::memset(inbox_base, 0, inbox_total);
    std::memset(credits, 0, sizeof(Credit) * kBanks * kRanks);
    auto inbox_payload = [&](int bank, int r) {
        return reinterpret_cast<float*>(inbox_base + (size_t(bank) * kRanks + r) * inbox_stride);
    };
    auto inbox_ready = [&](int bank, int r) -> volatile uint64_t& {
        return *reinterpret_cast<volatile uint64_t*>(
            inbox_base + (size_t(bank) * kRanks + r) * inbox_stride + Inbox::flag_off(a.rows));
    };

    // ---- bcast region: the placements (05 §2.5) ------------------------------------
    uint8_t* bcast_dev[kRanks] = {};
    uint8_t* bcast_host = nullptr;
    std::string placement_note;
    if (a.bcast == "host") {
        HIP_CHECK(hipHostMalloc(&bcast_host, bcast_bytes,
                                hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent),
                  "hipHostMalloc bcast host");
        void* dp = nullptr;
        HIP_CHECK(hipHostGetDevicePointer(&dp, bcast_host, 0), "hipHostGetDevicePointer");
        for (int r = 0; r < kRanks; ++r) bcast_dev[r] = static_cast<uint8_t*>(dp);
        placement_note = "host mapped (hipHostMalloc Mapped|Portable|Coherent)";
        print_pointer_info("host", bcast_host);
    } else if (a.bcast == "uncached") {
        HIP_CHECK(hipSetDevice(0), "hipSetDevice0");
        void* p = nullptr;
        const hipError_t e = hipExtMallocWithFlags(&p, bcast_bytes, hipDeviceMallocUncached);
        if (e != hipSuccess) {
            std::printf("hip_tp5_relay: placement 'uncached' refused (%s)\n",
                        hipGetErrorString(e));
            return 1;
        }
        bcast_dev[0] = static_cast<uint8_t*>(p);
        for (int r = 1; r < kRanks; ++r) {
            HIP_CHECK(hipSetDevice(r), "hipSetDevice r");
            int can = 0;
            HIP_CHECK(hipDeviceCanAccessPeer(&can, r, 0), "hipDeviceCanAccessPeer");
            if (can) {
                const hipError_t pe = hipDeviceEnablePeerAccess(0, 0);
                if (pe != hipSuccess && pe != hipErrorAlreadyAcquired) {
                    std::fprintf(stderr, "peer access %d->0 failed: %s\n", r,
                                 hipGetErrorString(pe));
                    return 1;
                }
                bcast_dev[r] = bcast_dev[0];
            } else {
                std::printf("hip_tp5_relay: rank %d cannot peer-access the bcast region - "
                            "REFUSED\n", r);
                return 1;
            }
        }
        bcast_host = static_cast<uint8_t*>(p);
        placement_note = "hipExtMallocWithFlags(Uncached) + peer map";
        print_pointer_info("uncached", p);
    } else if (a.bcast == "hsa-coarse") {
        hsa_check(hsa_init(), "hsa_init");
        struct Ctx { hsa_agent_t gpu, cpu; int gpu_set = 0, cpu_set = 0; } ctx;
        hsa_iterate_agents(
            [](hsa_agent_t agent, void* d) {
                auto* c = static_cast<Ctx*>(d);
                hsa_device_type_t t;
                hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &t);
                if (t == HSA_DEVICE_TYPE_GPU && !c->gpu_set) { c->gpu = agent; c->gpu_set = 1; }
                if (t == HSA_DEVICE_TYPE_CPU && !c->cpu_set) { c->cpu = agent; c->cpu_set = 1; }
                return HSA_STATUS_SUCCESS;
            }, &ctx);
        if (!ctx.gpu_set || !ctx.cpu_set) {
            std::printf("hip_tp5_relay: hsa-coarse needs a GPU and CPU agent\n");
            return 1;
        }
        // the LARGEST non-fine-grained global pool of that GPU (the 16 GiB coarse pool, not
        // the small reserved one)
        struct Best { hsa_amd_memory_pool_t pool; size_t size = 0; bool found = false; } best;
        hsa_amd_agent_iterate_memory_pools(
            ctx.gpu,
            [](hsa_amd_memory_pool_t p, void* d) {
                auto* b = static_cast<Best*>(d);
                hsa_amd_memory_pool_flag_t f;
                hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &f);
                hsa_amd_segment_t seg;
                hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
                const uint32_t f_val = static_cast<uint32_t>(f);
                if (seg != HSA_AMD_SEGMENT_GLOBAL || (f_val & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED))
                    return HSA_STATUS_SUCCESS;
                size_t sz = 0;
                hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SIZE, &sz);
                if (!b->found || sz > b->size) { b->pool = p; b->size = sz; b->found = true; }
                return HSA_STATUS_SUCCESS;
            }, &best);
        if (!best.found) { std::printf("hip_tp5_relay: no coarse global pool\n"); return 1; }
        void* p = nullptr;
        hsa_check(hsa_amd_memory_pool_allocate(best.pool, bcast_bytes, 0, &p), "pool allocate");
        hsa_check(hsa_amd_agents_allow_access(1, &ctx.cpu, nullptr, p), "allow access cpu");
        std::vector<hsa_agent_t> others;
        hsa_iterate_agents(
            [](hsa_agent_t agent, void* d) {
                auto* v = static_cast<std::vector<hsa_agent_t>*>(d);
                hsa_device_type_t t;
                hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &t);
                if (t == HSA_DEVICE_TYPE_GPU) v->push_back(agent);
                return HSA_STATUS_SUCCESS;
            }, &others);
        // allow EVERY GPU agent (the owner included - re-allowing is harmless).  The KFD
        // agent iteration order does NOT match the HIP device order (02-Q10), so skipping
        // "the first" (assumed to be the owner) could skip a DIFFERENT GPU and leave two
        // ranks unable to see the region - exactly the rank-1/rank-3 consumer timeouts the
        // 1000-token matrix hit.
        hsa_check(hsa_amd_agents_allow_access(int(others.size()), others.data(), nullptr, p),
                  "allow access gpus");
        std::printf("  hsa-coarse: allowed %zu gpu agents access to the region\n",
                    others.size());
        // per-GPU visibility verification: a touch kernel on EVERY device reads the region
        // and writes back a magic.  allow_access's agent list is in KFD order, which does
        // NOT match the HIP device order (02-Q10) - a silent miss leaves that rank's
        // consumer spinning forever.  The probe REFUSES the placement when any rank fails
        // (05 §2.5: report, never fall back).
        // (moved after the per-rank streams exist: see the hsa-coarse touch below)
        for (int r = 0; r < kRanks; ++r) bcast_dev[r] = static_cast<uint8_t*>(p);
        bcast_host = static_cast<uint8_t*>(p);
        placement_note =
            "hsa coarse pool (" + std::to_string(best.size >> 30) + " GiB) + allow_access";
        print_pointer_info("hsa-coarse", p);
    } else {
        std::fprintf(stderr, "unknown --bcast %s\n", a.bcast.c_str());
        return 2;
    }
    auto bcast_flag = [&](int bank, int r) -> uint64_t& {
        return *reinterpret_cast<uint64_t*>(bcast_host + (size_t(bank) * kRanks + r) *
                                                       bcast_stride + Bcast::flag_off(a.rows));
    };

    std::printf("placement: %s (%zu B, %d banks x %d ranks)\n", placement_note.c_str(),
                bcast_bytes, kBanks, kRanks);
    std::printf("mode %s  tokens %d  rows %d  stages %d  active_ranks %d%s\n", a.mode.c_str(),
                a.tokens, a.rows, stages, a.active_ranks,
                fault_run ? ("  faults: " + a.faults).c_str() : "");

    // ---- per-rank device state ------------------------------------------------------
    std::vector<float*> x(kRanks);
    StepState* steps_host = nullptr;
    // +1 slot: the wrong-bank fault's scratch StepState (beyond the stage array)
    HIP_CHECK(hipHostMalloc(&steps_host, sizeof(StepState) * size_t(stages + 1),
                            hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent),
              "steps");
    StepState* steps_dev = nullptr;
    { void* dp = nullptr; HIP_CHECK(hipHostGetDevicePointer(&dp, steps_host, 0), "steps dev");
      steps_dev = static_cast<StepState*>(dp); }
    // GLOBAL abort: a host-mapped word every running kernel re-reads every 4096 spins.
    // The per-stage StepState.abort is read ONCE at kernel entry, so a drain AFTER a kernel
    // started spinning could not reach it - the wrong-bank run drained 63 s late.  This word
    // is live for the kernel's whole lifetime.
    unsigned* g_abort_host = nullptr;
    HIP_CHECK(hipHostMalloc(&g_abort_host, sizeof(unsigned),
                            hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent),
              "g_abort");
    *g_abort_host = 0;
    volatile unsigned* g_abort_dev = nullptr;
    { void* dp = nullptr; HIP_CHECK(hipHostGetDevicePointer(&dp, g_abort_host, 0), "g_abort dev");
      g_abort_dev = static_cast<volatile unsigned*>(dp); }
    std::vector<unsigned long long*> err(kRanks);
    std::vector<unsigned int*> pcounters(kRanks), ccounters(kRanks);
    std::vector<unsigned int*> pbar(kRanks), cbar(kRanks);
    std::vector<StageTrace*> trace(kRanks);
    std::vector<uint64_t*> clkbuf(kRanks);
    std::vector<uint8_t*> inbox_dev(kRanks);
    std::vector<Credit*> credit_dev(kRanks);
    std::vector<hipStream_t> streams(kRanks), cstreams(kRanks);
    std::vector<hipEvent_t> cdone(kRanks);
    double ns_per_tick[kRanks] = {};
    std::vector<float> x0(n);
    for (int i = 0; i < n; ++i) x0[i] = 0.125f * float(i % 17) - 1.0f;

    // fault injection point: LAST stage of token 1.  A late-stage fault survives the
    // contracting loop to the token end (an early-stage fault decays by ~0.54^95 = 1e-24);
    // the per-token oracle compares after every token, so token 1's compare catches it.
    const uint64_t f_epoch = uint64_t(2) * stages; // token 1, stage stages-1
    const bool fault_withhold = a.faults.find("withhold") != std::string::npos;
    const bool fault_partial = a.faults.find("partial") != std::string::npos;
    const bool fault_truncate = a.faults.find("truncate") != std::string::npos;
    const bool fault_bank = a.faults.find("wrong-bank") != std::string::npos;

    for (int r = 0; r < kRanks; ++r) {
        HIP_CHECK(hipSetDevice(r), "dev loop");
        HIP_CHECK(hipMalloc(&x[r], n * 4), "hipMalloc x");
        HIP_CHECK(hipMemcpy(x[r], x0.data(), n * 4, hipMemcpyHostToDevice), "init x");
        HIP_CHECK(hipMalloc(&err[r], 8), "hipMalloc err");
        HIP_CHECK(hipMemset(err[r], 0, 8), "memset err");
        // per-epoch counter/barrier slots: each used exactly once, never cleared
        HIP_CHECK(hipMalloc(&pcounters[r], 4 * (epochs_total + 2)), "pcounters");
        HIP_CHECK(hipMalloc(&ccounters[r], 4 * (epochs_total + 2)), "ccounters");
        HIP_CHECK(hipMemset(pcounters[r], 0, 4 * (epochs_total + 2)), "memset pc");
        HIP_CHECK(hipMemset(ccounters[r], 0, 4 * (epochs_total + 2)), "memset cc");
        HIP_CHECK(hipMalloc(&pbar[r], 4 * (epochs_total + 2)), "pbar");
        HIP_CHECK(hipMalloc(&cbar[r], 4 * (epochs_total + 2)), "cbar");
        HIP_CHECK(hipMemset(pbar[r], 0, 4 * (epochs_total + 2)), "memset pbar");
        HIP_CHECK(hipMemset(cbar[r], 0, 4 * (epochs_total + 2)), "memset cbar");
        HIP_CHECK(hipMalloc(&trace[r], sizeof(StageTrace) * (epochs_total + 2)), "trace");
        HIP_CHECK(hipMemset(trace[r], 0, sizeof(StageTrace) * (epochs_total + 2)),
                  "memset trace");
        HIP_CHECK(hipMalloc(&clkbuf[r], 8), "clkbuf");
        void* dp = nullptr;
        HIP_CHECK(hipHostGetDevicePointer(&dp, inbox_base, 0), "inbox dev");
        inbox_dev[r] = static_cast<uint8_t*>(dp);
        HIP_CHECK(hipHostGetDevicePointer(&dp, credits, 0), "credit dev");
        credit_dev[r] = static_cast<Credit*>(dp);
        HIP_CHECK(hipStreamCreate(&streams[r]), "stream");
        HIP_CHECK(hipStreamCreate(&cstreams[r]), "cstream");
        HIP_CHECK(hipEventCreate(&cdone[r]), "event");
    }

    // ---- placement visibility touch (05 §2.5 evidence + refusal) ----------------------
    // The touch runs the protocol's own shape: the kernel WARM-READS the slot (populating
    // the GPU's L2 line, exactly like a consumer's first spin read of gen), reports the
    // warm value through a host-mapped word, the host waits for that report and THEN
    // rewrites the slot, and the kernel must observe the change.  A one-shot read is not
    // enough - it misses to memory and looks "visible" even when every later re-read serves
    // the STALE L2 line (gfx1030: hipDeviceMallocFinegrained AND the hsa coarse pool -
    // measured with /tmp/diag_race.cu: stage 0 passes, stage 1 spins forever).  A placement
    // that fails here cannot carry the relay's CPU->GPU gen publishes: REFUSED, no
    // silent fallback.
    {
        // host-mapped report word: seen[0] = warm value, seen[1] = final served value
        volatile uint64_t* seen = nullptr;
        HIP_CHECK(hipHostMalloc(const_cast<uint64_t**>(&seen), 16,
                                hipHostMallocMapped | hipHostMallocPortable |
                                    hipHostMallocCoherent),
                  "touch seen");
        volatile uint64_t* seen_dev = nullptr;
        { void* dp = nullptr; HIP_CHECK(hipHostGetDevicePointer(&dp, const_cast<uint64_t*>(seen), 0), "touch seen dev");
          seen_dev = static_cast<volatile uint64_t*>(dp); }
        volatile uint64_t* slot = reinterpret_cast<volatile uint64_t*>(
            bcast_host + Bcast::flag_off(a.rows)); // bank 0, rank 0's flag word (CPU alias)
        int blind = -1;
        for (int r = 0; r < kRanks && blind < 0; ++r) {
            HIP_CHECK(hipSetDevice(r), "touch setdev");
            *slot = uint64_t(0x1111 + r);   // this rank's warm value
            _mm_sfence();
            seen[0] = 0;
            seen[1] = 0;
            std::atomic_thread_fence(std::memory_order_release);
            touch_region(reinterpret_cast<const volatile uint64_t*>(
                             bcast_dev[r] + Bcast::flag_off(a.rows)),
                         seen_dev, cstreams[r]);
            // wait for the kernel's warm report (bounded)
            const auto dl = clk::now() + chrono::milliseconds(2000);
            while (seen[0] == 0) {
                if (clk::now() > dl) { blind = r; break; }
                std::this_thread::yield();
            }
            if (blind >= 0) break;
            // the host's rewrite - THE RELAY'S OWN gen PUBLISH DIRECTION
            *slot = 0x2222;
            _mm_sfence();
            HIP_CHECK(hipStreamSynchronize(cstreams[r]), "touch sync");
            const uint64_t warm = seen[0], final = seen[1];
            if (warm != static_cast<uint64_t>(0x1111 + r) || final != 0x2222ULL) {
                blind = r;
                std::printf("  placement touch: rank %d warm=0x%llx final=0x%llx (the GPU "
                            "kept serving the warm line after the host's rewrite)\n",
                            r, (unsigned long long)warm, (unsigned long long)final);
            }
        }
        *slot = 0;
        _mm_sfence();
        if (blind >= 0) {
            std::printf(
                "hip_tp5_relay: REFUSED - placement '%s' is GPU-cached: the host's rewrite "
                "never reaches a spinning consumer (rank %d).  Measured on gfx1030/ROCm "
                "7.15.26333: hipDeviceMallocFinegrained and the hsa coarse pool both fail "
                "this; only uncached VRAM and host-mapped memory carry the relay's CPU->GPU "
                "publishes.  Evidence: /tmp/diag_race.cu (stage-0 pass, stage-1 forever).\n",
                a.bcast.c_str(), blind);
            (void) hipHostFree(const_cast<uint64_t*>(seen));
            return 1;
        }
        (void) hipHostFree(const_cast<uint64_t*>(seen));
        std::printf("  placement touch: warm+rewrite visible on all %d ranks\n", kRanks);
    }

    // wall_clock64 calibration: two kernel reads 100 ms apart (per rank), cross-checked
    // against hipDeviceAttributeWallClockRate (kHz)
    {
        std::vector<uint64_t> t0(kRanks, 0), t1(kRanks, 0);
        for (int r = 0; r < kRanks; ++r) {
            HIP_CHECK(hipSetDevice(r), "clk setdev");
            read_wall_clock(clkbuf[r], streams[r]);
            HIP_CHECK(hipStreamSynchronize(streams[r]), "clk sync");
            HIP_CHECK(hipMemcpy(&t0[r], clkbuf[r], 8, hipMemcpyDeviceToHost), "clk copy");
        }
        std::this_thread::sleep_for(chrono::milliseconds(100));
        for (int r = 0; r < kRanks; ++r) {
            HIP_CHECK(hipSetDevice(r), "clk setdev");
            read_wall_clock(clkbuf[r], streams[r]);
            HIP_CHECK(hipStreamSynchronize(streams[r]), "clk sync");
            HIP_CHECK(hipMemcpy(&t1[r], clkbuf[r], 8, hipMemcpyDeviceToHost), "clk copy");
        }
        std::printf("wall_clock64 calibration (ns/tick):");
        for (int r = 0; r < kRanks; ++r) {
            int khz = 0;
            HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, r),
                      "wallclock rate");
            if (t1[r] > t0[r]) ns_per_tick[r] = 1e8 / double(t1[r] - t0[r]);
            std::printf(" r%d=%.4f(attr %.4f)", r, ns_per_tick[r], 1e6 / khz);
        }
        std::printf("\n");
    }

    // ---- graph mode (05 §3): one graph per rank, epochs read from steps_dev ----------
    // Captured on ONE stream per rank in p0, c0, p1, c1, ... order: the consumer's gen
    // spin provides the cross-rank sync, and the in-place x update is safe because
    // p_{s+1} follows c_s in stream order (a graph edge).  No hipEventRecord inside the
    // capture (05 §3: no cross-device event dependencies in the graph).
    std::vector<hipGraph_t> graphs(kRanks, nullptr);
    std::vector<hipGraphExec_t> gexecs(kRanks, nullptr);
    // ---- the CPU relay thread (05 §2.3) ----------------------------------------------
    struct RelayStats {
        std::atomic<uint64_t> token{0};
        std::atomic<int> failures{0};
        std::atomic<int> abort_flag{0};
        std::vector<double> cpu_sum_us, ready_first_us, ready_all_us;
    } stats;

    const auto relay_loop = [&](std::atomic<uint64_t>* served_upto) {
        std::vector<float> sums(n);
        const auto deadline = [&](int m) { return clk::now() + chrono::milliseconds(m); };
        for (uint64_t t = 0; t < uint64_t(a.tokens) && !stats.abort_flag.load(); ++t) {
            for (int s = 0; s < stages; ++s) {
                if (served_upto && uint64_t(s) + 1 <= served_upto->load()) {
                    // graph capture dry-run: the relay already served this epoch
                    continue;
                }
                const uint64_t epoch = t * uint64_t(stages) + s + 1;
                const int bank = int((epoch - 1) & 1);
                // 1. bank reuse guard: all five readers of the PREVIOUS use of this bank
                // posted GPU credits (05 §2.3)
                if (epoch > kBanks) {
                    const uint64_t prev = epoch - kBanks;
                    const auto dl = deadline(kTimeoutMs);
                    while (!stats.abort_flag.load()) {
                        bool all = true;
                        for (int r = 0; r < kRanks; ++r)
                            if (credits[bank * kRanks + r].consumed_epoch < prev) {
                                all = false;
                                break;
                            }
                        if (all) break;
                        if (clk::now() > dl) {
                            stats.failures++;
                            stats.abort_flag = 1;
                            std::fprintf(stderr, "relay: credit timeout epoch %llu\n",
                                         (unsigned long long)epoch);
                            return;
                        }
                        std::this_thread::yield();
                    }
                    if (stats.abort_flag.load()) return;
                }
                // 2. bounded wait for five ready epochs
                const auto t0 = clk::now();
                const auto dl = deadline(kTimeoutMs);
                int ready_seen = 0;
                while (true) {
                    ready_seen = 0;
                    for (int r = 0; r < kRanks; ++r)
                        if (inbox_ready(bank, r) >= epoch) ++ready_seen;
                    if (ready_seen == kRanks) break;
                    if (clk::now() > dl || stats.abort_flag.load()) {
                        stats.failures++;
                        stats.abort_flag = 1;
                        std::fprintf(stderr, "relay: ready timeout epoch %llu (%d/%d ready)\n",
                                     (unsigned long long)epoch, ready_seen, kRanks);
                        return;
                    }
                    std::this_thread::yield();
                }
                const auto t1 = clk::now();
                // 3. sum in rank order (FP32, fixed order)
                for (int i = 0; i < n; ++i) sums[i] = 0.0f;
                for (int r = 0; r < kRanks; ++r) {
                    const float* p = inbox_payload(bank, r);
                    for (int i = 0; i < n; ++i) sums[i] += p[i];
                }
                const auto t2 = clk::now();
                // 4. write the five bcast copies, fence, publish gen - except the faults
                if (fault_withhold && epoch == f_epoch) {
                    stats.failures++;
                    stats.abort_flag = 1;
                    std::fprintf(stderr, "relay: WITHHOLD at epoch %llu - gen never published\n",
                                 (unsigned long long)epoch);
                    return; // consumers time out; the host drains them via abort
                }
                const int gen_count = (fault_partial && epoch == f_epoch) ? 4 : kRanks;
                for (int r = 0; r < kRanks; ++r) {
                    uint8_t* bc = bcast_host + (size_t(bank) * kRanks + r) * bcast_stride;
                    std::memcpy(bc, sums.data(), n * 4);
                }
                std::atomic_thread_fence(std::memory_order_release);
                // 05-2.3: `_mm_sfence() then write gen` - a VOLATILE store, not an atomic.
                // An xchg/LOCK'd store to the hsa-coarse VRAM mapping is NOT visible to the
                // GPUs (measured: a plain-store touch reads back on every rank; the atomic
                // store never arrives).  The fence orders the payload stores before the
                // flag store; the flag itself only needs to land.
                for (int r = 0; r < gen_count; ++r) {
                    volatile uint64_t* g = &bcast_flag(bank, r);
                    *g = epoch;
                }
                _mm_sfence();
                if (gen_count < kRanks) {
                    stats.failures++;
                    stats.abort_flag = 1;
                    std::fprintf(stderr, "relay: PARTIAL at epoch %llu - rank 4's gen stale\n",
                                 (unsigned long long)epoch);
                    return;
                }
                const auto t3 = clk::now();
                if (t >= 20) {
                    stats.ready_first_us.push_back(us(t1 - t0));
                    stats.cpu_sum_us.push_back(us(t2 - t1));
                    stats.ready_all_us.push_back(us(t3 - t0));
                }
                stats.token = t + 1;
                if (served_upto) served_upto->store(epoch);
            }
        }
    };

    // ---- graph capture with a live relay (the capture runs a REAL token 0) -----------
    // The capture executes the kernels once; the relay serves those epochs; then the
    // captured graph is instantiated and re-used for tokens 1..N with fresh epochs read
    // from steps_dev.  The counters/barriers are per-epoch, so the dry-run's epochs (1..96)
    // do not collide with the replays (the graph reads epoch from steps_dev each launch).
    std::thread relay([&]() {
        std::atomic<uint64_t> dummy{0};
        relay_loop(nullptr);
    });

    const auto t_start = clk::now();
    std::vector<double> token_ms;
    uint64_t launch_count = 0;

    // fill token 0's StepStates BEFORE the capture (the capture runs the kernels for real)
    for (int s = 0; s < stages; ++s) {
        const uint64_t epoch = uint64_t(s) + 1;
        steps_host[s] = StepState{epoch, uint32_t((epoch - 1) & 1), 0};
    }
    std::atomic_thread_fence(std::memory_order_release);

    if (a.mode == "graph") {
        // Stream capture RECORDS the launch sequence without executing it (a HIP capture
        // never runs the kernels - the "live token 0" idea was wrong and timed out the
        // relay).  The kernels' StepState/counter/trace pointers are captured as POINTER
        // arguments; the VALUES (epoch/bank) are read from steps_dev at launch time, so the
        // same graph serves every token (05 §3: never a captured constant).
        // The token loop below starts at t=0 and fills steps BEFORE each hipGraphLaunch.
        for (int r = 0; r < kRanks; ++r) {
            HIP_CHECK(hipSetDevice(r), "graph capture setdev");
            HIP_CHECK(hipStreamBeginCapture(streams[r], hipStreamCaptureModeGlobal),
                      "begin cap");
            for (int s = 0; s < stages; ++s) {
                produce_stage(x[r], inbox_dev[r], inbox_stride, steps_dev + s, r, a.rows,
                              a.active_ranks, g_abort_dev, streams[r], pcounters[r],
                              trace[r]);
                consume_stage(bcast_dev[r], bcast_stride, x[r], steps_dev + s, credit_dev[r],
                              r, a.rows, g_abort_dev, streams[r], err[r], ccounters[r],
                              trace[r]);
            }
            HIP_CHECK(hipStreamEndCapture(streams[r], &graphs[r]), "end cap");
            HIP_CHECK(hipGraphInstantiate(&gexecs[r], graphs[r], nullptr, nullptr, 0),
                      "instantiate");
            size_t nn = 0;
            HIP_CHECK(hipGraphGetNodes(graphs[r], nullptr, &nn), "graph nodes");
            std::printf("rank %d graph: %zu nodes (2 x %d kernels), 1 launch/token\n", r, nn,
                        stages);
        }
    }

    // ---- the launcher loop ------------------------------------------------------------
    for (uint64_t t = 0; t < uint64_t(a.tokens); ++t) {
        const auto tt0 = clk::now();
        for (int s = 0; s < stages; ++s) {
            const uint64_t epoch = t * uint64_t(stages) + uint64_t(s) + 1;
            steps_host[s] = StepState{epoch, uint32_t((epoch - 1) & 1), 0};
        }
        std::atomic_thread_fence(std::memory_order_release);

        // truncate: at (t=1, last stage, rank 3) pre-publish ready and SKIP the producer
        // launch: the relay sums the STALE payload (a real "ready before data" error), and
        // the per-token oracle catches it at token 1's compare.
        const uint64_t skip_epoch = (fault_truncate && t == 1) ? f_epoch : 0;

        if (a.mode == "eager" || a.mode == "graph") {
            if (a.mode == "graph") {
                for (int r = 0; r < kRanks; ++r) {
                    HIP_CHECK(hipSetDevice(r), "graph launch setdev");
                    if (fault_truncate && t == 1) {
                        // pre-publish ready AFTER the producer nodes ran is impossible from
                        // the host mid-graph; instead publish BEFORE the launch: the
                        // graph's producer will overwrite the flag with the same epoch
                        // value (it publishes epoch too), so the relay sees ready on time
                        // but the PAYLOAD is still the stale bank data only if the producer
                        // never ran - which it did.  So in graph mode the truncate fault is
                        // emulated by corrupting the payload AFTER the producer: not
                        // possible without a graph node.  Graph mode + truncate is
                        // UNSUPPORTED: the eager path covers it (05 §2.4 runs it there).
                    }
                    HIP_CHECK(hipGraphLaunch(gexecs[r], streams[r]), "graph launch");
                    launch_count += 1;
                }
            } else {
                for (int s = 0; s < stages; ++s) {
                    const uint64_t epoch = t * uint64_t(stages) + uint64_t(s) + 1;
                    for (int r = 0; r < kRanks; ++r) {
                        HIP_CHECK(hipSetDevice(r), "launch setdev");
                        if (epoch == skip_epoch && r == 3) {
                            inbox_ready(int((epoch - 1) & 1), 3) = epoch; // pre-publish ready
                            continue;                                     // skip the producer
                        }
                        // wrong-bank: rank 2's consumer reads bank (epoch)&1 - the OTHER
                        // bank - whose gen holds the PREVIOUS epoch: it spins, times out,
                        // and the host's ready/credit waits catch it.  Implemented by
                        // flipping the bank in the StepState the consumer reads.
                        const StepState* cstep = steps_dev + s;
                        if (fault_bank && epoch == f_epoch && r == 2) {
                            // write the flipped copy into the LAST steps slot (stage
                            // stages-1 IS the fault stage here, so use a dedicated scratch
                            // slot beyond the stage array)
                            cstep = reinterpret_cast<StepState*>(steps_host + stages);
                            *const_cast<StepState*>(cstep) = steps_host[s];
                            const_cast<StepState*>(cstep)->bank ^= 1u;
                            std::atomic_thread_fence(std::memory_order_release);
                        }
                        if (s > 0 || t > 0)
                            HIP_CHECK(hipStreamWaitEvent(streams[r], cdone[r], 0), "wait cdone");
                        produce_stage(x[r], inbox_dev[r], inbox_stride, steps_dev + s, r,
                                      a.rows, a.active_ranks, g_abort_dev, streams[r],
                                      pcounters[r], trace[r]);
                        consume_stage(bcast_dev[r], bcast_stride, x[r], cstep,
                                      credit_dev[r], r, a.rows, g_abort_dev, cstreams[r],
                                      err[r], ccounters[r], trace[r]);
                        HIP_CHECK(hipEventRecord(cdone[r], cstreams[r]), "record cdone");
                        launch_count += 2;
                    }
                }
            }
        } else { // persistent
            for (int r = 0; r < kRanks; ++r) {
                HIP_CHECK(hipSetDevice(r), "persistent setdev");
                run_token_persistent(x[r], inbox_dev[r], inbox_stride, bcast_dev[r],
                                     bcast_stride, steps_dev, stages, r, a.rows,
                                     a.active_ranks, credit_dev[r], err[r], pcounters[r],
                                     ccounters[r], pbar[r], cbar[r], g_abort_dev, trace[r],
                                     streams[r]);
                launch_count += 1;
            }
        }

        // wait for the token's last credit (bounded)
        const uint64_t last_epoch = (t + 1) * uint64_t(stages);
        const int last_bank = int((last_epoch - 1) & 1);
        const auto dl = clk::now() + chrono::milliseconds(kTimeoutMs * 3);
        while (true) {
            bool done = true;
            for (int r = 0; r < kRanks; ++r)
                if (credits[last_bank * kRanks + r].consumed_epoch < last_epoch) {
                    done = false;
                    break;
                }
            if (done) break;
            if (stats.abort_flag.load() || clk::now() > dl) {
                std::string stuck;
                for (int r = 0; r < kRanks; ++r)
                    if (credits[last_bank * kRanks + r].consumed_epoch < last_epoch)
                        stuck += std::to_string(r) + " ";
                if (!stuck.empty())
                    std::fprintf(stderr,
                                 "launcher: token %llu credit wait ended with rank(s) [%s] "
                                 "stuck at epoch %llu\n",
                                 (unsigned long long)t, stuck.c_str(),
                                 (unsigned long long)last_epoch);
                break;
            }
            std::this_thread::yield();
        }
        token_ms.push_back(ms(clk::now() - tt0));
        if (stats.abort_flag.load()) break;
    }
    const double t_total = ms(clk::now() - t_start);

    // ---- failure drain: broadcast abort, sync, NEVER replay (05 §2.4) ----------------
    // g_abort first: it reaches kernels that are ALREADY spinning (the per-step abort only
    // helps kernels not yet launched)
    __atomic_store_n(g_abort_host, 1u, __ATOMIC_RELEASE);
    for (int s = 0; s < stages; ++s) steps_host[s].abort = 1;
    std::atomic_thread_fence(std::memory_order_release);
    for (int r = 0; r < kRanks; ++r) {
        HIP_CHECK(hipSetDevice(r), "drain setdev");
        HIP_CHECK(hipStreamSynchronize(streams[r]), "drain sync");
        if (a.mode == "eager") HIP_CHECK(hipStreamSynchronize(cstreams[r]), "drain sync c");
    }
    relay.join();
    // post-drain diagnostics: the consumer's GPU-side err flags (1 = spin timeout) and
    // the last stage trace per rank - printed BEFORE teardown frees them
    {
        unsigned long long e[kRanks] = {};
        for (int r = 0; r < kRanks; ++r) {
            HIP_CHECK(hipSetDevice(r), "diag setdev");
            if (hipMemcpy(&e[r], err[r], 8, hipMemcpyDeviceToHost) != hipSuccess) {}
        }
        for (int r = 0; r < kRanks; ++r) {
            // print the LAST COMPLETED stage's trace (done_ counts served stages; the stuck
            // stage's trace was never written, so done_-1 is the interesting slot only when
            // the loop completed - print both the last-served and the first stage)
            const uint64_t done_ = stats.token.load() > 0
                                       ? (stats.token.load() - 1) * uint64_t(stages) + 1
                                       : 0;   // first stage of the last served token
            const StageTrace& tr = trace[r][done_ > 0 ? done_ - 1 : 0];
            std::printf("  rank %d: err=%llu last_trace(prod_start=%llu prod_end=%llu "
                        "gen_seen=%llu cons_end=%s)\n",
                        r, e[r], (unsigned long long)tr.prod_start,
                        (unsigned long long)tr.prod_end, (unsigned long long)tr.gen_seen,
                        tr.cons_end == kTraceTimeout ? "TIMEOUT" : "ok");
        }
    }

    // ---- oracle (05 §2.4): recompute x on the CPU EVERY token ------------------------
    // sum_i = Σ_r g(x_s[r][i], r, s, i) over the ranks' OWN x (they diverge per rank after
    // stage 1); x_{s+1}[r][i] = h(sum, r, s, i).  The GPU contracts a*b+c into fma; the
    // host does not; 96 stages compound to ~1e-3 relative - far below any PROTOCOL error
    // (wrong bank / truncated / missing payloads move values by O(1)), so tight tolerance.
    int oracle_fail = 0;
    {
        std::vector<std::vector<float>> xr(kRanks, x0);
        std::vector<float> sum(n);
        std::vector<float> got(n);
        for (uint64_t t = 0; t < stats.token.load() && !oracle_fail; ++t) {
            for (int s = 0; s < stages; ++s) {
                const int stage = int((t * uint64_t(stages) + s) % 96);
                for (int i = 0; i < n; ++i) sum[i] = 0.0f;
                for (int r = 0; r < kRanks; ++r)
                    for (int i = 0; i < n; ++i)
                        sum[i] += ref_g(xr[r][i], r, stage, i, a.active_ranks);
                for (int r = 0; r < kRanks; ++r)
                    for (int i = 0; i < n; ++i)
                        xr[r][i] = ref_h(sum[i], r, stage, i);
            }
            for (int r = 0; r < kRanks && !oracle_fail; ++r) {
                HIP_CHECK(hipSetDevice(r), "oracle setdev");
                HIP_CHECK(hipMemcpy(got.data(), x[r], n * 4, hipMemcpyDeviceToHost),
                          "oracle copy");
                for (int i = 0; i < n; ++i) {
                    const double d = std::fabs(double(got[i]) - double(xr[r][i]));
                    const double tol = 1e-3 + 1e-3 * std::fabs(double(xr[r][i]));
                    if (d > tol) {
                        std::fprintf(stderr,
                                     "oracle: token %llu rank %d elem %d got %.7f want %.7f "
                                     "(%.2e > tol %.2e)\n",
                                     (unsigned long long)(t + 1), r, i, double(got[i]),
                                     double(xr[r][i]), d, tol);
                        oracle_fail = 1;
                        break;
                    }
                }
            }
            if (oracle_fail) break;
        }
    }

    // ---- report ------------------------------------------------------------------------
    const size_t toks_done = stats.token.load();
    auto median = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        const size_t m = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + m, v.end());
        return v[m];
    };
    auto p90f = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        const size_t m = size_t(0.9 * (v.size() - 1));
        std::nth_element(v.begin(), v.begin() + m, v.end());
        return v[m];
    };
    std::printf("tokens done %zu/%d  failures %d  oracle_fail %d  total %.1f ms  "
                "launches %llu (%.2f/token/rank)\n",
                toks_done, a.tokens, stats.failures.load(), oracle_fail, t_total,
                (unsigned long long)launch_count, double(launch_count) / 5.0 /
                    double(a.mode == "graph" ? a.tokens - 1 : a.tokens));
    if (!stats.ready_first_us.empty()) {
        std::printf("ready_first  median %.1f us  p90 %.1f us\n", median(stats.ready_first_us),
                    p90f(stats.ready_first_us));
        std::printf("cpu_sum     median %.1f us  p90 %.1f us\n", median(stats.cpu_sum_us),
                    p90f(stats.cpu_sum_us));
        std::printf("ready_all   median %.1f us  p90 %.1f us\n", median(stats.ready_all_us),
                    p90f(stats.ready_all_us));
    }
    if (!token_ms.empty())
        std::printf("token       median %.3f ms  p90 %.3f ms\n", median(token_ms),
                    p90f(token_ms));

    // ---- per-stage CSV (05 §2.6) ------------------------------------------------------
    if (!a.csv.empty()) {
        FILE* f = std::fopen(a.csv.c_str(), "w");
        if (f) {
            std::fprintf(f, "token,stage,rank,prod_us,publish_to_cpu_seen_us,cpu_sum_us,"
                            "gen_to_gpu_seen_us,consumer_us,stage_total_us\n");
            // GPU-side intervals from StageTrace (ns/tick converted); host-side per-stage
            // intervals only exist for stages the relay timed (the full host grid would
            // need the relay's own per-stage timestamps - the medians above carry those).
            for (uint64_t t = 0; t < uint64_t(a.tokens); ++t) {
                for (int s = 0; s < stages; ++s) {
                    const uint64_t e = t * uint64_t(stages) + s;
                    for (int r = 0; r < kRanks; ++r) {
                        const StageTrace& tr = trace[r][e];
                        if (tr.prod_start == 0) continue;
                        const double prod =
                            double(tr.prod_end - tr.prod_start) * ns_per_tick[r] / 1000.0;
                        const double gen2cons =
                            tr.cons_end != kTraceTimeout
                                ? double(tr.cons_end - tr.gen_seen) * ns_per_tick[r] / 1000.0
                                : -1.0;
                        const double total =
                            tr.cons_end != kTraceTimeout
                                ? double(tr.cons_end - tr.prod_start) * ns_per_tick[r] / 1000.0
                                : -1.0;
                        std::fprintf(f, "%llu,%d,%d,%.2f,,%.2f,,%.2f,%.2f\n",
                                     (unsigned long long)t, s, r, prod, gen2cons, gen2cons,
                                     total);
                    }
                }
            }
            std::fclose(f);
            std::printf("csv written %s\n", a.csv.c_str());
        }
    }

    // ---- teardown -----------------------------------------------------------------------
    for (int r = 0; r < kRanks; ++r) {
        HIP_CHECK(hipSetDevice(r), "teardown setdev");
        if (gexecs[r]) {
            HIP_CHECK(hipGraphExecDestroy(gexecs[r]), "exec destroy");
            gexecs[r] = nullptr;
        }
        if (graphs[r]) {
            HIP_CHECK(hipGraphDestroy(graphs[r]), "graph destroy");
            graphs[r] = nullptr;
        }
        (void) hipFree(x[r]);
        hipFree(err[r]);
        hipFree(pcounters[r]);
        hipFree(ccounters[r]);
        hipFree(pbar[r]);
        hipFree(cbar[r]);
        hipFree(trace[r]);
        hipFree(clkbuf[r]);
        hipStreamDestroy(streams[r]);
        if (a.mode == "eager") hipStreamDestroy(cstreams[r]);
        hipEventDestroy(cdone[r]);
    }
    hipHostFree(steps_host);
    hipHostFree(g_abort_host);
    if (a.bcast == "host") hipHostFree(bcast_host);
    else if (a.bcast == "uncached") hipFree(bcast_dev[0]);
    else if (a.bcast == "hsa-coarse") hsa_amd_memory_pool_free(bcast_host);
    hipHostFree(inbox_base);
    hipHostFree(credits);

    if (fault_run) {
        // 05 §2.4: a fault run PASSES when the fault is DETECTED
        //   withhold -> relay timeout + consumers drained (failures > 0)
        //   partial   -> rank 4's consumer spin timeout / relay credit timeout
        //   truncate  -> per-token oracle mismatch (stale payload summed)
        //   wrong-bank-> consumer timeout (stale gen) / oracle mismatch
        const bool detected = stats.failures.load() > 0 || oracle_fail > 0 ||
                              toks_done < size_t(a.tokens);
        std::printf(detected ? "fault detected: PASS\n" : "fault NOT detected: FAIL\n");
        return detected ? 0 : 1;
    }
    const bool ok = stats.failures.load() == 0 && oracle_fail == 0;
    std::printf(ok ? "hip_tp5_relay: PASS\n" : "hip_tp5_relay: FAIL\n");
    return ok ? 0 : 1;
}
