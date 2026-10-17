// include/strata/kernels/tp5_relay.hpp - the TP5 RELAY: 5-rank mailbox + CPU sum + broadcast.
//
// Phase 2 of the TP5-ROCm plan (tp5-rocm/05-phase2-relay.md).  One RELAY stage is:
//
//   GPU producer  (per rank) : payload -> inbox[bank][rank], then system-scope ready epoch
//   CPU relay thread          : wait 5 ready, sum in rank order, write 5 bcast regions, gen epoch
//   GPU consumer (per rank)  : wait gen epoch, read sum, transform into x_{s+1}, write credit
//
// The protocol rules that make it a RELAY and not a racy broadcast (05 §2.3, from the Vulkan
// implementation's own contract):
//   * bank = (epoch - 1) & 1, epoch = 1.., one independent slot per (sender, bank)
//   * the CPU may only REUSE a bank after every reader of the previous use of that bank has
//     posted a GPU-written credit (consumed epoch); host counters do not count
//   * ready only means the flag word landed; payload integrity is the oracle's job
//   * every wait is bounded (host 2000 ms, GPU spin max) and ABORT-AWARE, and failure DRAINS
//     - never replays, never deadlocks (a thread-0 early return before __syncthreads would
//     hang the block; every bail-out here is block-uniform via a shared flag)
//
// All flag words live on their own 128-byte line, away from payloads (05 §2.2).
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_TP5_HD __host__ __device__
#else
#define STRATA_TP5_HD
#endif

namespace strata::kernels::tp5 {

inline constexpr int kRanks = 5;
inline constexpr int kBanks = 2;
inline constexpr int kLine = 128; // GPU cache line; flags never share a line with payloads

/// The mailbox, one per rank per bank.  VARIABLE SIZE: the payload area is `rows * 2560`
/// floats (the caller computes the stride); only the flag layout is the struct's.
/// Indexing a fixed 2560 array with rows > 1 was a real overflow (SIGSEGV at rows=5).
struct Inbox {
    /// the flag word's offset from the region start: payload (rows*2560 floats) rounded up
    /// to a 128-byte line.  __host__ __device__ so the kernels compute it too.
    static STRATA_TP5_HD uint64_t flag_off(int rows) {
        return ((uint64_t(rows) * 2560 * 4 + kLine - 1) / kLine) * kLine;
    }
    static STRATA_TP5_HD uint64_t bytes(int rows) { return flag_off(rows) + kLine; }
};

/// The broadcast region, one per rank per bank: the same variable-size layout as Inbox.
struct Bcast {
    static STRATA_TP5_HD uint64_t flag_off(int rows) {
        return ((uint64_t(rows) * 2560 * 4 + kLine - 1) / kLine) * kLine;
    }
    static STRATA_TP5_HD uint64_t bytes(int rows) { return flag_off(rows) + kLine; }
};

/// GPU-written credit: the consumer of (bank, rank) finished with epoch `consumed_epoch`.
struct alignas(kLine) Credit {
    uint64_t consumed_epoch;
    uint64_t pad[15];
};
static_assert(sizeof(Credit) % kLine == 0);

/// Device-side step state the kernels read (NEVER a captured constant: 05 §3 - a constant
/// would replay the first token's epoch forever).  The host writes epoch/bank BEFORE the
/// launch and only ever writes `abort` (during failure drain) while a kernel may run.
struct StepState {
    uint64_t epoch;      ///< t * stages + s + 1
    uint32_t bank;       ///< (epoch - 1) & 1
    uint32_t abort;      ///< set by the host on failure; every kernel checks it (uniformly)
};

/// Per-stage GPU timestamps in wall_clock64 ticks (convert with the host-calibrated
/// ns-per-tick).  Indexed by epoch-1; one array per rank.
///   prod_start / prod_end : block 0 start / LAST arriving block's publish (producer)
///   gen_seen              : a consumer block's thread 0 first observing gen == epoch
///   cons_end              : LAST arriving block's credit publish; kTraceTimeout = spin
///                           timed out (the stage never completed)
struct StageTrace {
    uint64_t prod_start;
    uint64_t prod_end;
    uint64_t gen_seen;
    uint64_t cons_end;
};
inline constexpr uint64_t kTraceTimeout = ~0ull;

/// Eager producer (one launch per rank per stage): payload[i] = g(x[i], rank, stage, i) for
/// rank < active_ranks, else 0 (05 §2.7's ladder: K ranks publish real payloads, the rest
/// publish zeros and still participate).  The LAST arriving workgroup fences system-scope
/// and stores ready_epoch - only the last workgroup publishes; `__syncthreads()` does not
/// cross workgroups, which is why the old /tmp probes measured a phantom speed (02 Q8).
/// `counters` is `unsigned int[epoch + 2]`: each slot is used exactly once (epochs are
/// unique), so no clearing - the memset-before-launch race and the graph-mode memset node
/// both disappear.
void produce_stage(const float* x, void* inbox_base, uint64_t inbox_stride,
                   const StepState* step, int rank, int rows, int active_ranks,
                   const volatile unsigned* g_abort, void* stream, unsigned int* counters,
                   StageTrace* trace);

/// Eager consumer: bounded spin on bcast[bank][rank].gen == step->epoch (spin max 1e8,
/// re-checking abort every 4096 spins so a failed run drains in microseconds), then read the
/// whole sum[], write x_next[i] = h(sum[i], ...), and post the credit from the last arriving
/// block (system fence).  On timeout: err = 1 and the kTraceTimeout trace sentinel; the
/// host's bounded wait is the backstop.
void consume_stage(void* bcast_base, uint64_t bcast_stride, float* x_next,
                   const StepState* step, Credit* credit, int rank, int rows,
                   const volatile unsigned* g_abort, void* stream, unsigned long long* err,
                   unsigned int* counters, StageTrace* trace);

/// Persistent mode (05 §2.6): ONE kernel per rank per token walking all `nstages` stages
/// in place - producer half, then consumer half, with grid-wide barriers between the halves
/// so stage s+1's producer reads x only after EVERY block's stage-s consumer wrote it.
/// x is OVERWRITTEN in place (the ready -> CPU -> gen round trip orders every producer's
/// reads before every consumer's write of the same stage).  `pbar`/`cbar` are the
/// producer-side/consumer-side barrier slot arrays, each `unsigned int[epochs + 2]`,
/// indexed by epoch-1, each slot used exactly once.
void run_token_persistent(float* x, void* inbox_base, uint64_t inbox_stride, void* bcast_base,
                          uint64_t bcast_stride, const StepState* steps, int nstages, int rank,
                          int rows, int active_ranks, Credit* credit,
                          unsigned long long* err, unsigned int* pcounters,
                          unsigned int* ccounters, unsigned int* pbar, unsigned int* cbar,
                          const volatile unsigned* g_abort, StageTrace* trace, void* stream);

/// Reads wall_clock64() into out[0] - the host calibrates ns-per-tick with two reads a
/// known interval apart (and cross-checks hipDeviceAttributeWallClockRate).
void read_wall_clock(uint64_t* out, void* stream);

/// Placement visibility touch in the protocol's own shape: a warm read (populates the GPU
/// L2 line) followed by a bounded spin waiting for the host to CHANGE the value.
/// seen[0] packs (first_value << 32) | final_value: final != first means the host's
/// rewrite reached a GPU that had ALREADY cached the line - the relay's gen publish
/// direction.  final == first (timeout) REFUSES the placement: the GPU serves the stale
/// line forever, and the consumer would spin to its death.
void touch_region(const volatile uint64_t* region, volatile uint64_t* seen, void* stream);

/// CPU reference of g and h (05 §2.4's oracle): the host recomputes x stage by stage with
/// the SAME FP32 order.  `active_ranks` mirrors the producer's zero-payload rule.
float ref_g(float x, int rank, int stage, int i, int active_ranks);
float ref_h(float sum, int rank, int stage, int i);

} // namespace strata::kernels::tp5
