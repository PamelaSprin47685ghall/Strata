// src/kernels/cuda/tp5_relay.cu - the TP5 RELAY kernels (05-phase2 §2.2-2.3).
//
// Three launch shapes share one protocol (see the header for the contract):
//   * eager: produce_stage/consume_stage per (rank, stage) - two streams per rank
//   * persistent: run_token_persistent, one kernel per rank per token
//   * graph (05 §3): the eager per-stage sequence captured into one hipGraph per rank,
//     launched once per token; epochs/banks are READ from device memory (StepState),
//     never captured constants
//
// Every publish comes from the LAST arriving workgroup only, after a system-scope fence.
// Every wait is bounded AND abort-aware: the spin helper re-reads `abort` every 4096
// iterations, so a host-declared failure drains in microseconds instead of the full 1e8
// spin budget (the withhold test's 60 s hang).  Every bail-out is block-uniform: thread 0
// decides, stores the decision in shared memory, and __syncthreads() runs for the WHOLE
// block - a thread-0 early return before a barrier would hang the block forever.
//
// Grid-wide completion counters live in `counters[epoch]` - one slot per epoch, each used
// exactly once, so nothing is ever cleared: the host memset race (block 0 initializing
// while other blocks atomicAdd) and the graph-mode memset node both disappear.
#include "strata/kernels/tp5_relay.hpp"

#include "strata/kernels/dp4a.hpp"

#include <cuda_runtime.h>

namespace strata::kernels::tp5 {

namespace {

constexpr int kThreads = 256;
constexpr unsigned long long kSpinMax = 100000000ull; // GGML_TP5_SPIN_MAX (05 §2.3)

#if !defined(__HIPCC__)
// Like gpu_stamp, CUDA uses the device-global nanosecond timer; HIP keeps its
// native wall_clock64 ticks. The relay probe calibrates either counter on host.
__device__ __forceinline__ uint64_t wall_clock64() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#endif

// Bounded, deterministic, elementwise, TRANSCENDENTAL-FREE: the closed loop must stay
// finite over 96 stages x 1000 tokens (an amplification > 1 hit inf/nan at stage ~90 and
// made the oracle compare -inf against -nan), and close to bit-exact between GPU and the
// CPU oracle (tanhf and the host's tanh differ in the last ulp; 05 §2.4's
// "same order -> bit-identical" only holds for + and *).
//   g: |x|<=1 in, 0.3*(rank+1)*x + O(1e-3) -> |g| <= 1.5
//   h: 5-rank sum <= 7.5 in, *0.12 + O(1e-3) -> |h| <= 0.91  (contracting)
//   active_ranks < 5 (the ladder): inactive ranks publish ZERO payloads.
__device__ __forceinline__ float g_of(float x, int rank, int stage, int i, int active) {
    return (rank < active) ? x * (0.3f * (rank + 1)) + stage * 1e-3f + i * 1e-6f : 0.0f;
}

__device__ __forceinline__ float h_of(float sum, int rank, int stage, int i) {
    return sum * 0.12f + rank * 1e-3f + i * 1e-7f + stage * 1e-5f;
}

// Bounded spin on a volatile u64, abort-aware: a plain load can be hoisted out of the loop
// by the compiler; a spin without the abort re-check drains the full 1e8 iterations (~15 s
// per kernel) after the host already gave up.
__device__ __forceinline__ bool spin_until(volatile const uint64_t* addr, uint64_t want,
                                           uint32_t abort, const volatile unsigned* g_abort) {
    unsigned long long spins = 0;
    while (*addr != want) {
        if ((spins++ & 4095) == 0 && (abort || *g_abort)) return false;
        if (spins > kSpinMax) return false;
        strata_spin_pause();
    }
    return true;
}

// Grid-wide barrier on a once-only slot: every block's thread 0 increments the slot; the
// LAST one sets the release bit; everyone spins on it (abort-aware).  The slot is used
// exactly once, so it is never cleared.  Block-uniform bail-out: thread 0 stores the
// decision in shared memory, then the whole block reads it after __syncthreads().
__device__ __forceinline__ bool grid_barrier(unsigned int* slot, uint32_t abort,
                                             const volatile unsigned* g_abort, int nblocks) {
    __shared__ int s_bail;
    if (threadIdx.x == 0) s_bail = 0;
    __syncthreads();
    if (threadIdx.x == 0) {
        if (atomicAdd(slot, 1u) == unsigned(nblocks - 1)) {
            // last block: publish the release (high bit)
            __threadfence();
            *reinterpret_cast<volatile unsigned*>(slot) = 0x80000000u | *slot;
        } else {
            volatile unsigned* v = reinterpret_cast<volatile unsigned*>(slot);
            unsigned long long spins = 0;
            while ((*v & 0x80000000u) == 0) {
                if ((spins++ & 4095) == 0 && (abort || *g_abort)) { s_bail = 1; break; }
                if (spins > kSpinMax) { s_bail = 1; break; }
                strata_spin_pause();
            }
        }
    }
    __syncthreads();
    return s_bail == 0;
}

// ---- eager kernels ---------------------------------------------------------------

__global__ void producer_kernel(const float* __restrict__ x, uint8_t* __restrict__ inbox_base,
                                uint64_t inbox_stride, const StepState* __restrict__ step,
                                int rank, int rows, int active,
                                [[maybe_unused]] const volatile unsigned* g_abort,
                                StageTrace* __restrict__ trace,
                                unsigned int* __restrict__ counters) {
    if (step->abort) return; // block-uniform: the whole kernel returns before any barrier
    const uint64_t epoch = step->epoch;
    const uint32_t bank = step->bank;
    uint8_t* ib = inbox_base + (uint64_t(bank) * kRanks + rank) * inbox_stride;
    float* payload = reinterpret_cast<float*>(ib);
    volatile uint64_t* ready = reinterpret_cast<volatile uint64_t*>(ib + Inbox::flag_off(rows));
    const int n = rows * 2560;
    const int stage = int((epoch - 1) % 96);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x == 0 && threadIdx.x == 0) trace[epoch - 1].prod_start = wall_clock64();
    for (int k = i; k < n; k += gridDim.x * blockDim.x)
        payload[k] = g_of(x[k], rank, stage, k, active);
    // grid-wide completion: __syncthreads first (the WHOLE block's stores are ordered
    // before thread 0's fence), then every block increments; the LAST one fences and
    // publishes.
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        const unsigned arrived = atomicAdd(&counters[epoch], 1u) + 1;
        if (arrived == gridDim.x) {
            // VOLATILE store + system fence: a plain store to mapped memory stays in the
            // GPU's L2 on RDNA until a sync (elementwise.cu doorbell_ring's lesson).
            __threadfence_system();
            *ready = epoch;
            trace[epoch - 1].prod_end = wall_clock64();
        }
    }
}

__global__ void consumer_kernel(const uint8_t* __restrict__ bcast_base,
                                uint64_t bcast_stride, float* __restrict__ x_next,
                                const StepState* __restrict__ step, Credit* __restrict__ credit,
                                int rank, int rows, unsigned long long* __restrict__ err,
                                const volatile unsigned* g_abort,
                                unsigned int* __restrict__ counters,
                                StageTrace* __restrict__ trace) {
    if (step->abort) return; // block-uniform bail before any barrier
    const uint64_t epoch = step->epoch;
    __shared__ int s_ok;
    if (threadIdx.x == 0) {
        const uint32_t bank = step->bank;
        const uint8_t* bc = bcast_base + (uint64_t(bank) * kRanks + rank) * bcast_stride;
        const volatile uint64_t* gen =
            reinterpret_cast<const volatile uint64_t*>(bc + Bcast::flag_off(rows));
        const uint64_t t = wall_clock64();
        if (spin_until(gen, epoch, step->abort, g_abort)) {
            trace[epoch - 1].gen_seen = t;   // first observation among blocks (racy, benign)
            s_ok = 1;
        } else {
            s_ok = 0;
        }
    }
    __syncthreads();
    if (!s_ok) {
        // timeout or abort: record and leave - block-uniform, no barrier follows
        if (threadIdx.x == 0) {
            atomicExch(err, 1ull); // GPU-side record; the host's bounded wait is the backstop
            const uint32_t bank = step->bank;
            const uint8_t* bc = bcast_base + (uint64_t(bank) * kRanks + rank) * bcast_stride;
            const volatile uint64_t* gen =
                reinterpret_cast<const volatile uint64_t*>(bc + Bcast::flag_off(rows));
            if (*gen != epoch) trace[epoch - 1].cons_end = kTraceTimeout;
        }
        return;
    }
    __threadfence_system();
    const uint32_t bank = step->bank;
    const uint8_t* bc = bcast_base + (uint64_t(bank) * kRanks + rank) * bcast_stride;
    const float* sum = reinterpret_cast<const float*>(bc);
    const int n = rows * 2560;
    const int stage = int((epoch - 1) % 96);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = i; k < n; k += gridDim.x * blockDim.x)
        x_next[k] = h_of(sum[k], rank, stage, k);
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        const unsigned arrived = atomicAdd(&counters[epoch], 1u) + 1;
        if (arrived == gridDim.x) {
            __threadfence_system();
            *(volatile uint64_t*)&credit[bank * kRanks + rank].consumed_epoch = epoch;
            trace[epoch - 1].cons_end = wall_clock64();
        }
    }
}

// ---- persistent kernel (05 §2.6) --------------------------------------------------

// One kernel per rank per token: for each stage, producer half -> (CPU relay in between) ->
// consumer half, in place on x.  The mailbox protocol provides the cross-rank ordering; the
// grid barriers provide the intra-rank stage ordering (stage s+1's producer reads x only
// after every block's stage-s consumer wrote it).
__global__ void persistent_kernel(float* __restrict__ x, uint8_t* inbox_base,
                                  uint64_t inbox_stride, uint8_t* bcast_base,
                                  uint64_t bcast_stride, const StepState* __restrict__ steps,
                                  int nstages, int rank, int rows, int active,
                                  Credit* __restrict__ credit,
                                  unsigned long long* __restrict__ err,
                                  unsigned int* __restrict__ pcounters,
                                  unsigned int* __restrict__ ccounters, unsigned int* pbar,
                                  unsigned int* cbar, const volatile unsigned* g_abort,
                                  StageTrace* __restrict__ trace) {
    const int n = rows * 2560;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int nblocks = int(gridDim.x);

    for (int s = 0; s < nstages; ++s) {
        const StepState st = steps[s]; // abort/epoch read once per stage
        if (st.abort) return;          // block-uniform (same steps for every block)
        const uint64_t epoch = st.epoch;
        const uint32_t bank = st.bank;
        const int stage = int((epoch - 1) % 96);

        uint8_t* ib = inbox_base + (uint64_t(bank) * kRanks + rank) * inbox_stride;
        float* payload = reinterpret_cast<float*>(ib);
        volatile uint64_t* ready = reinterpret_cast<volatile uint64_t*>(ib + Inbox::flag_off(rows));
        const uint8_t* bc = bcast_base + (uint64_t(bank) * kRanks + rank) * bcast_stride;
        const float* sum = reinterpret_cast<const float*>(bc);
        const volatile uint64_t* gen =
            reinterpret_cast<const volatile uint64_t*>(bc + Bcast::flag_off(rows));

        // ---- producer half
        if (blockIdx.x == 0 && threadIdx.x == 0) trace[epoch - 1].prod_start = wall_clock64();
        for (int k = i; k < n; k += gridDim.x * blockDim.x)
            payload[k] = g_of(x[k], rank, stage, k, active);
        if (!grid_barrier(&pbar[epoch - 1], st.abort, g_abort, nblocks)) return;
        if (threadIdx.x == 0) {
            const unsigned arrived = atomicAdd(&pcounters[epoch], 1u) + 1;
            if (arrived == static_cast<unsigned>(nblocks)) {
                __threadfence_system();
                *ready = epoch;
                trace[epoch - 1].prod_end = wall_clock64();
            }
        }

        // ---- consumer half
        __shared__ int s_ok;
        if (threadIdx.x == 0) {
            const uint64_t t = wall_clock64();
            if (spin_until(gen, epoch, st.abort, g_abort)) {
                trace[epoch - 1].gen_seen = t;
                s_ok = 1;
            } else {
                s_ok = 0;
            }
        }
        __syncthreads();
        if (!s_ok) {
            if (threadIdx.x == 0) {
                atomicExch(err, 1ull);
                if (*gen != epoch) trace[epoch - 1].cons_end = kTraceTimeout;
            }
            return;
        }
        __threadfence_system();
        for (int k = i; k < n; k += gridDim.x * blockDim.x)
            x[k] = h_of(sum[k], rank, stage, k);
        if (!grid_barrier(&cbar[epoch - 1], st.abort, g_abort, nblocks)) return;
        if (threadIdx.x == 0) {
            const unsigned arrived = atomicAdd(&ccounters[epoch], 1u) + 1;
            if (arrived == static_cast<unsigned>(nblocks)) {
                __threadfence_system();
                *(volatile uint64_t*)&credit[bank * kRanks + rank].consumed_epoch = epoch;
                trace[epoch - 1].cons_end = wall_clock64();
            }
        }
    }
}

__global__ void wall_clock_kernel(uint64_t* out) { out[0] = wall_clock64(); }

// Placement visibility touch in the PROTOCOL'S OWN SHAPE: the kernel reads the slot ONCE
// (populating the GPU's L2 line, exactly like a consumer's first spin read of gen), then
// spins for the host to CHANGE it, and reports the first value it saw and the value it
// ended on.  A placement whose CPU writes do not invalidate the GPU's L2 (gfx1030:
// hipDeviceMallocFinegrained AND the hsa coarse pool - measured) serves the stale line
// forever and fails here in seconds instead of hanging the relay.  A one-shot read is NOT
// enough: it misses to memory and looks "visible" even when every later re-read is stale.
__global__ void touch_kernel(const volatile uint64_t* region, volatile uint64_t* seen) {
    __shared__ uint64_t s_first;
    if (threadIdx.x == 0) s_first = *region;   // the warm read (populates the L2 line)
    __syncthreads();
    if (threadIdx.x == 0) {
        // report the warm value FIRST (host-mapped): the host waits for this before
        // rewriting, so the warm read is guaranteed to precede the publish (a plain
        // launch-then-write races the kernel's own start and reads the NEW value as "warm")
        __threadfence_system();
        seen[0] = s_first;
        __threadfence_system();
        unsigned long long spins = 0;
        while (*region == s_first) {            // wait for the host's rewrite
            if (++spins > 100000000ull) break;
        }
        __threadfence_system();
        seen[1] = *region;                      // what the GPU ended up serving
    }
}

} // namespace

void produce_stage(const float* x, void* inbox_base, uint64_t inbox_stride,
                   const StepState* step, int rank, int rows, int active,
                   const volatile unsigned* g_abort, void* stream,
                   unsigned int* counters, StageTrace* trace) {
    const int n = rows * 2560;
    const int blocks = (n + kThreads - 1) / kThreads;
    producer_kernel<<<blocks, kThreads, 0, (cudaStream_t)stream>>>(
        x, static_cast<uint8_t*>(inbox_base), inbox_stride, step, rank, rows, active,
        g_abort, trace, counters);
}

void consume_stage(void* bcast_base, uint64_t bcast_stride, float* x_next,
                   const StepState* step, Credit* credit, int rank, int rows,
                   const volatile unsigned* g_abort, void* stream,
                   unsigned long long* err, unsigned int* counters, StageTrace* trace) {
    const int n = rows * 2560;
    const int blocks = (n + kThreads - 1) / kThreads;
    consumer_kernel<<<blocks, kThreads, 0, (cudaStream_t)stream>>>(
        static_cast<uint8_t*>(bcast_base), bcast_stride, x_next, step, credit, rank, rows, err,
        g_abort, counters, trace);
}

void run_token_persistent(float* x, void* inbox_base, uint64_t inbox_stride, void* bcast_base,
                          uint64_t bcast_stride, const StepState* steps, int nstages, int rank,
                          int rows, int active, Credit* credit, unsigned long long* err,
                          unsigned int* pcounters, unsigned int* ccounters, unsigned int* pbar,
                          unsigned int* cbar, const volatile unsigned* g_abort,
                          StageTrace* trace, void* stream) {
    const int n = rows * 2560;
    const int blocks = (n + kThreads - 1) / kThreads;
    persistent_kernel<<<blocks, kThreads, 0, (cudaStream_t)stream>>>(
        x, static_cast<uint8_t*>(inbox_base), inbox_stride,
        static_cast<uint8_t*>(bcast_base), bcast_stride, steps, nstages, rank, rows, active,
        credit, err, pcounters, ccounters, pbar, cbar, g_abort, trace);
}

void read_wall_clock(uint64_t* out, void* stream) {
    wall_clock_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(out);
}

void touch_region(const volatile uint64_t* region, volatile uint64_t* seen, void* stream) {
    touch_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(region, seen);
}

float ref_g(float x, int rank, int stage, int i, int active) {
    return (rank < active) ? x * (0.3f * (rank + 1)) + stage * 1e-3f + i * 1e-6f : 0.0f;
}

float ref_h(float sum, int rank, int stage, int i) {
    return sum * 0.12f + rank * 1e-3f + i * 1e-7f + stage * 1e-5f;
}

} // namespace strata::kernels::tp5
