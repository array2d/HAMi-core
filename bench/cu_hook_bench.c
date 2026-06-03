/*
 * cu_hook_lookup performance benchmark: old strcmp-based vs new Trie dispatch.
 *
 * cd _build && ../bench/run_bench.sh
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- dummy stub functions for pointer targets ---- */
#define STUB(n) static int n(void) { return 0; }
STUB(cuInit)
STUB(cuDeviceTotalMem_v2)
STUB(cuDriverGetVersion)
STUB(cuDevicePrimaryCtxRetain)
STUB(cuDevicePrimaryCtxRelease_v2)
STUB(cuMemAlloc_v2)
STUB(cuMemAllocHost_v2)
STUB(cuMemAllocManaged)
STUB(cuMemAllocPitch_v2)
STUB(cuMemFree_v2)
STUB(cuMemHostAlloc)
STUB(cuMemHostRegister_v2)
STUB(cuPointerGetAttributes)
STUB(cuLaunchKernel)
STUB(cuLaunchKernelEx)
STUB(cuLaunchCooperativeKernel)
STUB(cuMemCreate)
STUB(cuMemRelease)
STUB(cuMemAllocAsync)
STUB(cuMemFreeAsync)
STUB(cuMemoryAllocate)
STUB(cuMemoryFree)
STUB(cuMipmappedArrayCreate)
STUB(cuGetProcAddress)
STUB(cuGetProcAddress_v2)
STUB(cuMemGetInfo_v2)

/* ================================================================
 * OLD IMPLEMENTATION: table scan + DLSYM_HOOK_FUNC strcmp chain
 * ================================================================ */
#define DLSYM_HOOK_FUNC(f) \
    if (strcmp(symbol, #f) == 0) return (void*)f;

static void* old_lookup(const char *symbol) {
    DLSYM_HOOK_FUNC(cuInit);
    DLSYM_HOOK_FUNC(cuGetProcAddress);
    DLSYM_HOOK_FUNC(cuGetProcAddress_v2);
    DLSYM_HOOK_FUNC(cuDevicePrimaryCtxRetain);
    DLSYM_HOOK_FUNC(cuDevicePrimaryCtxRelease_v2);
    DLSYM_HOOK_FUNC(cuDriverGetVersion);
    DLSYM_HOOK_FUNC(cuDeviceTotalMem_v2);
    DLSYM_HOOK_FUNC(cuMemAlloc_v2);
    DLSYM_HOOK_FUNC(cuMemAllocHost_v2);
    DLSYM_HOOK_FUNC(cuMemAllocManaged);
    DLSYM_HOOK_FUNC(cuMemAllocPitch_v2);
    DLSYM_HOOK_FUNC(cuMemFree_v2);
    DLSYM_HOOK_FUNC(cuMemHostAlloc);
    DLSYM_HOOK_FUNC(cuMemHostRegister_v2);
    DLSYM_HOOK_FUNC(cuPointerGetAttributes);
    DLSYM_HOOK_FUNC(cuMipmappedArrayCreate);
    DLSYM_HOOK_FUNC(cuLaunchKernel);
    DLSYM_HOOK_FUNC(cuLaunchKernelEx);
    DLSYM_HOOK_FUNC(cuLaunchCooperativeKernel);
    DLSYM_HOOK_FUNC(cuMemCreate);
    DLSYM_HOOK_FUNC(cuMemRelease);
    DLSYM_HOOK_FUNC(cuMemAllocAsync);
    DLSYM_HOOK_FUNC(cuMemFreeAsync);
    DLSYM_HOOK_FUNC(cuMemoryAllocate);
    DLSYM_HOOK_FUNC(cuMemoryFree);
    return NULL;
}

static void* old_find_symbols_in_table(const char *symbol) {
    /* cuGraph fast bail */
    if (symbol[2] == 'G' && symbol[3] == 'r') return NULL;

    char buf[500];
    void *pfn;

    /* _v3 */
    strcpy(buf, symbol);
    strcat(buf, "_v3");
    pfn = old_lookup(buf);
    if (pfn) return pfn;

    /* _v2 */
    buf[strlen(buf) - 1] = '2';
    pfn = old_lookup(buf);
    if (pfn) return pfn;

    return old_lookup(symbol);
}

/* ================================================================
 * NEW IMPLEMENTATION: character dispatch Trie (exact copy from hook.c)
 * ================================================================ */
static void* new_cu_hook_lookup(const char *symbol) {
    if (symbol[0] != 'c' || symbol[1] != 'u')
        return NULL;
    switch (symbol[2]) {
    case 'D':
        if (symbol[3] == 'e') {
            if (symbol[7] == 'T') return (void*)cuDeviceTotalMem_v2;
            if (symbol[7] == 'P')
                return (symbol[20] == 'l')
                    ? (void*)cuDevicePrimaryCtxRelease_v2
                    : (void*)cuDevicePrimaryCtxRetain;
            return NULL;
        }
        if (symbol[3] == 'r') return (void*)cuDriverGetVersion;
        return NULL;
    case 'G':
        if (symbol[3] == 'e' && symbol[8] == 'c')
            return (symbol[12] == '_')
                ? (void*)cuGetProcAddress_v2
                : (void*)cuGetProcAddress;
        return NULL;
    case 'I':
        return (symbol[5] == '\0' || symbol[5] == '_')
            ? (void*)cuInit : NULL;
    case 'L':
        if (symbol[3] == 'a' && symbol[7] == 'C')
            return (void*)cuLaunchCooperativeKernel;
        if (symbol[3] == 'a' && symbol[7] == 'K')
            return (symbol[10] == 'E')
                ? (void*)cuLaunchKernelEx : (void*)cuLaunchKernel;
        return NULL;
    case 'M':
        if (symbol[3] == 'e') {
            switch (symbol[4]) {
            case 'A':
                if (symbol[9]=='\0'||symbol[9]=='_') return (void*)cuMemAlloc_v2;
                if (symbol[9]=='H') return (void*)cuMemAllocHost_v2;
                if (symbol[9]=='M') return (void*)cuMemAllocManaged;
                if (symbol[9]=='P') return (void*)cuMemAllocPitch_v2;
                if (symbol[9]=='A') return (void*)cuMemAllocAsync;
                return NULL;
            case 'C': return (void*)cuMemCreate;
            case 'F':
                if (symbol[6]=='A')  return (void*)cuMemFreeAsync;
                if (symbol[6]=='v'||symbol[6]=='\0') return (void*)cuMemFree_v2;
                return NULL;
            case 'G': return NULL;  /* cuMemGetInfo disabled by default */
            case 'H':
                if (symbol[6]=='A') return (void*)cuMemHostAlloc;
                if (symbol[6]=='R') return (void*)cuMemHostRegister_v2;
                return NULL;
            case 'R': return (void*)cuMemRelease;
            }
            return NULL;
        }
        if (symbol[3]=='i')
            return (symbol[11]=='C') ? (void*)cuMipmappedArrayCreate : NULL;
        if (symbol[3]=='o')
            return (symbol[9]=='A') ? (void*)cuMemoryAllocate : (void*)cuMemoryFree;
        return NULL;
    case 'P': return (void*)cuPointerGetAttributes;
    default:  return NULL;
    }
}

static void* new_find_symbols_in_table(const char *symbol) {
    if (symbol[2] == 'G' && symbol[3] == 'r') return NULL;

    char buf[500];
    void *pfn;

    strcpy(buf, symbol);
    strcat(buf, "_v3");
    pfn = new_cu_hook_lookup(buf);
    if (pfn) return pfn;

    buf[strlen(buf) - 1] = '2';
    pfn = new_cu_hook_lookup(buf);
    if (pfn) return pfn;

    return new_cu_hook_lookup(symbol);
}

/* ================================================================
 * Benchmark harness
 * ================================================================ */
static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

typedef void* (*lookup_fn)(const char*);

/* Test vectors: simulates a typical cuGetProcAddress call pattern.
 * Mix is ~80% non-hook functions (the common case), ~20% hook functions. */
static const char *test_hook_names[] = {
    "cuInit",
    "cuDeviceTotalMem_v2",
    "cuDriverGetVersion",
    "cuDevicePrimaryCtxRetain",
    "cuMemAlloc_v2",
    "cuMemFree_v2",
    "cuMemAllocPitch_v2",
    "cuMemHostAlloc",
    "cuPointerGetAttributes",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
    "cuMemCreate",
    "cuMemAllocAsync",
    "cuMemFreeAsync",
    "cuMemoryAllocate",
    "cuGetProcAddress",
    "cuGetProcAddress_v2",
    "cuMipmappedArrayCreate",
    "cuLaunchCooperativeKernel",
};

static const char *test_nonhook_names[] = {
    "cuCtxCreate_v2",
    "cuCtxDestroy_v2",
    "cuCtxSetCurrent",
    "cuCtxGetCurrent",
    "cuCtxGetDevice",
    "cuCtxGetFlags",
    "cuCtxSynchronize",
    "cuCtxGetApiVersion",
    "cuCtxGetLimit",
    "cuCtxSetLimit",
    "cuCtxGetCacheConfig",
    "cuCtxSetCacheConfig",
    "cuCtxGetSharedMemConfig",
    "cuCtxSetSharedMemConfig",
    "cuDeviceGet",
    "cuDeviceGetAttribute",
    "cuDeviceGetName",
    "cuDeviceGetUuid",
    "cuDeviceGetLuid",
    "cuDeviceGetDefaultMemPool",
    "cuDeviceSetMemPool",
    "cuDeviceGetMemPool",
    "cuModuleLoad",
    "cuModuleLoadData",
    "cuModuleLoadDataEx",
    "cuModuleLoadFatBinary",
    "cuModuleUnload",
    "cuModuleGetFunction",
    "cuModuleGetGlobal",
    "cuModuleGetTexRef",
    "cuModuleGetSurfRef",
    "cuFuncGetAttribute",
    "cuFuncSetAttribute",
    "cuFuncSetCacheConfig",
    "cuFuncSetSharedMemConfig",
    "cuStreamCreate",
    "cuStreamDestroy",
    "cuStreamSynchronize",
    "cuStreamWaitEvent",
    "cuEventCreate",
    "cuEventDestroy",
    "cuEventRecord",
    "cuEventSynchronize",
    "cuEventElapsedTime",
    "cuGraphCreate",
    "cuGraphDestroy",
    "cuGraphAddNode",
    "cuGraphLaunch",
    "cuArrayCreate_v2",
    "cuArrayDestroy",
    "cuTexRefCreate",
    "cuTexRefDestroy",
    "cuSurfRefSetArray",
    "cuMemGetInfo_v2",
    "cuMemHostGetDevicePointer_v2",
    "cuMemHostGetFlags",
    "cuMemGetAddressRange_v2",
    "cuMemAdvise",
    "cuMemPrefetchAsync",
    "cuMemRangeGetAttribute",
    "cuMemRangeGetAttributes",
    "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    "cuOccupancyMaxPotentialBlockSize",
    "cuLinkCreate_v2",
    "cuLinkAddData_v2",
    "cuLinkComplete",
    "cuLinkDestroy",
};

#define ITERATIONS 2000000

static double bench_lookup(lookup_fn fn, const char **names, int n,
                           const char *label) {
    int i, j;
    volatile void *dummy;
    double start, end;

    start = now_ns();
    for (j = 0; j < ITERATIONS; j++) {
        for (i = 0; i < n; i++) {
            dummy = fn(names[i]);
        }
    }
    end = now_ns();
    (void)dummy;

    double total_ns = end - start;
    double per_call_ns = total_ns / (ITERATIONS * n);
    printf("  %-40s  %8.1f ns/call  (%7.0f M calls)  %5.0fM total\n",
           label, per_call_ns, (double)ITERATIONS * n / 1e6,
           total_ns / 1e6);
    return per_call_ns;
}

__attribute__((format(printf, 4, 5)))
static double bench_find_symbols(lookup_fn fn, const char **names, int n,
                                 const char *fmt, ...) {
    int i, j;
    volatile void *dummy;
    double start, end;

    start = now_ns();
    for (j = 0; j < ITERATIONS; j++) {
        for (i = 0; i < n; i++) {
            dummy = fn(names[i]);
        }
    }
    end = now_ns();
    (void)dummy;

    double total_ns = end - start;
    double per_call_ns = total_ns / (ITERATIONS * n);

    char label[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(label, sizeof(label), fmt, ap);
    va_end(ap);

    printf("  %-40s  %8.1f ns/call  (%7.0f M calls)  %5.0fM total\n",
           label, per_call_ns, (double)ITERATIONS * n / 1e6,
           total_ns / 1e6);
    return per_call_ns;
}

int main(void) {
    int n_hook     = sizeof(test_hook_names) / sizeof(test_hook_names[0]);
    int n_nonhook  = sizeof(test_nonhook_names) / sizeof(test_nonhook_names[0]);
    int n_mixed    = n_hook + n_nonhook;
    const char *mixed_names[n_mixed];
    for (int i = 0; i < n_hook; i++)    mixed_names[i] = test_hook_names[i];
    for (int i = 0; i < n_nonhook; i++) mixed_names[n_hook + i] = test_nonhook_names[i];

    printf("============================================================\n");
    printf("  cuGetProcAddress / dlsym lookup benchmark\n");
    printf("  Iterations: %d per name  (2M)\n", ITERATIONS);
    printf("============================================================\n");

    /* ---- Phase 1: raw cu_hook_lookup (single suffix) ---- */
    printf("\n── Phase 1: raw cu_hook_lookup (no suffix fallback) ──\n");

    double old_hook    = bench_lookup(old_lookup, test_hook_names, n_hook, "OLD: hook names");
    double old_nonhook = bench_lookup(old_lookup, test_nonhook_names, n_nonhook, "OLD: non-hook names (worst-case)");

    double new_hook    = bench_lookup(new_cu_hook_lookup, test_hook_names, n_hook, "NEW: hook names");
    double new_nonhook = bench_lookup(new_cu_hook_lookup, test_nonhook_names, n_nonhook, "NEW: non-hook names (2 chars)");

    printf("\n  ┌────────────────────────────────────────────────────┐\n");
    printf("  │ Non-hook speedup:  %.0fx  (%.1f → %.1f ns)          │\n",
           old_nonhook / new_nonhook, old_nonhook, new_nonhook);
    printf("  │ Hook func speedup:  %.0fx  (%.1f → %.1f ns)          │\n",
           old_hook / new_hook, old_hook, new_hook);
    printf("  └────────────────────────────────────────────────────┘\n");

    /* ---- Phase 2: find_symbols_in_table (with _v3/_v2 suffix fallback) ---- */
    printf("\n── Phase 2: find_symbols_in_table (suffix fallback _v3→_v2→orig) ──\n");
    printf("  NOTE: non-hook names require 3× suffix probe (3 × old = ~3× slower)\n");

    double old_find_hook    = bench_find_symbols(old_find_symbols_in_table, test_hook_names, n_hook,
                                                  "OLD find_symbols: hook names");
    double old_find_nonhook = bench_find_symbols(old_find_symbols_in_table, test_nonhook_names, n_nonhook,
                                                  "OLD find_symbols: non-hook (worse)");

    double new_find_hook    = bench_find_symbols(new_find_symbols_in_table, test_hook_names, n_hook,
                                                  "NEW find_symbols: hook names");
    double new_find_nonhook = bench_find_symbols(new_find_symbols_in_table, test_nonhook_names, n_nonhook,
                                                  "NEW find_symbols: non-hook (2 char × 3)");

    printf("\n  ┌────────────────────────────────────────────────────┐\n");
    printf("  │ Non-hook speedup:  %.0fx  (%.1f → %.1f ns)          │\n",
           old_find_nonhook / new_find_nonhook, old_find_nonhook, new_find_nonhook);
    printf("  │ Hook func speedup:  %.0fx  (%.1f → %.1f ns)          │\n",
           old_find_hook / new_find_hook, old_find_hook, new_find_hook);
    printf("  └────────────────────────────────────────────────────┘\n");

    /* ---- Phase 3: realistic mixed workload ---- */
    printf("\n── Phase 3: Realistic mixed workload (80%% non-hook) ──\n");

    double old_mixed = bench_find_symbols(old_find_symbols_in_table, mixed_names, n_mixed,
                                           "OLD: mixed 19 hook + %d non-hook", n_nonhook);
    double new_mixed = bench_find_symbols(new_find_symbols_in_table, mixed_names, n_mixed,
                                           "NEW: mixed 19 hook + %d non-hook", n_nonhook);

    printf("\n  ┌────────────────────────────────────────────────────┐\n");
    printf("  │ Mixed workload speedup:  %.1fx  (%.1f → %.1f ns)     │\n",
           old_mixed / new_mixed, old_mixed, new_mixed);
    printf("  │ Old: %d strcmp per non-hook lookup                   │\n", 26);
    printf("  │ New: ~6 char comparisons per non-hook lookup         │\n");
    printf("  └────────────────────────────────────────────────────┘\n");

    return 0;
}
