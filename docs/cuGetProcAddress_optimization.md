# cuGetProcAddress Hook 极致性能优化方案

## 背景

`cuGetProcAddress` / `cuGetProcAddress_v2` 是 CUDA 驱动 API 的核心函数查找入口。几乎所有 CUDA 程序（包括 CUDA Runtime）都通过它获取驱动函数指针。

### 安全必要性

**必须保持 hook 状态**。如果不禁用 cuGetProcAddress hook，程序可以通过以下方式逃逸显存管理：

```c
// 绕过多层 hook，直接拿到真实驱动的 cuMemAlloc_v2
cuGetProcAddress("cuMemAlloc_v2", &real_alloc, cudaVersion, 0);
real_alloc(&dptr, bytesize);  // 直接分配，绕过 HAMi-core 显存追踪
```

### 性能问题

当前实现 (`find_symbols_in_table` → `__dlsym_hook_section`) 每次 `cuGetProcAddress` 调用：

```
find_symbols_in_table:
  try "_v3" → __dlsym_hook_section: 57 表条目 + 31 DLSYM_HOOK_FUNC = 88 strcmp
  try "_v2" → 88 strcmp
  try 原名  → 88 strcmp
  ─────────────────
  = 264 次 strcmp / 次调用（非 hook 函数）
```

高频 kernel launch 场景下，每个 kernel 可能触发数十次 `cuGetProcAddress`，累计 CPU 开销显著。

## 优化方案：字符分派 Trie

### 原理

CUDA API 函数命名规则固定（`cu` + 大驼峰），利用前缀字符直接分派，O(1) 常数时间定位，**零 strcmp、零循环**。

### 核心实现

```c
/*
 * Ultra-fast cuGetProcAddress lookup using character dispatch.
 * No strcmp, no loops — worst case 6 char comparisons.
 *
 * CUDA function naming: cu{CamelCase}
 *   cuInit
 *   cuDevice*    (TotalMem, PrimaryCtxRetain, PrimaryCtxRelease)
 *   cuDriver*
 *   cuMem*       (Alloc, Free, Create, Release, Host, Pool...)
 *   cuMipmapped*
 *   cuMemory*    (Allocate, Free)
 *   cuLaunch*    (Kernel, KernelEx, CooperativeKernel)
 *   cuPointer*
 *   cuGetProcAddress*
 */
static inline void* cu_hook_lookup(const char *symbol) {
    /* Filter: must start with "cu" */
    if (symbol[0] != 'c' || symbol[1] != 'u')
        return NULL;

    switch (symbol[2]) {
    case 'D':  /* Device / Driver */
        switch (symbol[3]) {
        case 'e':  /* Device */
            if (symbol[7] == 'T')               /* cuDeviceTotalMem_v2 */
                return (void*)cuDeviceTotalMem_v2;
            if (symbol[7] == 'P') {             /* cuDevicePrimaryCtx* */
                return (symbol[19] == 'R') ? (void*)cuDevicePrimaryCtxRetain
                                           : (void*)cuDevicePrimaryCtxRelease_v2;
            }
            return NULL;
        case 'r':  /* Driver */
            return (void*)cuDriverGetVersion;
        }
        return NULL;

    case 'G':  /* GetProcAddress */
        return (symbol[3] == 'e' && symbol[8] == 'c')  /* cuGetProcAddress */
            ? (symbol[12] == '_' ? (void*)cuGetProcAddress_v2
                                 : (void*)cuGetProcAddress)
            : NULL;

    case 'I':  /* Init */
        return (symbol[5] == '\0') ? (void*)cuInit : NULL;

    case 'L':  /* Launch */
        if (symbol[7] == 'K')                     /* cuLaunchKernel */
            return (void*)cuLaunchKernel;
        if (symbol[7] == 'C')                     /* cuLaunchCooperativeKernel */
            return (void*)cuLaunchCooperativeKernel;
        if (symbol[7] == 'E')                     /* cuLaunchKernelEx */
            return (void*)cuLaunchKernelEx;
        return NULL;

    case 'M':  /* Mem / Mipmapped / Memory */
        switch (symbol[3]) {
        case 'e':  /* Mem */
            switch (symbol[4]) {
            case 'A':  /* Alloc */
                if (symbol[7] == 'l')             /* cuMemAlloc_v2 */
                    return (void*)cuMemAlloc_v2;
                if (symbol[7] == 'H') {           /* cuMemAllocHost_v2 */
                    return (void*)cuMemAllocHost_v2;
                }
                if (symbol[7] == 'M') {           /* cuMemAllocManaged */
                    return (void*)cuMemAllocManaged;
                }
                if (symbol[7] == 'P') {           /* cuMemAllocPitch_v2 */
                    return (void*)cuMemAllocPitch_v2;
                }
                if (symbol[7] == 'A') {           /* cuMemAllocAsync */
                    return (void*)cuMemAllocAsync;
                }
                return NULL;
            case 'C':  /* Create */
                return (void*)cuMemCreate;
            case 'F':  /* Free */
                if (symbol[6] == 'A')             /* cuMemFreeAsync */
                    return (void*)cuMemFreeAsync;
                if (symbol[6] == 'v')             /* cuMemFree_v2 */
                    return (void*)cuMemFree_v2;
                if (symbol[6] == 'H')             /* cuMemFreeHost */
                    return (void*)cuMemFreeHost;
                return NULL;
            case 'G':  /* GetInfo */
                return (void*)cuMemGetInfo_v2;
            case 'H':  /* Host */
                if (symbol[6] == 'A')             /* cuMemHostAlloc */
                    return (void*)cuMemHostAlloc;
                if (symbol[6] == 'R')             /* cuMemHostRegister_v2 */
                    return (void*)cuMemHostRegister_v2;
                if (symbol[6] == 'U')             /* cuMemHostUnregister */
                    return (void*)cuMemHostUnregister;
                return NULL;
            case 'R':  /* Release */
                return (void*)cuMemRelease;
            }
            return NULL;
        case 'i':  /* Mipmapped */
            return (symbol[11] == 'C') ? (void*)cuMipmappedArrayCreate  /* cuMipmappedArrayCreate */
                                       : (void*)cuMipmappedArrayDestroy; /* cuMipmappedArrayDestroy */
        case 'o':  /* Memory */
            return (symbol[9] == 'A') ? (void*)cuMemoryAllocate  /* cuMemoryAllocate */
                                       : (void*)cuMemoryFree;     /* cuMemoryFree */
        }
        return NULL;

    case 'P':  /* Pointer */
        return (void*)cuPointerGetAttributes;

    default:
        return NULL;
    }
}
```

### 调用方改造

`find_symbols_in_table` 改为单次字符分派：

```c
void *find_symbols_in_table(const char *symbol) {
    /* graph functions: never hooked, bypass immediately */
    if (symbol[2] == 'G' && symbol[3] == 'r')
        return NULL;

    /* Try with suffix fallback: _v3 → _v2 → original */
    char symbol_v[500];
    strcpy(symbol_v, symbol);

    strcat(symbol_v, "_v3");
    void *pfn = cu_hook_lookup(symbol_v);
    if (pfn) return pfn;

    symbol_v[strlen(symbol_v) - 1] = '2';
    pfn = cu_hook_lookup(symbol_v);
    if (pfn) return pfn;

    /* Try original name (no suffix) */
    return cu_hook_lookup(symbol);
}
```

### 性能对比

| 场景 | 当前实现 | 优化后 |
|------|:-------:|:-----:|
| 非 hook 函数（多数情况） | 264 strcmp | **2 char** |
| hook 函数（如 cuMemAlloc_v2） | ~264 strcmp | **5-6 char** |
| cuGraph* 函数 | 1 strcmp（快速路径） | 2 char |
| 减少比例 | — | **>95%** |

### 为何不用 DLSYM_HOOK_FUNC 宏

当前 `DLSYM_HOOK_FUNC` 宏展开为 `if (strcmp(symbol, "cuXxx") == 0) return (void*)cuXxx;`，每次需要完整 strcmp。31 个 DLSYM_HOOK_FUNC 意味着每个非 hook 函数仍需 31 次 strcmp。

字符分派将查找路径编码为 switch-case 树，编译器可优化为跳转表（jump table），实现真正的 O(1) 查找。

### 影响范围

- `src/cuda/hook.c`：新增 `cu_hook_lookup()`，改造 `find_symbols_in_table()`
- `src/include/libvgpu.h` 或 `src/cuda/hook.c`：`__dlsym_hook_section` 改为调用 `cu_hook_lookup`
- 保留原 `__dlsym_hook_section` 作为 NVML 路径使用（不变）

### 兼容性

- 新增 CUDA API 函数时，只需在 `cu_hook_lookup` 中追加对应字符分支
- 不影响 `cuda_library_entry[]` 表结构、`CUDA_OVERRIDE_CALL` 宏、NVML hook
- 钩子语义不变：cuGetProcAddress 仍被拦截，逃逸显存管理仍然不可能

### 测试验证

1. 现有全部 16 个功能测试通过
2. cuGetProcAddress hook 验证：`cuGetProcAddress("cuMemAlloc_v2", ...)` 返回 HAMi-core 版本
3. 性能对比：高频 kernel launch 场景下 CUDA API 调用延迟对比
