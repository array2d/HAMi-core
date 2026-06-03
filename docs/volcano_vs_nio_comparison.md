# Volcano vs NIO 分支对比文档

> 基准: `nio` (commit `0bc4a07`) vs `volcano` (commit `3002658`)  
> 52 文件变更, +3427 / -828 行

---

## 1. 架构级差异

### 1.1 cuGetProcAddress 查找 — 字符分派 Trie (volcano 新增)

| 维度 | NIO 分支 | Volcano 分支 |
|------|----------|-------------|
| 查找方式 | `__dlsym_hook_section` → 10 个 `DLSYM_HOOK_FUNC` strcmp 链 | `cu_hook_lookup()` 字符分派 switch-case |
| 时间复杂度 | O(n) — 每个非 hook 函数 ~10 strcmp | O(1) — 每个函数 ~2-6 char 比较 |
| 非 hook 耗时 | ~233 ns/call | ~3.9 ns/call (**60x**) |
| cuGetProcAddress hook | 已启用 (strcmp 查找) | 已启用 + 后缀回退 (_v3→_v2→orig) |
| `find_symbols_in_table` | 调用 `__dlsym_hook_section` 3 次 | 调用 `cu_hook_lookup` 3 次 |
| 真实混合负载 | ~656 ns/call | ~22.6 ns/call (**29x**) |

**NIO `__dlsym_hook_section`** (strcmp 链):
```c
void* __dlsym_hook_section(void* handle, const char* symbol) {
    DLSYM_HOOK_FUNC(cuInit);
    DLSYM_HOOK_FUNC(cuDeviceTotalMem_v2);
    DLSYM_HOOK_FUNC(cuDriverGetVersion);
    DLSYM_HOOK_FUNC(cuDevicePrimaryCtxRetain);
    DLSYM_HOOK_FUNC(cuDevicePrimaryCtxRelease_v2);
    DLSYM_HOOK_FUNC(cuMemAlloc_v2);
    DLSYM_HOOK_FUNC(cuMemAllocHost_v2);
    DLSYM_HOOK_FUNC(cuMemAllocManaged);
    DLSYM_HOOK_FUNC(cuMemAllocPitch_v2);
    DLSYM_HOOK_FUNC(cuMemFree_v2);
    // ... 共 ~10 个 (kernel launch 被注释)
    DLSYM_HOOK_FUNC(cuGetProcAddress);
    DLSYM_HOOK_FUNC(cuGetProcAddress_v2);
    return NULL;
}
```

**Volcano `__dlsym_hook_section`** (Trie 委托):
```c
void* __dlsym_hook_section(void* handle, const char* symbol) {
    return cu_hook_lookup(symbol);  // 单行委托, O(1)
}
```

**Volcano `cu_hook_lookup`** 核心实现: 见 `src/cuda/hook.c:140-242`，约 100 行 switch-case 树，编译器优化为跳转表。

**基准测试**: `bench/run_bench.sh` 可独立运行验证。

### 1.2 dlsym 初始化 — 多 glibc 版本 + CUDA_REDIRECT (volcano 新增)

| 维度 | NIO | Volcano |
|------|-----|---------|
| glibc 版本 | 单一 `GLIBC_2.2.5` | 遍历 7 个版本 (amd64 + arm64) |
| 回退方案 | `_dl_sym` (内部符号, 已废弃) | `dlopen("libc.so.6")` → `dlsym` |
| CUDA 转发 | 无 | `CUDA_REDIRECT` 环境变量 / `/usr/local/vgpu/libvgpu.so` |
| cuGetExportTable | 直接 hook | 跳过 preInit (兼容 CUDA 12.8+) |

**NIO** dlsym 回退:
```c
real_dlsym = dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5");
// ...
real_dlsym = _dl_sym(RTLD_NEXT, "dlsym", dlsym);  // 不可移植!
```

**Volcano** dlsym 回退:
```c
const char* glibc_versions[] = {
    "GLIBC_2.2.5",  // amd64
    "GLIBC_2.17",   // arm64
    "GLIBC_2.3", "GLIBC_2.4", "GLIBC_2.10",
    "GLIBC_2.18", "GLIBC_2.22", NULL
};
for (int i = 0; glibc_versions[i] != NULL; i++) { ... }

// 回退: dlopen libc → dlsym (可移植)
void *libc_handle = dlopen("libc.so.6", RTLD_LAZY);
real_dlsym = dlsym(libc_handle, "dlsym");

// CUDA 转发 (vgpulib)
char *path = getenv("CUDA_REDIRECT");
vgpulib = dlopen(path ?: "/usr/local/vgpu/libvgpu.so", RTLD_LAZY);
```

### 1.3 Hook 表 — 全面覆盖 vs 最小子集

| 类别 | NIO 条目数 | Volcano 条目数 |
|------|-----------|---------------|
| 设备管理 | 2 (cuDeviceTotalMem, cuDriverGetVersion) | 3 (+cuDeviceGetCount) |
| 上下文 | 2 | 3 (+cuGetExportTable) |
| 内存分配 | 7 | 20 (+Array, Host, Mipmapped, 虚拟内存) |
| 内存池 | 0 | 13 (完整 Pool API) |
| Kernel Launch | 0 (注释掉) | 3 (cuLaunchKernel + Ex + Cooperative) |
| 外部资源互操作 | 0 | 8 (ExternalMemory, ExternalSemaphore) |
| Proc Address | 2 | 2 (已恢复, 用 Trie 优化) |
| **总计** | **~16** | **~57** |

NIO 的 kernel launch hook 被注释掉:
```c
// NIO hook.c
// {.name = "cuLaunchKernel"},       // ← 注释
// {.name = "cuLaunchCooperativeKernel"},  // ← 注释
```

Volcano 启用完整 kernel launch 链:
```c
{.name = "cuLaunchKernel"},
{.name = "cuLaunchKernelEx"},          // ← CUDA 12.0+
{.name = "cuLaunchCooperativeKernel"},
```

对应的 rate limiter 实现也在 `src/cuda/memory.c` 中恢复。

**注意**: NIO 注释掉 kernel launch 是为了在特定场景减少 hook 开销，但会导致 rate limiter (SM 利用率限制) 失效。

### 1.4 日志系统 — 缓存日志级别 (volcano 新增)

| 维度 | NIO | Volcano |
|------|-----|---------|
| 日志级别获取 | 每次 `getenv("LIBCUDA_LOG_LEVEL")` + `atoi` | 启动时缓存到 `g_log_level` |
| 每次 LOG 调用开销 | ~2 次函数调用 + 字符串比较 | 1 次整数比较 |
| 模块化 | 所有宏在 log_utils.h 内联 | `g_log_level` 在 `log_utils.c` 中定义 |
| 文件日志 | 仅 stderr | 支持 `FILEDEBUG` 编译选项 → `/tmp/vgpulog` |
| 默认级别 | 无明确默认 (依赖 getenv NULL) | `g_log_level = 2` (WARN) |

**NIO** 每次 LOG_INFO:
```c
if ((getenv("LIBCUDA_LOG_LEVEL")!=NULL) && (atoi(getenv("LIBCUDA_LOG_LEVEL"))>=3))
    fprintf(...)
```

**Volcano** 每次 LOG_INFO:
```c
if (g_log_level >= 3)
    fprintf(...)
// log_utils_init() 在 preInit 中调用一次, 缓存 getenv 结果
```

---

## 2. CUDA API 函数级差异

### 2.1 内存分配 — 新增 OOM 检查的函数

**Volcano 新增** (NIO 无 hook):
- `cuMemAllocHost_v2` — 带 OOM 回退 (OOM 时 `cuMemFreeHost`)
- `cuMemHostAlloc` — 带 OOM 回退
- `cuMemHostRegister_v2` — 带 OOM 回退 (`cuMemHostUnregister`)
- `cuMipmappedArrayCreate` — 带 OOM 回退
- `cuMemCreate` — 线程安全修复 + OOM 检查
- `cuMemRelease` — 带 `remove_chunk_only`

**cuMemoryAllocate / cuMemoryFree 实现变化**:

NIO:
```c
CUresult cuMemoryAllocate(CUdeviceptr* dptr, size_t bytesize, size_t* bytesallocated, void* data) {
    if (bytesallocated != NULL)
        *bytesallocated = bytesize;
    return cuMemAllocManaged(dptr, bytesize, CU_MEM_ATTACH_GLOBAL);
}
```

Volcano:
```c
CUresult cuMemoryAllocate(CUdeviceptr* dptr, size_t bytesize, void* data) {
    return cuMemAlloc_v2(dptr, bytesize);  // 走 allocator 路径, 支持 oversized 分片
}
```

### 2.2 Kernel Launch — Rate Limiter (volcano 恢复)

NIO 注释掉了全部 kernel launch hook，Volcano 恢复了三个:
- `cuLaunchKernel` — 标准 launch + rate limiter
- `cuLaunchKernelEx` — CUDA 12.0+ 扩展 launch (NIO 不存在)  
- `cuLaunchCooperativeKernel` — Cooperative launch

### 2.3 cuMemGetInfo_v2 — 显存信息修复

NIO:
```c
*free = limit - usage;
*total = limit;
```

Volcano (修复 limit > physical total 的边界情况):
```c
size_t actual_limit = (limit > *total) ? *total : limit;
*free = (actual_limit > usage) ? (actual_limit - usage) : 0;
*total = actual_limit;
```

---

## 3. 分配器 (allocator) 差异

### 3.1 设备感知追踪

| 维度 | NIO | Volcano |
|------|-----|---------|
| chunk 结构 | `{address, length}` | `{address, length, dev}` — 记录设备 ID |
| `add_chunk_only` 签名 | `(addr, size)` | `(addr, size, dev)` |
| `oom_check` | 每次调用 `cuDeviceGetCount` | 不调用 `cuDeviceGetCount` |
| async 分配列表 | 复用 `device_overallocated` | 独立 `device_allocasync` 列表 |

### 3.2 锁优化

NIO 的 `add_chunk` 全程持锁（含 GPU 分配）:
```c
int add_chunk(CUdeviceptr *address, size_t size) {
    // 整个函数在锁内
    cuMemoryAllocate(...);  // 慢
    LIST_ADD(...);
}
```

Volcano 锁外分配 + 锁内追踪:
```c
int add_chunk(CUdeviceptr *address, size_t size) {
    // 1. OOM 预检 (无锁)
    if (oom_check(dev, size)) return CUDA_ERROR_OUT_OF_MEMORY;
    // 2. GPU 分配 (无锁 — 慢操作)
    res = cuMemoryAllocate(address, size, NULL);
    // 3. 追踪 + 二次 OOM 检查 (锁内 — 微秒级)
    pthread_mutex_lock(&mutex);
    if (oom_check(dev, size)) { ... }  // 竞态二次确认
    LIST_ADD(...);
    pthread_mutex_unlock(&mutex);
}
```

### 3.3 Async 显存池追踪

Volcano 新增 `device_allocasync` 独立列表，通过 `cuMemPoolGetAttribute` 获取池限制，实现异步分配的精确内存计费，避免重复统计。

---

## 4. NVML Hook 差异

| 维度 | NIO | Volcano |
|------|-----|---------|
| hook 表 | 5 函数 | 6 函数 (+`nvmlDeviceGetIndex`) |
| `_nvmlDeviceGetMemoryInfo` | 签名用 `nvmlMemory_t*` 强类型 | 签名用 `void*` + version switch |
| cuda↔nvml 映射 | `cuda_to_nvml_map[16]` 全局数组 | `cuda_to_nvml_map_array[CUDA_DEVICE_MAX_COUNT]` + 内联函数 |
| 初始化 | 直接 `load_nvml_libraries` | 先 `ensure_initialized()` 再加载 |
| dlsym 回退 | `_dl_sym` | `dlvsym` → `libc dlopen` |
| `nvmlPostInit` | 无 | `init_device_info()` |
| `nvmlInitWithFlags` | 调用 real nvml | 先 `pthread_once` 预初始化 |
| 返回码处理 | 无 default case | `default: return NVML_ERROR_INVALID_ARGUMENT` |

---

## 5. 进程间通信差异

### 5.1 Unified Lock

NIO (基于文件存在性 — 竞态条件风险):
```c
int try_lock_unified_lock() {
    int fd = open("/tmp/vgpulock/lock", O_CREAT | O_EXCL, S_IRWXU);
    while (fd == -1) {
        if (cnt == 18) remove("/tmp/vgpulock/lock");  // 超时强制删除
        usleep(rand() % 5 * 100000 + 100000);
        fd = open(...);
    }
}
```

Volcano (基于 `flock` — 内核级文件锁):
```c
int try_lock_unified_lock() {
    lock_fd = open("/tmp/vgpulock/lock", O_CREAT | O_RDWR, 0666);
    flock(lock_fd, LOCK_EX);  // 阻塞等待, 内核保证原子性
}
```

### 5.2 dlsym 递归检测

NIO (int 截断风险 — `pthread_t` 可能 > INT_MAX):
```c
typedef struct { int tid; void *pointer; } tid_dl_map;
// ...
int tid = pthread_self();  // 截断!
if (dlmap[i].tid == tid) ...
```

Volcano (使用 `pthread_equal` — 可移植):
```c
typedef struct { pthread_t tid; void *pointer; } tid_dl_map;
// ...
pthread_t tid = pthread_self();
if (pthread_equal(dlmap[i].tid, tid)) ...
```

### 5.3 Post-Init 主机 PID 检测

NIO:
```c
try_lock_unified_lock();
res = set_task_pid();
try_unlock_unified_lock();
```

Volcano (超时保护, 防止死进程永久阻塞):
```c
int lock_acquired = lock_postinit();  // 带超时
if (lock_acquired) {
    res = set_task_pid();
    unlock_postinit();
} else {
    LOG_WARN("Skipped host PID detection due to lock timeout");
    res = NVML_ERROR_TIMEOUT;
}
```

### 5.4 Context 管理

NIO (context_size 未正确清理):
```c
// cuDevicePrimaryCtxRelease_v2: context_size 从未被 rm_gpu_device_memory_usage
// 导致显存泄漏
```

Volcano (正确清理 + 防护):
```c
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    if (ctx_activate[dev] == 1) {
        rm_gpu_device_memory_usage(getpid(), dev, context_size, 0);
    }
    ctx_activate[dev] = 0;
    ...
}
```

---

## 6. 新增功能 / 文件

| 文件 | 说明 |
|------|------|
| `src/log_utils.c` | 缓存日志级别, 替代每次 getenv/atoi |
| `src/include/multi_func_hook.h` | CUDA 版本感知函数名映射 (如 cuGraphAddKernelNode → _v2) |
| `bench/cu_hook_bench.c` | cu_hook_lookup 性能基准测试 |
| `bench/run_bench.sh` | 基准测试运行脚本 |
| `docs/cuGetProcAddress_optimization.md` | Trie 优化方案文档 |
| `test/test_mem_create.c` | cuMemCreate 功能测试 |
| `test/test_multi_gpu_utilization.cu` | 多 GPU 利用率测试 |
| `hack/check_cuda_hook_consistency.py` | Hook 一致性检查脚本 |

---

## 7. 外部资源互操作 (volcano 新增)

NIO 完全没有这 8 个 External Resource Interop 函数:

- `cuImportExternalMemory` / `cuDestroyExternalMemory`
- `cuExternalMemoryGetMappedBuffer` / `cuExternalMemoryGetMappedMipmappedArray`
- `cuImportExternalSemaphore` / `cuDestroyExternalSemaphore`
- `cuSignalExternalSemaphoresAsync` / `cuWaitExternalSemaphoresAsync`

这些函数在 `cuda_library_entry[]` 中有条目但无对应的 hook 实现（属于 pass-through，仅用于 `CUDA_OVERRIDE_CALL` 正确解析真实函数指针）。

---

## 8. 代码质量改进

| 改进项 | 详情 |
|--------|------|
| 线程安全 | `pthread_equal` 替代 `==`, `pthread_t` 替代 `int` |
| 内存安全 | `CUDA_OVERRIDE_CALL` 增加 NULL 检查日志 |
| 边界检查 | `cuMemGetInfo_v2` limit > total 的修复 |
| 可移植性 | 移除 `_dl_sym` (glibc 内部符号), 用 `libc dlopen`; 多 glibc 版本支持 |
| OOM 安全 | `cuMemHostAlloc`/`cuMemHostRegister_v2` OOM 时自动释放已分配资源 |
| 初始化顺序 | `cuGetExportTable` 特殊处理 (CUDA 12.8+ 兼容), `ensure_initialized` 在 NVML 初始化前 |

---

## 9. 关键选择与权衡

### NIO 分支优势
- **更小攻击面**: hook 表只有 ~16 条目, 减少 hook 相关 bug 的可能性
- **更简单调试**: 较少的 hook 层意味着问题定位更直接
- **特定场景优化**: 注释掉 kernel launch hook 可减少 CPU 开销

### Volcano 分支优势
- **完整 GPU 资源控制**: 所有分配路径都有 OOM 检查和追踪
- **Rate Limiter 完整**: kernel launch hook + rate limiter 实现 SM 利用率限制
- **cuGetProcAddress 性能**: Trie 优化使 hook 开销降低 97%
- **生产级健壮性**: flock 锁、超时保护、多 glibc 版本、CUDA 12.8+ 兼容
- **多架构支持**: arm64 glibc 版本支持

### 风险评估
- Volcano 的 hook 表更大 (~57 条目), 新增 CUDA API 版本可能需要同步更新 `cuda_library_entry[]`
- Volcano 的 `cu_hook_lookup` 新增函数时需要在 switch-case 中追加 branch (编译时校验)
- NIO 注释掉 kernel launch 意味着 `CUDA_DEVICE_SM_LIMIT` 不生效
