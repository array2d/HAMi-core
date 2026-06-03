#ifndef __LIBCUDA_HOOK_H__
#define __LIBCUDA_HOOK_H__

#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include <cuda.h>
#include <pthread.h>
#include "include/log_utils.h"

typedef struct {
  void *fn_ptr;
  char *name;
} cuda_entry_t;

#define FILENAME_MAX 4096

#define CONTEXT_SIZE 104857600

typedef CUresult (*cuda_sym_t)();

#define CUDA_OVERRIDE_ENUM(x) OVERRIDE_##x

#define CUDA_FIND_ENTRY(table, sym) ({ (table)[CUDA_OVERRIDE_ENUM(sym)].fn_ptr; })

#define CUDA_OVERRIDE_CALL(table, sym, ...)                                    \
  ({    \
    LOG_DEBUG("Hijacking %s", #sym);                                           \
    cuda_sym_t _entry = (cuda_sym_t)CUDA_FIND_ENTRY(table, sym);               \
    if (_entry == NULL) {                                                      \
      LOG_ERROR("Hijack failed: %s is NULL", #sym);                            \
    }                                                                          \
    _entry(__VA_ARGS__);                                                       \
  })

typedef enum {
    /* Init Part */
    CUDA_OVERRIDE_ENUM(cuInit),
    /* Device Part */
    CUDA_OVERRIDE_ENUM(cuDeviceTotalMem_v2),
    CUDA_OVERRIDE_ENUM(cuDriverGetVersion),
    CUDA_OVERRIDE_ENUM(cuDeviceGetCount),
    /* Context Part */
    CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxRetain),
    CUDA_OVERRIDE_ENUM(cuDevicePrimaryCtxRelease_v2),
    CUDA_OVERRIDE_ENUM(cuGetExportTable),
    /* Memory Part */
    CUDA_OVERRIDE_ENUM(cuArray3DCreate_v2),
    CUDA_OVERRIDE_ENUM(cuArrayCreate_v2),
    CUDA_OVERRIDE_ENUM(cuArrayDestroy),
    CUDA_OVERRIDE_ENUM(cuMemAlloc_v2),
    CUDA_OVERRIDE_ENUM(cuMemAllocHost_v2),
    CUDA_OVERRIDE_ENUM(cuMemAllocManaged),
    CUDA_OVERRIDE_ENUM(cuMemAllocPitch_v2),
    CUDA_OVERRIDE_ENUM(cuMemFree_v2),
    CUDA_OVERRIDE_ENUM(cuMemFreeHost),
    CUDA_OVERRIDE_ENUM(cuMemHostAlloc),
    CUDA_OVERRIDE_ENUM(cuMemHostRegister_v2),
    CUDA_OVERRIDE_ENUM(cuMemHostUnregister),
    CUDA_OVERRIDE_ENUM(cuMipmappedArrayCreate),
    CUDA_OVERRIDE_ENUM(cuMipmappedArrayDestroy),
    CUDA_OVERRIDE_ENUM(cuMemGetInfo_v2),
    CUDA_OVERRIDE_ENUM(cuPointerGetAttributes),
    /* Virtual Memory Part */
    CUDA_OVERRIDE_ENUM(cuMemAddressReserve),
    CUDA_OVERRIDE_ENUM(cuMemCreate),
    CUDA_OVERRIDE_ENUM(cuMemRelease),
    CUDA_OVERRIDE_ENUM(cuMemMap),
    CUDA_OVERRIDE_ENUM(cuMemImportFromShareableHandle),
    CUDA_OVERRIDE_ENUM(cuMemAllocAsync),
    CUDA_OVERRIDE_ENUM(cuMemFreeAsync),
    /* Memory Pool Part */
    CUDA_OVERRIDE_ENUM(cuMemPoolTrimTo),
    CUDA_OVERRIDE_ENUM(cuMemPoolSetAttribute),
    CUDA_OVERRIDE_ENUM(cuMemPoolGetAttribute),
    CUDA_OVERRIDE_ENUM(cuMemPoolSetAccess),
    CUDA_OVERRIDE_ENUM(cuMemPoolGetAccess),
    CUDA_OVERRIDE_ENUM(cuMemPoolCreate),
    CUDA_OVERRIDE_ENUM(cuMemPoolDestroy),
    CUDA_OVERRIDE_ENUM(cuMemAllocFromPoolAsync),
    CUDA_OVERRIDE_ENUM(cuMemPoolExportToShareableHandle),
    CUDA_OVERRIDE_ENUM(cuMemPoolImportFromShareableHandle),
    CUDA_OVERRIDE_ENUM(cuMemPoolExportPointer),
    CUDA_OVERRIDE_ENUM(cuMemPoolImportPointer),
    CUDA_OVERRIDE_ENUM(cuDeviceGetMemPool),
    /* Kernel Launch */
    CUDA_OVERRIDE_ENUM(cuLaunchKernel),
    CUDA_OVERRIDE_ENUM(cuLaunchKernelEx),
    CUDA_OVERRIDE_ENUM(cuLaunchCooperativeKernel),
    /* External Resource Interop */
    CUDA_OVERRIDE_ENUM(cuImportExternalMemory),
    CUDA_OVERRIDE_ENUM(cuExternalMemoryGetMappedBuffer),
    CUDA_OVERRIDE_ENUM(cuExternalMemoryGetMappedMipmappedArray),
    CUDA_OVERRIDE_ENUM(cuDestroyExternalMemory),
    CUDA_OVERRIDE_ENUM(cuImportExternalSemaphore),
    CUDA_OVERRIDE_ENUM(cuSignalExternalSemaphoresAsync),
    CUDA_OVERRIDE_ENUM(cuWaitExternalSemaphoresAsync),
    CUDA_OVERRIDE_ENUM(cuDestroyExternalSemaphore),
    /* Proc Address */
    CUDA_OVERRIDE_ENUM(cuGetProcAddress),
    CUDA_OVERRIDE_ENUM(cuGetProcAddress_v2),
    CUDA_ENTRY_END
}cuda_override_enum_t;

extern cuda_entry_t cuda_library_entry[];

#endif

#undef cuGetProcAddress
CUresult cuGetProcAddress( const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags );
