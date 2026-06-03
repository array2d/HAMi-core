#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"

extern size_t context_size;
extern int ctx_activate[16];

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev){
    LOG_INFO("dev=%d context_size=%ld",dev,context_size);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxRetain,pctx,dev);
    if (ctx_activate[dev] == 0) {
        add_gpu_device_memory_usage(getpid(),dev,context_size,0);
    }
    if (context_size>0) {
        ctx_activate[dev] = 1;
    }
    return res;
}

CUresult cuDevicePrimaryCtxRelease_v2( CUdevice dev ){
    if (ctx_activate[dev] == 1) {
        rm_gpu_device_memory_usage(getpid(),dev,context_size,0);
    }
    ctx_activate[dev] = 0;
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDevicePrimaryCtxRelease_v2,dev);
    return res;
}
