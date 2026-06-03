#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"
#include "include/nvml_prefix.h"
#include "include/libnvml_hook.h"

#include "allocator/allocator.h"
#include "include/memory_limit.h"

CUresult cuDeviceTotalMem_v2 ( size_t* bytes, CUdevice dev ) {
    LOG_DEBUG("into cuDeviceTotalMem");
    ENSURE_INITIALIZED();
    size_t limit = get_current_device_memory_limit(dev);
    *bytes = limit;
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int *driverVersion) {
    LOG_DEBUG("into cuDriverGetVersion__");

    //stub dlsym to preload cuda functions
    dlsym(RTLD_DEFAULT,"cuDriverGetVersion");

    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDriverGetVersion,driverVersion);
    if ((res==CUDA_SUCCESS) && (driverVersion!=NULL)) {
        LOG_INFO("driver version=%d",*driverVersion);
    }
    return res;
}
