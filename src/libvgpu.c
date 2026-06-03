//#include "memory_limit.h"
#include <fcntl.h>
#include <dlfcn.h>
#include <pthread.h>
#include "include/nvml_prefix.h"
#include <nvml.h>
#include "include/nvml_prefix.h"
#include "include/log_utils.h"
#include "include/libcuda_hook.h"
#include "include/libvgpu.h"
#include "include/utils.h"
#include "include/nvml_override.h"
#include "allocator/allocator.h"
#include "multiprocess/multiprocess_memory_limit.h"

extern void init_utilization_watcher(void);
extern void utilization_watcher(void);
extern void initial_virtual_map(void); 
extern int set_host_pid(int hostpid);
extern void allocator_init(void);
extern CUresult cuMemoryAllocate(CUdeviceptr* dptr, size_t bytesize, void* data);
extern CUresult cuMemoryFree(CUdeviceptr dptr);
void preInit();
char *(*real_realpath)(const char *path, char *resolved_path);
void *vgpulib;

pthread_once_t pre_cuinit_flag = PTHREAD_ONCE_INIT;
pthread_once_t post_cuinit_flag = PTHREAD_ONCE_INIT;
pthread_once_t dlsym_init_flag = PTHREAD_ONCE_INIT;

/* pidfound is to enable core utilization, if we don't find hostpid in container, then we have no
 where to find its core utilization */
extern int pidfound;

/* used to switch on/off the core utilization limitation*/
extern int env_utilization_switch;

/* context size for a certain task, we need to add it into device-memory usage*/
extern size_t context_size;

/* This is the symbol search function */
fp_dlsym real_dlsym = NULL;

pthread_mutex_t dlsym_lock;
typedef struct {
    pthread_t tid;
    void *pointer;
}tid_dl_map;

#define DLMAP_SIZE 100
tid_dl_map dlmap[DLMAP_SIZE];
int dlmap_count=0;

void init_dlsym(){
    LOG_DEBUG("init_dlsym\n");
    pthread_mutex_init(&dlsym_lock,NULL);
    dlmap_count=0;
    memset(dlmap, 0, sizeof(tid_dl_map)*DLMAP_SIZE);
}

int check_dlmap(pthread_t tid, void *pointer){
    int i;
    int cursor = (dlmap_count < DLMAP_SIZE) ? dlmap_count : DLMAP_SIZE;
    for (i=cursor-1; i>=0; i--) {
        if ((dlmap[i].pointer == pointer) && pthread_equal(dlmap[i].tid, tid))
            return 1;
    }
    cursor = dlmap_count % DLMAP_SIZE;
    dlmap[cursor].tid = tid;
    dlmap[cursor].pointer = pointer;
    dlmap_count++;
    return 0;
}

FUNC_ATTR_VISIBLE void* dlsym(void* handle, const char* symbol) {
    LOG_DEBUG("into dlsym %s",symbol);
    pthread_once(&dlsym_init_flag,init_dlsym);
    if (real_dlsym == NULL) {
        const char* glibc_versions[] = {
                "GLIBC_2.2.5",  // for amd64
                "GLIBC_2.17",   // for arm64
                "GLIBC_2.3",
                "GLIBC_2.4",
                "GLIBC_2.10",
                "GLIBC_2.18",
                "GLIBC_2.22",
                NULL
        };
        for (int i = 0; glibc_versions[i] != NULL; i++) {
            real_dlsym = dlvsym(RTLD_NEXT, "dlsym", glibc_versions[i]);
            if (real_dlsym != NULL) {
                LOG_DEBUG("found dlsym with version: %s", glibc_versions[i]);
                break;
            }
        }
        char *path_search=getenv("CUDA_REDIRECT");
        if ((path_search!=NULL) && (strlen(path_search)>0)){
            vgpulib = dlopen(path_search,RTLD_LAZY);
        }else{
            vgpulib = dlopen("/usr/local/vgpu/libvgpu.so",RTLD_LAZY);
        }
        if (real_dlsym == NULL) {
            LOG_ERROR("real dlsym not found");
            void *libc_handle = dlopen("libc.so.6", RTLD_LAZY);
            if (libc_handle) {
                real_dlsym = dlsym(libc_handle, "dlsym");
            }
            if (real_dlsym == NULL)
                LOG_ERROR("real dlsym not found after trying libc.so.6");
        }
    }
    if (handle == RTLD_NEXT) {
        void *h = real_dlsym(RTLD_NEXT,symbol);
        pthread_mutex_lock(&dlsym_lock);
        pthread_t tid = pthread_self();
        if (check_dlmap(tid,h)){
            LOG_WARN("recursive dlsym : %s\n",symbol);
            h = NULL;
        }
        pthread_mutex_unlock(&dlsym_lock);
        return h;
    }
    if (symbol[0] == 'c' && symbol[1] == 'u') {
        //Compatible with cuda 12.8+ fix
        if (strcmp(symbol,"cuGetExportTable")!=0)
            pthread_once(&pre_cuinit_flag,(void(*)(void))preInit);
        void *f = real_dlsym(vgpulib,symbol);
        if (f!=NULL)
            return f;
    }
    #ifdef HOOK_NVML_ENABLE
    if (symbol[0] == 'n' && symbol[1] == 'v' &&
          symbol[2] == 'm' && symbol[3] == 'l' ) {
        void* f = __dlsym_hook_section_nvml(handle, symbol);
        if (f != NULL) {
            return f;
        }
    }
#endif
    return real_dlsym(handle, symbol);
}

void* __dlsym_hook_section(void* handle, const char* symbol) {
    int it;
    for (it=0;it<CUDA_ENTRY_END;it++){
        if (strcmp(cuda_library_entry[it].name,symbol) == 0){
            if (cuda_library_entry[it].fn_ptr == NULL) {
                LOG_WARN("NEED TO RETURN NULL");
                return NULL;
            }else{
                break;
            }
        }
    }
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
#ifdef HOOK_MEMINFO_ENABLE
    DLSYM_HOOK_FUNC(cuMemGetInfo_v2);
#endif
    return NULL;
}

void* __dlsym_hook_section_nvml(void* handle, const char* symbol) {
    DLSYM_HOOK_FUNC(nvmlInit);
    DLSYM_HOOK_FUNC(nvmlInit_v2);
    DLSYM_HOOK_FUNC(nvmlInitWithFlags);
    DLSYM_HOOK_FUNC(nvmlDeviceGetMemoryInfo);
    DLSYM_HOOK_FUNC(nvmlDeviceGetMemoryInfo_v2);
    return NULL;
}

void preInit(){
    log_utils_init();
    LOG_MSG("Initializing.....");
    if (real_dlsym == NULL) {
        real_dlsym = dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5");
        if (real_dlsym == NULL) {
            LOG_ERROR("real dlsym not found");
            void *libc_handle = dlopen("libc.so.6", RTLD_LAZY);
            if (libc_handle) {
                real_dlsym = dlsym(libc_handle, "dlsym");
            } else {
                LOG_ERROR("real dlsym not found");
            }
        }
    }
    real_realpath = NULL;
    load_cuda_libraries();
    //nvmlInit();
    ENSURE_INITIALIZED();
}

void postInit(){
    allocator_init();
    map_cuda_visible_devices();

    // Use shared memory semaphore to serialize host PID detection
    // Returns 1 if lock acquired, 0 if timeout (skip detection)
    int lock_acquired = lock_postinit();
    nvmlReturn_t res = NVML_SUCCESS;

    if (lock_acquired) {
        // Lock acquired - safe to call set_task_pid()
        res = set_task_pid();
        unlock_postinit();
    } else {
        // Timeout - another process likely crashed holding the lock
        // Skip host PID detection for this process
        LOG_WARN("Skipped host PID detection due to lock timeout");
        res = NVML_ERROR_TIMEOUT;
    }

    LOG_MSG("Initialized");
    if (res != NVML_SUCCESS) {
        LOG_WARN("SET_TASK_PID FAILED - using container PID for accounting");
        pidfound = 0;
    } else {
        pidfound = 1;
    }

    //add_gpu_device_memory_usage(getpid(),0,context_size,0);
    env_utilization_switch = set_env_utilization_switch();
    init_utilization_watcher();
}

CUresult cuInit(unsigned int Flags){
    LOG_INFO("Into cuInit");
    pthread_once(&pre_cuinit_flag,(void(*)(void))preInit);
    ENSURE_INITIALIZED();
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuInit,Flags);
    if (res != CUDA_SUCCESS){
        LOG_ERROR("cuInit failed:%d",res);
        return res;
    }
    pthread_once(&post_cuinit_flag, (void(*) (void))postInit);
    return CUDA_SUCCESS;
}
