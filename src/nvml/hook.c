#include <string.h>
#include <ctype.h>
#include <dlfcn.h>
#include "include/libnvml_hook.h"
#include "include/nvml-subset.h"
#include "include/utils.h"
#include "multiprocess/multiprocess_memory_limit.h"

entry_t nvml_library_entry[] = {
    {.name = "nvmlInit"},
    {.name = "nvmlInit_v2"},
    {.name = "nvmlInitWithFlags"},
    {.name = "nvmlDeviceGetMemoryInfo"},
    {.name = "nvmlDeviceGetMemoryInfo_v2"},
    {.name = "nvmlDeviceGetIndex"},
};

pthread_once_t init_virtual_map_pre_flag = PTHREAD_ONCE_INIT;
pthread_once_t init_virtual_map_post_flag = PTHREAD_ONCE_INIT;

typedef void* (*fp_dlsym)(void*, const char*);
extern fp_dlsym real_dlsym;

void load_nvml_libraries() {
    void *table = NULL;
    char driver_filename[FILENAME_MAX];

    if (real_dlsym == NULL) {
        real_dlsym = dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5");
        if (real_dlsym == NULL) {
            void *libc_handle = dlopen("libc.so.6", RTLD_LAZY);
            if (libc_handle) {
                real_dlsym = dlsym(libc_handle, "dlsym");
            }
            if (real_dlsym == NULL)
                LOG_ERROR("real dlsym not found");
        }
    }
    snprintf(driver_filename, FILENAME_MAX - 1, "%s", "libnvidia-ml.so.1");
    driver_filename[FILENAME_MAX - 1] = '\0';

    table = dlopen(driver_filename, RTLD_NOW | RTLD_NODELETE);
    if (!table) {
        LOG_WARN("can't find library %s", driver_filename);
    }
    int i;
    for (i = 0; i < NVML_ENTRY_END; i++) {
        LOG_DEBUG("loading %s:%d",nvml_library_entry[i].name,i);
        nvml_library_entry[i].fn_ptr = real_dlsym(table, nvml_library_entry[i].name);
        if (!nvml_library_entry[i].fn_ptr) {
            LOG_INFO("can't find function %s in %s", nvml_library_entry[i].name,
                driver_filename);
        }
    }
    LOG_INFO("loaded nvml libraries");
    dlclose(table);
}

void nvml_preInit() {
    ensure_initialized();
    load_env_from_file(ENV_OVERRIDE_FILE);
    load_nvml_libraries();
}

void nvml_postInit() {
    init_device_info();
}

nvmlReturn_t _nvmlDeviceGetMemoryInfo(nvmlDevice_t device, void* memory, int version) {
    LOG_DEBUG("into nvmlDeviceGetMemoryInfo");
    if (memory == NULL) {
        return NVML_SUCCESS;
    }
    unsigned int dev_id;

    switch (version) {
        case 1:
            CHECK_NVML_API(NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetMemoryInfo, device, memory));
            break;
        case 2:
            CHECK_NVML_API(NVML_OVERRIDE_CALL(nvml_library_entry,nvmlDeviceGetMemoryInfo_v2, device, (nvmlMemory_v2_t *)memory));
            break;
        default:
            return NVML_ERROR_INVALID_ARGUMENT;
    }
    CHECK_NVML_API(NVML_OVERRIDE_CALL(nvml_library_entry, nvmlDeviceGetIndex, device, &dev_id));
    size_t usage = get_current_device_memory_usage(dev_id);
    size_t monitor = get_current_device_memory_monitor(dev_id);
    size_t limit = get_current_device_memory_limit(dev_id);
    LOG_DEBUG("usage=%ld limit=%ld monitor=%ld", usage, limit, monitor);
    if (limit == 0) {
        switch (version) {
        case 1:
            ((nvmlMemory_t*)memory)->used = usage;
            return NVML_SUCCESS;
        case 2:
            ((nvmlMemory_v2_t *)memory)->used = usage;
            return NVML_SUCCESS;
        }
    } else {
        switch (version) {
        case 1:
            ((nvmlMemory_t*)memory)->free = (limit-usage);
            ((nvmlMemory_t*)memory)->total = limit;
            ((nvmlMemory_t*)memory)->used = usage;
            return NVML_SUCCESS;
        case 2:
            ((nvmlMemory_v2_t *)memory)->free = (limit-usage);
            ((nvmlMemory_v2_t *)memory)->total = limit;
            ((nvmlMemory_v2_t *)memory)->used = usage;
            return NVML_SUCCESS;
        }
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) {
    return _nvmlDeviceGetMemoryInfo(device,memory,1);
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory) {
    return _nvmlDeviceGetMemoryInfo(device,memory,2);
}

nvmlReturn_t nvmlInitWithFlags( unsigned int  flags ) {
    LOG_DEBUG("nvmlInitWithFlags")
    pthread_once(&init_virtual_map_pre_flag, (void(*) (void))nvml_preInit);
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry, nvmlInitWithFlags,flags);
    pthread_once(&init_virtual_map_post_flag,(void (*)(void))nvml_postInit);
    return res;
}

nvmlReturn_t nvmlInit(void) {
    LOG_DEBUG("nvmlInit")
    pthread_once(&init_virtual_map_pre_flag,(void (*)(void))nvml_preInit);
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry, nvmlInit_v2);
    pthread_once(&init_virtual_map_post_flag,(void (*)(void))nvml_postInit);
    return res;
}

nvmlReturn_t nvmlInit_v2(void) {
    LOG_DEBUG("nvmlInit_v2");
    pthread_once(&init_virtual_map_pre_flag,(void (*)(void))nvml_preInit);
    nvmlReturn_t res = NVML_OVERRIDE_CALL(nvml_library_entry, nvmlInit_v2);
    pthread_once(&init_virtual_map_post_flag,(void (*)(void))nvml_postInit);
    return res;
}
