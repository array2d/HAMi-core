#include "include/libcuda_hook.h"
#include <string.h>
#include "include/libvgpu.h"
#include "include/multi_func_hook.h"


typedef void* (*fp_dlsym)(void*, const char*);
extern fp_dlsym real_dlsym;

extern CUresult cuMemoryAllocate(CUdeviceptr* dptr, size_t bytesize, void* data);
extern CUresult cuMemoryFree(CUdeviceptr dptr);

cuda_entry_t cuda_library_entry[] = {
    /* Init Part */
    {.name = "cuInit"},
    /* Device Part — only functions that modify driver behavior */
    {.name = "cuDeviceTotalMem_v2"},
    {.name = "cuDriverGetVersion"},
    {.name = "cuDeviceGetCount"},
    /* Context Part — only functions that track context memory */
    {.name = "cuDevicePrimaryCtxRetain"},
    {.name = "cuDevicePrimaryCtxRelease_v2"},
    {.name = "cuGetExportTable"},
    /* Memory Part — functions with allocation tracking / OOM check / behavior override */
    {.name = "cuArray3DCreate_v2"},
    {.name = "cuArrayCreate_v2"},
    {.name = "cuArrayDestroy"},
    {.name = "cuMemAlloc_v2"},
    {.name = "cuMemAllocHost_v2"},
    {.name = "cuMemAllocManaged"},
    {.name = "cuMemAllocPitch_v2"},
    {.name = "cuMemFree_v2"},
    {.name = "cuMemFreeHost"},
    {.name = "cuMemHostAlloc"},
    {.name = "cuMemHostRegister_v2"},
    {.name = "cuMemHostUnregister"},
    {.name = "cuMipmappedArrayCreate"},
    {.name = "cuMipmappedArrayDestroy"},
    {.name = "cuMemGetInfo_v2"},
    {.name = "cuPointerGetAttributes"},
    /* Virtual Memory Part */
    {.name = "cuMemAddressReserve"},
    {.name = "cuMemCreate"},
    {.name = "cuMemRelease"},
    {.name = "cuMemMap"},
    {.name = "cuMemImportFromShareableHandle"},
    {.name = "cuMemAllocAsync"},
    {.name = "cuMemFreeAsync"},
    /* Memory Pool Part */
    {.name = "cuMemPoolTrimTo"},
    {.name = "cuMemPoolSetAttribute"},
    {.name = "cuMemPoolGetAttribute"},
    {.name = "cuMemPoolSetAccess"},
    {.name = "cuMemPoolGetAccess"},
    {.name = "cuMemPoolCreate"},
    {.name = "cuMemPoolDestroy"},
    {.name = "cuMemAllocFromPoolAsync"},
    {.name = "cuMemPoolExportToShareableHandle"},
    {.name = "cuMemPoolImportFromShareableHandle"},
    {.name = "cuMemPoolExportPointer"},
    {.name = "cuMemPoolImportPointer"},
    {.name = "cuDeviceGetMemPool"},
    /* Kernel Launch — rate limiter */
    {.name = "cuLaunchKernel"},
    {.name = "cuLaunchKernelEx"},
    {.name = "cuLaunchCooperativeKernel"},
    /* External resource interop */
    {.name = "cuImportExternalMemory"},
    {.name = "cuExternalMemoryGetMappedBuffer"},
    {.name = "cuExternalMemoryGetMappedMipmappedArray"},
    {.name = "cuDestroyExternalMemory"},
    {.name = "cuImportExternalSemaphore"},
    {.name = "cuSignalExternalSemaphoresAsync"},
    {.name = "cuWaitExternalSemaphoresAsync"},
    {.name = "cuDestroyExternalSemaphore"},
    /* Proc address — self-referencing hook */
    {.name = "cuGetProcAddress"},
    {.name = "cuGetProcAddress_v2"},
};

int prior_function(char tmp[500]) {
    char *pos = tmp + strlen(tmp) - 3;
    if (pos[0]=='_' && pos[1]=='v') {
        if (pos[2]=='2')
            pos[0]='\0';
        else
            pos[2]--;
        return 1;
    }
    return 0;
}

void load_cuda_libraries() {
    void *table = NULL;
    int i = 0;
    char cuda_filename[FILENAME_MAX];
    char tmpfunc[500];

    LOG_INFO("Start hijacking");

    snprintf(cuda_filename, FILENAME_MAX - 1, "%s","libcuda.so.1");
    cuda_filename[FILENAME_MAX - 1] = '\0';

    table = dlopen(cuda_filename, RTLD_NOW | RTLD_NODELETE);
    if (!table) {
        LOG_WARN("can't find library %s", cuda_filename);
    }

    for (i = 0; i < CUDA_ENTRY_END; i++) {
        LOG_DEBUG("LOADING %s %d",cuda_library_entry[i].name,i);
        cuda_library_entry[i].fn_ptr = real_dlsym(table, cuda_library_entry[i].name);
        if (!cuda_library_entry[i].fn_ptr) {
            cuda_library_entry[i].fn_ptr=real_dlsym(RTLD_NEXT,cuda_library_entry[i].name);
            if (!cuda_library_entry[i].fn_ptr){
                LOG_INFO("can't find function %s in %s", cuda_library_entry[i].name,cuda_filename);
                memset(tmpfunc,0,500);
                strcpy(tmpfunc,cuda_library_entry[i].name);
                while (prior_function(tmpfunc)) {
                    cuda_library_entry[i].fn_ptr=real_dlsym(RTLD_NEXT,tmpfunc);
                    if (cuda_library_entry[i].fn_ptr) {
                        LOG_INFO("found prior function %s",tmpfunc);
                        break;
                    } 
                }
            }
        }
    }
    LOG_INFO("loaded_cuda_libraries");
    if (cuda_library_entry[0].fn_ptr==NULL){
        LOG_WARN("is NULL");
    }
    dlclose(table);
}


/*
 * O(1) function lookup via character dispatch — no strcmp, no loops.
 * Falls through to real driver for functions without local overrides.
 */
void *cu_hook_lookup(const char *symbol) {
    if (symbol[0] != 'c' || symbol[1] != 'u')
        return NULL;

    switch (symbol[2]) {
    case 'D':  /* Device / Driver */
        if (symbol[3] == 'e') {           /* Device* */
            if (symbol[7] == 'T') {       /* cuDeviceTotalMem_v2 */
                return (void*)cuDeviceTotalMem_v2;
            }
            if (symbol[7] == 'P') {       /* cuDevicePrimaryCtx* */
                /* symbol[20]='l'=Release, 't'=Retain */
                return (symbol[20] == 'l')
                    ? (void*)cuDevicePrimaryCtxRelease_v2
                    : (void*)cuDevicePrimaryCtxRetain;
            }
            return NULL;
        }
        if (symbol[3] == 'r')             /* cuDriverGetVersion */
            return (void*)cuDriverGetVersion;
        return NULL;

    case 'G':  /* GetProcAddress */
        if (symbol[3] == 'e' && symbol[8] == 'c') {  /* cuGetProcAddress */
            return (symbol[12] == '_')
                ? (void*)cuGetProcAddress_v2
                : (void*)cuGetProcAddress;
        }
        return NULL;

    case 'I':  /* Init */
        return (symbol[5] == '\0' || symbol[5] == '_')
            ? (void*)cuInit : NULL;

    case 'L':  /* Launch */
        if (symbol[3] == 'a' && symbol[7] == 'C')   /* cuLaunchCooperativeKernel */
            return (void*)cuLaunchCooperativeKernel;
        if (symbol[3] == 'a' && symbol[7] == 'K') { /* cuLaunchKernel */
            return (symbol[10] == 'E')               /* cuLaunchKernelEx */
                ? (void*)cuLaunchKernelEx
                : (void*)cuLaunchKernel;
        }
        return NULL;

    case 'M':  /* Mem / Mipmapped / Memory */
        if (symbol[3] == 'e') {           /* cuMem* */
            switch (symbol[4]) {
            case 'A':  /* cuMemAlloc* */
                if (symbol[9] == '\0' || symbol[9] == '_')
                    return (void*)cuMemAlloc_v2;       /* cuMemAlloc[_v2] */
                if (symbol[9] == 'H')
                    return (void*)cuMemAllocHost_v2;   /* cuMemAllocHost_v2 */
                if (symbol[9] == 'M')
                    return (void*)cuMemAllocManaged;    /* cuMemAllocManaged */
                if (symbol[9] == 'P')
                    return (void*)cuMemAllocPitch_v2;  /* cuMemAllocPitch_v2 */
                if (symbol[9] == 'A')
                    return (void*)cuMemAllocAsync;     /* cuMemAllocAsync */
                return NULL;
            case 'C':  /* cuMemCreate */
                return (void*)cuMemCreate;
            case 'F':  /* cuMemFree* */
                if (symbol[6] == 'A')
                    return (void*)cuMemFreeAsync;      /* cuMemFreeAsync */
                if (symbol[6] == 'v' || symbol[6] == '\0')
                    return (void*)cuMemFree_v2;        /* cuMemFree[_v2] */
                return NULL;
            case 'G':  /* cuMemGetInfo_v2 */
#ifdef HOOK_MEMINFO_ENABLE
                return (void*)cuMemGetInfo_v2;
#else
                return NULL;
#endif
            case 'H':  /* cuMemHost* */
                if (symbol[6] == 'A')
                    return (void*)cuMemHostAlloc;      /* cuMemHostAlloc */
                if (symbol[6] == 'R')
                    return (void*)cuMemHostRegister_v2; /* cuMemHostRegister_v2 */
                return NULL;
            case 'R':  /* cuMemRelease */
                return (void*)cuMemRelease;
            }
            return NULL;
        }
        if (symbol[3] == 'i') {           /* cuMipmapped* */
            return (symbol[11] == 'C')
                ? (void*)cuMipmappedArrayCreate   /* cuMipmappedArrayCreate */
                : NULL;
        }
        if (symbol[3] == 'o') {           /* cuMemory* */
            return (symbol[9] == 'A')
                ? (void*)cuMemoryAllocate          /* cuMemoryAllocate */
                : (void*)cuMemoryFree;             /* cuMemoryFree */
        }
        return NULL;

    case 'P':  /* Pointer */
        return (void*)cuPointerGetAttributes;

    default:
        return NULL;
    }
}

/* find_symbols_in_table_by_cudaversion: multi-version-aware lookup.
 * Tries _v3 → _v2 → original name via cu_hook_lookup. */
static const char* get_real_func_name(const char* base_name, int cuda_version) {
    int i;
    for (i = 0; i < sizeof(g_func_map) / sizeof(g_func_map[0]); ++i) {
        CudaFuncMapEntry *entry = &g_func_map[i];
        if (strcmp(entry->func_name, base_name) != 0) continue;
        if (cuda_version >= entry->min_ver && cuda_version <= entry->max_ver)
            return entry->real_name;
    }
    return NULL;
}

void *find_symbols_in_table(const char *symbol) {
    char buf[500];
    void *pfn;

    /* cuGraph* functions are never hooked — fast bail-out */
    if (symbol[2] == 'G' && symbol[3] == 'r')
        return NULL;

    /* Try _v3 suffix */
    strcpy(buf, symbol);
    strcat(buf, "_v3");
    pfn = cu_hook_lookup(buf);
    if (pfn) return pfn;

    /* Try _v2 suffix */
    buf[strlen(buf) - 1] = '2';
    pfn = cu_hook_lookup(buf);
    if (pfn) return pfn;

    /* Try original name */
    return cu_hook_lookup(symbol);
}

void *find_symbols_in_table_by_cudaversion(const char *symbol, int cudaVersion) {
    const char *real_symbol = get_real_func_name(symbol, cudaVersion);
    if (real_symbol == NULL)
        return find_symbols_in_table(symbol);
    return find_symbols_in_table(real_symbol);
}


CUresult (*cuGetProcAddress_real)(const char*, void**, int, cuuint64_t);

CUresult _cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
    LOG_INFO("into _cuGetProcAddress symbol=%s:%d", symbol, cudaVersion);
    *pfn = find_symbols_in_table_by_cudaversion(symbol, cudaVersion);
    if (*pfn == NULL) {
        return CUDA_OVERRIDE_CALL(cuda_library_entry, cuGetProcAddress, symbol, pfn, cudaVersion, flags);
    }
    LOG_DEBUG("found symbol %s", symbol);
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
    LOG_INFO("into cuGetProcAddress symbol=%s:%d", symbol, cudaVersion);
    *pfn = find_symbols_in_table_by_cudaversion(symbol, cudaVersion);
    if (strcmp(symbol, "cuGetProcAddress") == 0) {
        CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGetProcAddress, symbol, pfn, cudaVersion, flags);
        if (res == CUDA_SUCCESS) {
            cuGetProcAddress_real = *pfn;
            *pfn = _cuGetProcAddress;
        }
        return res;
    }
    if (*pfn == NULL) {
        return CUDA_OVERRIDE_CALL(cuda_library_entry, cuGetProcAddress, symbol, pfn, cudaVersion, flags);
    }
    LOG_DEBUG("found symbol %s", symbol);
    return CUDA_SUCCESS;
}

CUresult _cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
    LOG_INFO("into _cuGetProcAddress_v2 symbol=%s:%d", symbol, cudaVersion);
    *pfn = find_symbols_in_table_by_cudaversion(symbol, cudaVersion);
    if (*pfn == NULL) {
        return CUDA_OVERRIDE_CALL(cuda_library_entry, cuGetProcAddress_v2, symbol, pfn, cudaVersion, flags, symbolStatus);
    }
    LOG_DEBUG("found symbol %s", symbol);
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
    LOG_INFO("into cuGetProcAddress_v2 symbol=%s:%d", symbol, cudaVersion);
    *pfn = find_symbols_in_table_by_cudaversion(symbol, cudaVersion);
    if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
        CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuGetProcAddress_v2, symbol, pfn, cudaVersion, flags, symbolStatus);
        if (res == CUDA_SUCCESS) {
            cuGetProcAddress_real = *pfn;
            *pfn = _cuGetProcAddress_v2;
        }
        return res;
    }
    if (*pfn == NULL) {
        return CUDA_OVERRIDE_CALL(cuda_library_entry, cuGetProcAddress_v2, symbol, pfn, cudaVersion, flags, symbolStatus);
    }
    LOG_DEBUG("found symbol %s", symbol);
    void *optr;
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuGetProcAddress_v2, symbol, &optr, cudaVersion, flags, symbolStatus);
}
