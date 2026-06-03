#ifndef __LIBNVML_HOOK_H__
#define __LIBNVML_HOOK_H__

#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda.h>
#include <pthread.h>
#include "include/nvml-subset.h"
#include "include/log_utils.h"
#include "include/nvml_prefix.h"

#define FILENAME_MAX 4096

typedef nvmlReturn_t (*driver_sym_t)();

#define NVML_OVERRIDE_ENUM(x) OVERRIDE_##x

#define NVML_FIND_ENTRY(table, sym) ({ (table)[NVML_OVERRIDE_ENUM(sym)].fn_ptr; })

#define NVML_OVERRIDE_CALL(table, sym, ...)                                    \
  ({                                                                           \
    LOG_DEBUG("Hijacking %s", #sym);                                           \
    driver_sym_t _entry = NVML_FIND_ENTRY(table, sym);                         \
    _entry(__VA_ARGS__);                                                       \
  })

#define NVML_OVERRIDE_CALL_NO_LOG(table, sym, ...)                             \
  ({                                                                           \
    driver_sym_t _entry = NVML_FIND_ENTRY(table, sym);                         \
    _entry(__VA_ARGS__);                                                       \
  })

typedef enum {
  NVML_OVERRIDE_ENUM(nvmlInit),
  NVML_OVERRIDE_ENUM(nvmlInit_v2),
  NVML_OVERRIDE_ENUM(nvmlInitWithFlags),
  NVML_OVERRIDE_ENUM(nvmlDeviceGetMemoryInfo),
  NVML_OVERRIDE_ENUM(nvmlDeviceGetMemoryInfo_v2),
  NVML_OVERRIDE_ENUM(nvmlDeviceGetIndex),
  NVML_ENTRY_END
} NVML_OVERRIDE_ENUM_t;

#endif
