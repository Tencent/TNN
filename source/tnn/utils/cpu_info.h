// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef TNN_SOURCE_TNN_UTILS_CPU_INFO_H_
#define TNN_SOURCE_TNN_UTILS_CPU_INFO_H_

#if defined(__ANDROID__) || defined(__linux__)

#include <stdint.h>

/* No hard limit in the kernel, maximum length observed on non-rogue kernels is 64 */
#define CPUINFO_HARDWARE_VALUE_MAX 64
/* As per include/sys/system_properties.h in Android NDK */
#define CPUINFO_BUILD_PROP_VALUE_MAX 92

struct cpuinfo_android_properties {
    char proc_cpuinfo_hardware[CPUINFO_HARDWARE_VALUE_MAX];
    char ro_product_board[CPUINFO_BUILD_PROP_VALUE_MAX];
    char ro_board_platform[CPUINFO_BUILD_PROP_VALUE_MAX];
    char ro_mediatek_platform[CPUINFO_BUILD_PROP_VALUE_MAX];
    char ro_arch[CPUINFO_BUILD_PROP_VALUE_MAX];
    char ro_chipname[CPUINFO_BUILD_PROP_VALUE_MAX];
    char ro_hardware_chipname[CPUINFO_BUILD_PROP_VALUE_MAX];
};

enum cpuinfo_arm_chipset_vendor {
    cpuinfo_arm_chipset_vendor_unknown = 0,
    cpuinfo_arm_chipset_vendor_samsung,
};

enum cpuinfo_arm_chipset_series {
    cpuinfo_arm_chipset_series_unknown = 0,
    cpuinfo_arm_chipset_series_samsung_exynos,
};

struct cpuinfo_arm_chipset {
    enum cpuinfo_arm_chipset_vendor vendor;
    enum cpuinfo_arm_chipset_series series;
    uint32_t model;
};

#define CPUINFO_ARM_MIDR_IMPLEMENTER_MASK uint32_t(0xFF000000)
#define CPUINFO_ARM_MIDR_VARIANT_MASK uint32_t(0x00F00000)
#define CPUINFO_ARM_MIDR_ARCHITECTURE_MASK uint32_t(0x000F0000)
#define CPUINFO_ARM_MIDR_PART_MASK uint32_t(0x0000FFF0)
#define CPUINFO_ARM_MIDR_REVISION_MASK uint32_t(0x0000000F)

#define CPUINFO_ARM_MIDR_IMPLEMENTER_OFFSET 24
#define CPUINFO_ARM_MIDR_VARIANT_OFFSET 20
#define CPUINFO_ARM_MIDR_ARCHITECTURE_OFFSET 16
#define CPUINFO_ARM_MIDR_PART_OFFSET 4
#define CPUINFO_ARM_MIDR_REVISION_OFFSET 0

struct cpuinfo_arm_linux_processor {
    /**
     * Main ID Register value.
     */
    uint32_t midr = 0;
};

bool cpuinfo_arm_linux_parse_proc_cpuinfo(char *hardware, struct cpuinfo_arm_linux_processor *processor);
#ifdef __ANDROID__
void cpuinfo_arm_android_parse_properties(struct cpuinfo_android_properties *properties);
#endif
struct cpuinfo_arm_chipset cpuinfo_arm_android_decode_chipset(const struct cpuinfo_android_properties *properties);

#endif  // __ANDROID__ || __linux__

#endif  // TNN_SOURCE_TNN_UTILS_CPU_INFO_H_
