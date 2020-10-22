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

#include "tnn/utils/cpu_utils.h"
#include <stdio.h>
#include <string.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ANDROID__) || defined(__linux__)
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__ANDROID__)
#include <sys/auxv.h>
#define AT_HWCAP  16
#define AT_HWCAP2 26
// from arch/arm64/include/uapi/asm/hwcap.h
#define HWCAP_FPHP    (1 << 9)
#define HWCAP_ASIMDHP (1 << 10)
#endif  // __ANDROID__

#if defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#include <mach/machine.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#define __IOS__ 1
// A11
#ifndef CPUFAMILY_ARM_MONSOON_MISTRAL
#define CPUFAMILY_ARM_MONSOON_MISTRAL 0xe81e7ef6
#endif
// A12
#ifndef CPUFAMILY_ARM_VORTEX_TEMPEST
#define CPUFAMILY_ARM_VORTEX_TEMPEST 0x07d34b9f
#endif
// A13
#ifndef CPUFAMILY_ARM_LIGHTNING_THUNDER
#define CPUFAMILY_ARM_LIGHTNING_THUNDER 0x462504d2
#endif
#endif  // TARGET_OS_IPHONE
#endif  // __APPLE__

#include "tnn/core/macro.h"

namespace TNN_NS {

#ifdef __ANDROID__
static int GetMaxFreqOfCpu(int cpuid) {
    // first try, for all possible cpu
    char path[256];
    snprintf(path, 256, "/sys/devices/system/cpu/cpufreq/stats/cpu%d/time_in_state", cpuid);

    FILE* fp = fopen(path, "rb");

    if (!fp) {
        // second try, for online cpu
        snprintf(path, 256, "/sys/devices/system/cpu/cpu%d/cpufreq/stats/time_in_state", cpuid);
        fp = fopen(path, "rb");

        if (fp) {
            int max_freq_khz = 0;
            while (!feof(fp)) {
                int freq_khz = 0;
                int nscan = fscanf(fp, "%d %*d", &freq_khz);
                if (nscan != 1)
                    break;

                if (freq_khz > max_freq_khz)
                    max_freq_khz = freq_khz;
            }

            fclose(fp);

            if (max_freq_khz != 0)
                return max_freq_khz;

            fp = NULL;
        }

        if (!fp) {
            // third try, for online cpu
            snprintf(path, 256, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpuid);
            fp = fopen(path, "rb");

            if (!fp)
                return -1;

            int max_freq_khz = -1;
            fscanf(fp, "%d", &max_freq_khz);

            fclose(fp);

            return max_freq_khz;
        }
    }

    int max_freq_khz = 0;
    while (!feof(fp)) {
        int freq_khz = 0;
        int nscan = fscanf(fp, "%d %*d", &freq_khz);
        if (nscan != 1)
            break;

        if (freq_khz > max_freq_khz)
            max_freq_khz = freq_khz;
    }

    fclose(fp);

    return max_freq_khz;
}

static int GetCpuCount() {
    // get cpu count from /proc/cpuinfo
    FILE* fp = fopen("/proc/cpuinfo", "rb");
    if (!fp)
        return 1;

    int count = 0;
    char line[1024];
    while (!feof(fp)) {
        char* s = fgets(line, 1024, fp);
        if (!s)
            break;

        if (memcmp(line, "processor", 9) == 0) {
            count++;
        }
    }

    fclose(fp);

    if (count < 1)
        count = 1;

    return count;
}

static int SortCpuidByMaxFrequency(std::vector<int>& cpuids, int* little_cluster_offset) {
    const int cpu_count = cpuids.size();

    *little_cluster_offset = 0;

    if (cpu_count == 0)
        return 0;

    std::vector<int> cpu_max_freq_khz;
    cpu_max_freq_khz.resize(cpu_count);

    for (int i = 0; i < cpu_count; i++) {
        int max_freq_khz = GetMaxFreqOfCpu(i);

        //         printf("%d max freq = %d khz\n", i, max_freq_khz);

        cpuids[i]           = i;
        cpu_max_freq_khz[i] = max_freq_khz;
    }

    // sort cpuid as big core first
    // simple bubble sort
    for (int i = 0; i < cpu_count; i++) {
        for (int j = i + 1; j < cpu_count; j++) {
            if (cpu_max_freq_khz[i] < cpu_max_freq_khz[j]) {
                // swap
                int tmp   = cpuids[i];
                cpuids[i] = cpuids[j];
                cpuids[j] = tmp;

                tmp                 = cpu_max_freq_khz[i];
                cpu_max_freq_khz[i] = cpu_max_freq_khz[j];
                cpu_max_freq_khz[j] = tmp;
            }
        }
    }

    // SMP
    int mid_max_freq_khz = (cpu_max_freq_khz.front() + cpu_max_freq_khz.back()) / 2;
    if (mid_max_freq_khz == cpu_max_freq_khz.back())
        return 0;

    for (int i = 0; i < cpu_count; i++) {
        if (cpu_max_freq_khz[i] < mid_max_freq_khz) {
            *little_cluster_offset = i;
            break;
        }
    }

    return 0;
}
#endif  // __ANDROID__

static int SetSchedAffinity(const std::vector<int>& cpuids) {
#if defined(__ANDROID__) || defined(__linux__)
    // cpu_set_t definition
    // ref
    // http://stackoverflow.com/questions/16319725/android-set-thread-affinity
#define TNN_CPU_SETSIZE 1024
#define TNN_NCPUBITS (8 * sizeof(unsigned long))
    typedef struct {
        unsigned long __bits[TNN_CPU_SETSIZE / TNN_NCPUBITS];
    } cpu_set_t;

#define TNN_CPU_SET(cpu, cpusetp) ((cpusetp)->__bits[(cpu) / TNN_NCPUBITS] |= (1UL << ((cpu) % TNN_NCPUBITS)))

#define TNN_CPU_ZERO(cpusetp) memset((cpusetp), 0, sizeof(cpu_set_t))

    // set affinity for thread
#ifdef __GLIBC__
    pid_t pid = syscall(SYS_gettid);
#else
#ifdef PI3
    pid_t pid  = getpid();
#else
    pid_t pid = gettid();
#endif
#endif
    cpu_set_t mask;
    TNN_CPU_ZERO(&mask);
    for (int i = 0; i < (int)cpuids.size(); i++) {
        TNN_CPU_SET(cpuids[i], &mask);
    }

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallret) {
        fprintf(stderr, "syscall error %d\n", syscallret);
        return -1;
    }
#endif
    return 0;
}

Status CpuUtils::SetCpuPowersave(int powersave) {
#ifdef __ANDROID__
    static std::vector<int> sorted_cpuids;
    static int little_cluster_offset = 0;
    static int cpucount              = GetCpuCount();

    if (sorted_cpuids.empty()) {
        // 0 ~ g_cpucount

        sorted_cpuids.resize(cpucount);
        for (int i = 0; i < cpucount; i++) {
            sorted_cpuids[i] = i;
        }

        // descent sort by max frequency
        SortCpuidByMaxFrequency(sorted_cpuids, &little_cluster_offset);
    }

    if (little_cluster_offset == 0 && powersave != 0) {
        powersave = 0;
        fprintf(stderr, "SMP cpu powersave not supported\n");
    }

    // prepare affinity cpuid
    std::vector<int> cpuids;
    if (powersave == 0) {
        cpuids = sorted_cpuids;
    } else if (powersave == 1) {
        cpuids = std::vector<int>(sorted_cpuids.begin() + little_cluster_offset, sorted_cpuids.end());
    } else if (powersave == 2) {
        cpuids = std::vector<int>(sorted_cpuids.begin(), sorted_cpuids.begin() + little_cluster_offset);
    } else {
        fprintf(stderr, "powersave %d not supported\n", powersave);
        return TNNERR_SET_CPU_AFFINITY;
    }

#ifdef _OPENMP
    // set affinity for each thread
    int num_threads = cpuids.size();
    omp_set_num_threads(num_threads);
    std::vector<int> ssarets(num_threads, 0);
#pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        ssarets[i] = SetSchedAffinity(cpuids);
    }
    for (int i = 0; i < num_threads; i++) {
        if (ssarets[i] != 0) {
            return TNNERR_SET_CPU_AFFINITY;
        }
    }
#else
    int ssaret = SetSchedAffinity(cpuids);
    if (ssaret != 0) {
        return TNNERR_SET_CPU_AFFINITY;
    }
#endif

    return TNN_OK;
#else
    // TODO
    (void)powersave;  // Avoid unused parameter warning.
    return TNNERR_SET_CPU_AFFINITY;
#endif
}

Status CpuUtils::SetCpuAffinity(const std::vector<int>& cpu_list) {
#if defined(__ANDROID__) || defined(__linux__)
    if (0 != SetSchedAffinity(cpu_list)) {
        return TNNERR_SET_CPU_AFFINITY;
    }
    return TNN_OK;
#else
    return TNNERR_SET_CPU_AFFINITY;
#endif
}

bool CpuUtils::CpuSupportFp16() {
    bool fp16arith = false;

#ifdef __aarch64__

#ifdef __ANDROID__
    unsigned int hwcap = getauxval(AT_HWCAP);
    fp16arith = hwcap & HWCAP_FPHP &&
                hwcap & HWCAP_ASIMDHP;
#endif  // __ANDROID__

#ifdef __IOS__
    unsigned int cpu_family = 0;
    size_t len = sizeof(cpu_family);
    sysctlbyname("hw.cpufamily", &cpu_family, &len, NULL, 0);
    fp16arith = cpu_family == CPUFAMILY_ARM_MONSOON_MISTRAL ||
                cpu_family == CPUFAMILY_ARM_VORTEX_TEMPEST ||
                cpu_family == CPUFAMILY_ARM_LIGHTNING_THUNDER;
#endif  // __IOS__

#endif  // __aarch64__

    return fp16arith;
}

}  // namespace TNN_NS
