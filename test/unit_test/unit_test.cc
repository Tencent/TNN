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

#include <gtest/gtest.h>
#include <sys/time.h>
#include "test/flags.h"
#include "test/test_utils.h"
#include "test/unit_test/unit_test_common.h"

#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

void ShowUsage() {
    printf("    -dt \"<device type>\"  %s \n", device_type_message);
    printf("    -lp \"<dependent library path>\"  %s \n", library_path_message);
    printf("    -ic \"<number>\"        %s \n", iterations_count_message);
    printf("    -ub \"<bool>\"          %s \n", unit_test_benchmark_message);
    printf("    -th \"<bumber>\"        %s \n", cpu_thread_num_message);
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        ShowUsage();
        return false;
    }

    return true;
}

}  // namespace TNN_NS

GTEST_API_ int main(int argc, char **argv) {
    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    gettimeofday(&time1, &zone);

    int result = 0;
    try {
        ::testing::InitGoogleTest(&argc, argv);
        if (TNN_NS::ParseAndCheckCommandLine(argc, argv)) {
            LOGD("run unit for device type: %s \n", TNN_NS::FLAGS_dt.c_str());
            result = RUN_ALL_TESTS();
        }
    } catch (std::exception e) {
        LOGE("unit test catches an exception: %s \n", e.what());
    }

    gettimeofday(&time2, &zone);
    float cost_sec = (time2.tv_sec - time1.tv_sec) + (time2.tv_usec - time1.tv_usec) / 1000000.0;
    printf("=== Unit Test Cost: %f s ===\n", cost_sec);

    return result;
}
