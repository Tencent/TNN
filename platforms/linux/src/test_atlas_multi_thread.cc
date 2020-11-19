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

/*
 * This is a demo for the huawei atlas devices.
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <fstream>
#include <memory>
#include <string>

#include "atlas_common.h"
#include "test_common.h"
#include "memory_usage.h"
#include "tnn/core/instance.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/dims_vector_utils.h"

#define THREAD_NUM 100

using namespace TNN_NS;
TNN net_;

int main(int argc, char* argv[]) {
    printf("Run Atlas test ...\n");
    if (argc == 1) {
        printf("./AtlasTestMultiThread <om_file> <input_filename> <thread num> <loop num>\n");
        return 0;
    } else {
        if (argc < 5) {
            printf("invalid args\n");
            return 0;
        }
        for (int i = 1; i < argc; i++) {
            printf("arg%d: %s\n", i - 1, argv[i]);
        }
    }

    Status error;
    int ret;
    ModelConfig config;
    config.model_type = MODEL_TYPE_ATLAS;
    // use om file path
    config.params.push_back(argv[1]);
    // use om file content
    //{
    //    std::ifstream model_stream(argv[1], std::ios::binary);
    //    if (!model_stream.is_open() || !model_stream.good()) {
    //        printf("invalid argv[1]: %s\n", argv[1]);
    //        return -1;
    //    }
    //    auto model_content =
    //        std::string((std::istreambuf_iterator<char>(model_stream)), std::istreambuf_iterator<char>());
    //    config.params.push_back(model_content);
    //}

    error = net_.Init(config);  // init the net
    if (TNN_OK != error) {
        printf("TNN init failed\n");
        return -1;
    }

    TNNParam thread_param[THREAD_NUM];
    pthread_t thread[THREAD_NUM];

    int thread_num = atoi(argv[3]);
    long long loop_count = atoi(argv[4]);
    printf("thread num: %d  loop count %d\n", thread_num, loop_count);
    do {
        for (int i = 0; i < thread_num; ++i) {
            thread_param[i].input_file = argv[2];
            thread_param[i].device_id  = 0;
            thread_param[i].thread_id  = i;
            thread_param[i].tnn_net    = &net_;
        }

        for (int t = 0; t < thread_num; ++t) {
            if (pthread_create(&thread[t], NULL, &RunTNN, (void*)&thread_param[t]) != 0) {
                return -1;
            }
        }

        for (int t = 0; t < thread_num; t++) {
            pthread_join(thread[t], NULL);
        }

        loop_count--;
        PrintMemInfo();
    } while (loop_count > 0);

    net_.DeInit();
    return 0;
}
