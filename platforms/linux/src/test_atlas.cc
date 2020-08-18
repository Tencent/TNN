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
#include "tnn/core/instance.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/mat_utils.h"

using namespace TNN_NS;
TNN net_;

int main(int argc, char* argv[]) {
    printf("Run Atlas test ...\n");
    if (argc == 1) {
        printf("./AtlasTest <om_file> <input_filename>\n");
        return 0;
    } else {
        if (argc < 3) {
            printf("invalid args\n");
            return 0;
        }
        for (int i = 1; i < argc; i++) {
            printf("arg%d: %s\n", i - 1, argv[i]);
        }
    }

    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    struct timeval time_begin, time_end;
    float delta = 0;

    Status error;
    int ret;
    gettimeofday(&time1, NULL);
    ModelConfig config;
    config.model_type = MODEL_TYPE_ATLAS;
    config.params.push_back(argv[1]);

    error = net_.Init(config);  // init the net
    if (TNN_OK != error) {
        printf("TNN init failed\n");
        return -1;
    }
    gettimeofday(&time2, NULL);
    delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
    printf("init tnn time cost: %g ms\n", delta);

    TNNParam run_param;
    run_param.input_file = argv[2];
    run_param.device_id  = 0;
    run_param.tnn_net    = &net_;

    RunTNN(&run_param);

    net_.DeInit();
    printf("network deinit done!\n");
    return 0;
}
