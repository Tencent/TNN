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
#include "tnn/utils/dims_vector_utils.h"

#define INSTANCE_NUM 4

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

    Status error;
    int ret;
    ModelConfig config;
    config.model_type = MODEL_TYPE_ATLAS;
    config.params.push_back(argv[1]);

    error = net_.Init(config);  // init the net
    if (TNN_OK != error) {
        printf("TNN init failed\n");
        return -1;
    }

    TNNParam run_param[INSTANCE_NUM];
    for (int i = 0; i < INSTANCE_NUM; ++i) {
        run_param[i].input_file = argv[2];
        run_param[i].device_id  = 0;
        run_param[i].thread_id  = i;
        run_param[i].tnn_net    = &net_;
    }

    for (int i = 0; i < INSTANCE_NUM; ++i) {
        RunTNN(&run_param[i]);
    }

    net_.DeInit();
    return 0;
}
