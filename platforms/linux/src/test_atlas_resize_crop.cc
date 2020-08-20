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

#include "test_common.h"
#include "tnn/core/instance.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/mat_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

using namespace TNN_NS;
TNN net_;

#define MID_WIDTH 608
#define MID_HEIGHT 352

int main(int argc, char* argv[]) {
    printf("Run Atlas test ...\n");
    if (argc == 1) {
        printf("./AtlasTest <om_file> <input_jpg>\n");
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

    NetworkConfig network_config;
    network_config.network_type = NETWORK_TYPE_ATLAS;
    network_config.device_type  = DEVICE_ATLAS;
    network_config.device_id    = 0;

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

    auto instance_ = net_.CreateInst(network_config, error);
    if (CheckResult("create instance", error) != true) {
        printf("error info: %s\n", error.description().c_str());
        return -1;
    }

    // Get command queue
    void* command_queue;
    instance_->GetCommandQueue(&command_queue);

    // load input from image
    int height, width, channel;
    printf("load input %s\n", argv[2]);
    unsigned char* input_data_ptr = stbi_load(argv[2], &width, &height, &channel, 0);
    if (nullptr == input_data_ptr) {
        printf("invalid input file: %s\n", argv[2]);
        return -1;
    }

    printf("input dims: c: %d  h: %d  w: %d\n", channel, height, width);
    std::vector<int> input_dims = {1, channel, height, width};
    std::vector<int> mid_dims   = {1, 3, MID_HEIGHT, MID_WIDTH};
    std::vector<int> dump_dims  = mid_dims;

    int index = 10;
    printf("input_data_ptr[%d] = %f\n", index, (float)input_data_ptr[index]);

    Status tnn_ret;
    // copy input data into atlas
    Mat input_mat_org(DEVICE_NAIVE, N8UC3, input_dims, input_data_ptr);
    Mat input_mat(DEVICE_ATLAS, NNV12, mid_dims, nullptr);
    Mat dump_mat(DEVICE_ARM, NNV12, dump_dims, nullptr);

    // resize
    printf("resize1 from (width x height) %d x %d --->  %d x %d\n", input_mat_org.GetWidth(), input_mat_org.GetHeight(),
           input_mat.GetWidth(), input_mat.GetHeight());
    ResizeParam param_resize;
    float scale_w = (float)MID_WIDTH / (float)width;
    float scale_h = (float)MID_HEIGHT / (float)height;
    float scale   = scale_w > scale_h ? scale_h : scale_w;
    param_resize.scale_w = scale;
    param_resize.scale_h = scale;
    PasteParam paste_param;
    paste_param.type      = PASTE_TYPE_CENTER_ALIGN;
    paste_param.pad_value = 128;
    tnn_ret               = MatUtils::ResizeAndPaste(input_mat_org, input_mat, param_resize, paste_param, command_queue);
    if (tnn_ret != TNN_OK) {
        printf("Mat Resize falied (%s)\n", tnn_ret.description().c_str());
        return -1;
    }

    // crop
    // printf("crop from %d x %d --->  %d x %d\n", input_mat_org.GetWidth(), input_mat_org.GetHeight(),
    // input_mat.GetWidth(), input_mat.GetHeight()); CropParam param_crop; param_crop.top_left_x = 0;
    // param_crop.top_left_x = 0;
    // param_crop.top_left_y = 0;
    // param_crop.width = mid_dims[2];
    // param_crop.height = mid_dims[1];
    // tnn_ret = MatUtils::Crop(input_mat_org, input_mat, param_crop, command_queue);
    // if (tnn_ret != TNN_OK) {
    //    printf("Mat Crop falied (%s)\n", tnn_ret.description().c_str());
    //    return -1;
    //}

    printf("actual output: (width x height)  %d x %d\n", input_mat.GetWidth(), input_mat.GetHeight());

    // resize to dump data to cpu
    printf("resize2 form (width x height) %d x %d --->  %d x %d\n", input_mat.GetWidth(), input_mat.GetHeight(), dump_mat.GetWidth(),
           dump_mat.GetHeight());
    ResizeParam param_resize2;
    param_resize2.scale_w = 1.0;
    param_resize2.scale_h = 1.0;
    PasteParam paste_param2;
    paste_param2.type      = PASTE_TYPE_CENTER_ALIGN;
    paste_param2.pad_value = 128;
    tnn_ret                = MatUtils::ResizeAndPaste(input_mat, dump_mat, param_resize2, paste_param2, command_queue);
    if (tnn_ret != TNN_OK) {
        printf("Mat Resize falied (%s)\n", tnn_ret.description().c_str());
        return -1;
    }

    printf("actual output: (width x height)  %d x %d\n", dump_mat.GetWidth(), dump_mat.GetHeight());

    DumpDataToBin((char*)dump_mat.GetData(), {1, 1, 1, dump_mat.GetWidth() * dump_mat.GetHeight() * 3 / 2},
                  "output.bin");

    stbi_image_free(input_data_ptr);

    instance_.reset();
    net_.DeInit();
    return 0;
}
