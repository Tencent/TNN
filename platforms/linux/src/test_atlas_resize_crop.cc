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

#define OUTPUT_WIDTH 300
#define OUTPUT_HEIGHT 300

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
    std::vector<int> input_dims  = {1, channel, height, width};
    std::vector<int> output_dims = {1, 3, OUTPUT_HEIGHT, OUTPUT_WIDTH};

    int index = 10;
    printf("input_data_ptr[%d] = %f\n", index, (float)input_data_ptr[index]);

    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    float delta = 0;

    Status tnn_ret;
    // copy input data into atlas
    Mat input_mat_org(DEVICE_NAIVE, N8UC3, input_dims, input_data_ptr);
    Mat input_mat_org_device(DEVICE_ATLAS, N8UC3, input_dims);
    Mat output_mat(DEVICE_ATLAS, NNV12, output_dims, nullptr);

    for (int i = 0; i < 5; ++i) {
        printf("\n-----------------------------\n");
        gettimeofday(&time1, nullptr);
        // copy from host to device
        tnn_ret = MatUtils::Copy(input_mat_org, input_mat_org_device, command_queue);
        if (tnn_ret != TNN_OK) {
            printf("Mat Copy falied (%s)\n", tnn_ret.description().c_str());
            return -1;
        }
        gettimeofday(&time2, nullptr);
        delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
        printf("copy from host to device cost: %g ms\n", delta);

        gettimeofday(&time1, nullptr);
        // resize
        printf("resize from (width x height) %d x %d --->  %d x %d\n", input_mat_org_device.GetWidth(),
                input_mat_org_device.GetHeight(), output_mat.GetWidth(), output_mat.GetHeight());
        ResizeParam param_resize;
        float scale_w        = (float)OUTPUT_WIDTH / (float)width;
        float scale_h        = (float)OUTPUT_HEIGHT / (float)height;
        float scale          = scale_w > scale_h ? scale_h : scale_w;
        param_resize.scale_w = scale;
        param_resize.scale_h = scale;
        //param_resize.scale_w = scale_w;
        //param_resize.scale_h = scale_h;
        PasteParam paste_param;
        paste_param.type      = PASTE_TYPE_CENTER_ALIGN;
        paste_param.pad_value = 128;
        tnn_ret = MatUtils::ResizeAndPaste(input_mat_org_device, output_mat, param_resize, paste_param, command_queue);
        if (tnn_ret != TNN_OK) {
            printf("Mat Resize falied (%s)\n", tnn_ret.description().c_str());
            return -1;
        }

        // crop
        // printf("crop from %d x %d --->  %d x %d\n", input_mat_org_device.GetWidth(), input_mat_org_device.GetHeight(),
        // output_mat.GetWidth(), output_mat.GetHeight()); CropParam param_crop; param_crop.top_left_x = 0;
        // param_crop.top_left_y = 0;
        // param_crop.width = output_mat.GetWidth();
        // param_crop.height = output_mat.GetHeight();
        // tnn_ret = MatUtils::Crop(input_mat_org_device, output_mat, param_crop, command_queue);
        // if (tnn_ret != TNN_OK) {
        //   printf("Mat Crop falied (%s)\n", tnn_ret.description().c_str());
        //   return -1;
        //}

        printf("actual output: (width x height)  %d x %d\n", output_mat.GetWidth(), output_mat.GetHeight());
        gettimeofday(&time2, nullptr);
        delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
        printf("mat utils (rezie/crop) cost: %g ms\n", delta);
        printf("-----------------------------\n\n");
    }

    gettimeofday(&time1, nullptr);
    // copy from device to cpu
    Mat dump_mat(DEVICE_ARM, NNV12, output_mat.GetDims());
    tnn_ret = MatUtils::Copy(output_mat, dump_mat, command_queue);
    if (tnn_ret != TNN_OK) {
        printf("Mat Copy falied (%s)\n", tnn_ret.description().c_str());
        return -1;
    }
    gettimeofday(&time2, nullptr);
    delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
    printf("copy from device to host cost: %g ms\n", delta);

    DumpDataToBin((char*)dump_mat.GetData(), {1, 1, 1, dump_mat.GetWidth() * dump_mat.GetHeight() * 3 / 2},
                  "output.bin");

    stbi_image_free(input_data_ptr);

    instance_.reset();
    net_.DeInit();
    return 0;
}
