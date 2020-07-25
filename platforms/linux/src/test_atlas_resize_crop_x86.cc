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

using namespace TNN_NS;
TNN net_;

int main(int argc, char* argv[]) {
    printf("Run Atlas test ...\n");
    if (argc == 1) {
        printf("./AtlasTest <config_filename> <input_filename>\n");
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

    BlobMap input_blobs;
    error       = instance_->GetAllInputBlobs(input_blobs);
    Blob* input = input_blobs.begin()->second;

    // load input
    unsigned char* input_data_ptr = nullptr;
    std::vector<int> input_dims   = {1,683,1024,3};
    std::vector<int> mid_dims     = {1,641,360,3};
    std::vector<int> output_dims  = {1,641,360,3};
    auto input_format             = input->GetBlobDesc().data_format;
    if (DATA_FORMAT_NCHW == input_format) {
        ret = ReadFromNchwtoNhwcU8FromTxt(input_data_ptr, argv[2], input_dims);
    } else if (DATA_FORMAT_NHWC == input_format) {
        ret = ReadFromNchwtoNhwcU8FromTxt(input_data_ptr, argv[2],
                                          {input_dims[0], input_dims[3], input_dims[1], input_dims[2]});
    } else {
        printf("invalid model input format\n");
        return -1;
    }
    if (CheckResult("load input data", ret) != true)
        return -1;
    int index = 10;
    printf("input_data_ptr[%d] = %f\n", index, (float)input_data_ptr[index]);

    Status tnn_ret;
    // copy input data into atlas
    Mat input_mat_org(DEVICE_NAIVE, N8UC3, {input_dims[0], input_dims[3], input_dims[1], input_dims[2]}, input_data_ptr);
    Mat input_mat(DEVICE_ATLAS, NNV12, {mid_dims[0], mid_dims[3], mid_dims[1], mid_dims[2]}, nullptr);
    Mat output_mat(DEVICE_ARM, NNV12, {output_dims[0], output_dims[3], output_dims[1], output_dims[2]}, nullptr);

    // resize
    printf("resize from %d x %d --->  %d x %d\n", input_mat_org.GetWidth(), input_mat_org.GetHeight(), input_mat.GetWidth(), input_mat.GetHeight());
    ResizeParam param_resize;
    param_resize.scale_w = 0.3;
    param_resize.scale_h = 0.3;
    tnn_ret = MatUtils::Resize(input_mat_org, input_mat, param_resize, command_queue);
    if (tnn_ret != TNN_OK) {
        printf("Mat Crop falied (%s)\n", tnn_ret.description().c_str());
        return -1;
    }

    // crop
    //printf("crop from %d x %d --->  %d x %d\n", input_mat_org.GetWidth(), input_mat_org.GetHeight(), input_mat.GetWidth(), input_mat.GetHeight());
    //CropParam param_crop;
    //param_crop.top_left_x = 0;
    //param_crop.top_left_y = 0;
    //param_crop.width = mid_dims[2];
    //param_crop.height = mid_dims[1];
    //tnn_ret = MatUtils::Crop(input_mat_org, input_mat, param_crop, command_queue);
    //if (tnn_ret != TNN_OK) {
    //    printf("Mat Crop falied (%s)\n", tnn_ret.description().c_str());
    //    return -1;
    //}

    printf("actual output:  %d x %d\n", input_mat.GetWidth(), input_mat.GetHeight());

    // resize
    printf("resize form %d x %d -->  %d x %d\n", input_mat.GetWidth(), input_mat.GetHeight(), output_mat.GetWidth(), output_mat.GetHeight());
    ResizeParam param_resize2;
    tnn_ret = MatUtils::Resize(input_mat, output_mat, param_resize2, command_queue);
    if (tnn_ret != TNN_OK) {
        printf("Mat Resize falied (%s)\n", tnn_ret.description().c_str());
        return -1;
    }

    printf("actual output:  %d x %d\n", output_mat.GetWidth(), output_mat.GetHeight());
    printf("actual output memory:  %d x %d\n", (output_mat.GetWidth() + 15) / 16 * 16, (output_mat.GetHeight() + 1) / 2 * 2);

    DumpDataToBin((char*)output_mat.GetData(), {1, 1, 1, (output_mat.GetWidth() + 15) / 16 * 16 * (output_mat.GetHeight() + 1) / 2 * 2 * 3 / 2}, "dump_data/output.bin");

    if (input_data_ptr != nullptr)
        free(input_data_ptr);

    instance_.reset();
    net_.DeInit();
    return 0;
}
