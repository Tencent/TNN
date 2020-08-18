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
#include "tnn/utils/mat_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

using namespace TNN_NS;
TNN net_;

int main(int argc, char* argv[]) {
    printf("Run Atlas test ...\n");
    if (argc < 4) {
        printf("./AtlasTest <om_file> <input_list_file> <output_file_name>\n");
        return 0;
    }
    for (int i = 1; i < argc; i++) {
        printf("arg%d: %s\n", i - 1, argv[i]);
    }

    Status tnn_ret;

    NetworkConfig network_config;
    network_config.network_type = NETWORK_TYPE_ATLAS;
    network_config.device_type  = DEVICE_ATLAS;
    network_config.device_id    = 0;

    ModelConfig config;
    config.model_type = MODEL_TYPE_ATLAS;
    config.params.push_back(argv[1]);

    tnn_ret = net_.Init(config);  // init the net
    if (TNN_OK != tnn_ret) {
        printf("TNN init failed\n");
        return -1;
    }

    auto instance_ = net_.CreateInst(network_config, tnn_ret);
    if (CheckResult("create instance", tnn_ret) != true) {
        printf("error info: %s\n", tnn_ret.description().c_str());
        return -1;
    }

    // Get command queue
    void* command_queue;
    instance_->GetCommandQueue(&command_queue);

    // Reshape
    BlobMap input_blobs_temp;
    tnn_ret = instance_->GetAllInputBlobs(input_blobs_temp);
    InputShapesMap input_shapemap;
    input_shapemap[input_blobs_temp.begin()->first]    = input_blobs_temp.begin()->second->GetBlobDesc().dims;
    input_shapemap[input_blobs_temp.begin()->first][0] = 1;
    tnn_ret                                            = instance_->Reshape(input_shapemap);
    if (TNN_OK != tnn_ret) {
        printf("TNN reshape failed\n");
        return -1;
    }

    // Get input/output blobs
    BlobMap input_blobs, output_blobs;
    tnn_ret     = instance_->GetAllInputBlobs(input_blobs);
    Blob* input = input_blobs.begin()->second;
    for (auto it = input_blobs.begin(); it != input_blobs.end(); ++it) {
        printf("input(%s) data shape [ %d %d %d %d ]\n", it->first.c_str(), it->second->GetBlobDesc().dims[0],
               it->second->GetBlobDesc().dims[1], it->second->GetBlobDesc().dims[2], it->second->GetBlobDesc().dims[3]);
    }
    instance_->GetAllOutputBlobs(output_blobs);

    for (auto it = output_blobs.begin(); it != output_blobs.end(); ++it) {
        printf("output(%s) data shape [ %d %d %d %d ]\n", it->first.c_str(), it->second->GetBlobDesc().dims[0],
               it->second->GetBlobDesc().dims[1], it->second->GetBlobDesc().dims[2], it->second->GetBlobDesc().dims[3]);
    }

    // BlobConvert
    std::shared_ptr<BlobConverter> input_cvt;
    std::map<std::string, std::shared_ptr<BlobConverter>> output_cvt_map;
    input_cvt.reset(new BlobConverter(input));
    for (auto item : output_blobs) {
        output_cvt_map[item.first].reset(new BlobConverter(item.second));
    }

    std::vector<std::string> input_list;
    input_list.clear();
    std::ifstream f_in(argv[2]);
    std::string line;
    while (getline(f_in, line)) {
        input_list.push_back(line);
    }

    std::ofstream f_out(argv[3]);

    for (auto input_file : input_list) {
        auto input_dims = input_blobs.begin()->second->GetBlobDesc().dims;
        int input_c     = input_dims[3];
        int input_h     = input_dims[1];
        int input_w     = input_dims[2];

        assert(input_c == 3);

        // load input from image
        int height, width, channel;
        printf("load input %s\n", input_file.c_str());
        unsigned char* input_data_ptr = stbi_load(input_file.c_str(), &width, &height, &channel, input_c);
        if (nullptr == input_data_ptr) {
            printf("invalid input file: %s\n", input_file.c_str());
            continue;
        }

        if (input_h != height || input_w != width) {
            printf("resize from %dx%dx%d to %dx%dx%d\n", channel, height, width, input_c, input_h, input_w);
            unsigned char* input_resized = (unsigned char*)malloc(DimsVectorUtils::Count(input_dims));
            int ret = stbir_resize_uint8(input_data_ptr, width, height, 0, input_resized, input_w, input_h, 0, input_c);
            if (ret == 0) {
                free(input_resized);
                printf("resize image falied!\n");
                continue;
            }

            stbi_image_free(input_data_ptr);
            input_data_ptr = input_resized;
        }

        // copy input data into atlas
        Mat input_mat(DEVICE_NAIVE, N8UC3, {1, input_c, input_h, input_w}, input_data_ptr);
        MatConvertParam input_param;
        input_param.scale           = {0.00392156862745, 0.00392156862745, 0.00392156862745, 0.00392156862745};
        input_param.reverse_channel = false;

        tnn_ret = input_cvt->ConvertFromMat(input_mat, input_param, command_queue);
        if (tnn_ret != TNN_OK) {
            printf("ConvertFromMat falied (%s)\n", tnn_ret.description().c_str());
            return -1;
        }

        // Forward on atlas device.
        tnn_ret = instance_->Forward();
        if (tnn_ret != TNN_OK) {
            printf("instance Forward falied (%s)\n", tnn_ret.description().c_str());
        }

        // copy data from atlas buffer, then dump to files
        for (auto output : output_blobs) {
            Mat output_mat(DEVICE_NAIVE, NCHW_FLOAT, output.second->GetBlobDesc().dims);
            MatConvertParam output_param;
            tnn_ret = output_cvt_map[output.first]->ConvertToMat(output_mat, output_param, command_queue);
            if (tnn_ret != TNN_OK) {
                printf("ConvertToMat falied (%s)\n", tnn_ret.description().c_str());
                continue;
            }

            float* output_ptr = (float*)output_mat.GetData();
            f_out << input_file << " ";
            int output_count = DimsVectorUtils::Count(output.second->GetBlobDesc().dims);
            for (int i = 0; i < output_count; ++i) {
                f_out << output_ptr[i] << " ";
            }
            f_out << "\n";
        }

        // free input data ptr
        stbi_image_free(input_data_ptr);
    }

    instance_.reset();
    net_.DeInit();
    return 0;
}
