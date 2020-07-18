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

/* read input from text files */
int ReadFromNchwtoNhwcU8FromTxt(unsigned char*& img, std::string file_path, std::vector<int> dims) {
    printf("read from txt file! (%s)\n", file_path.c_str());
    std::ifstream f(file_path);
    int dim_size = DimsVectorUtils::Count(dims, 0);
    int chw_size = DimsVectorUtils::Count(dims, 1);
    printf("\tdim:[%d,%d,%d,%d]  size:%d\n", dims[0], dims[1], dims[2], dims[3], dim_size);

    img = (unsigned char*)malloc(dim_size);
    if (img == NULL) {
        printf("allocate memory failed!\n");
        return -1;
    }

    std::shared_ptr<unsigned char> img_org(new unsigned char[chw_size], [](unsigned char* p) { delete[] p; });

    float tmp = 0;
    for (int i = 0; i < chw_size; i++) {
        f >> tmp;
        *(img_org.get() + i) = (unsigned char)tmp;
    }

    int channel = dims[1];
    int height  = dims[2];
    int width   = dims[3];
    for (int c = 0; c < channel; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int src_idx  = c * height * width + h * width + w;
                int dst_idx  = h * width * channel + w * channel + c;
                img[dst_idx] = *(img_org.get() + src_idx);
            }
        }
    }

    int offset = chw_size;
    for (int n = 1; n < dims[0]; ++n) {
        memcpy(img + offset, img, chw_size);
        offset += chw_size;
    }

    f.close();
    return 0;
}

// Read input data from text files and copy to multi batch.
int ReadFromTxtToBatch(float*& img, std::string file_path, std::vector<int> dims, bool nchw_to_nhwc) {
    printf("read from txt file! (%s)\n", file_path.c_str());
    std::ifstream f(file_path);
    int dim_size = TNN_NS::DimsVectorUtils::Count(dims);
    printf("\tdim:[%d,%d,%d,%d]  size:%d\n", dims[0], dims[1], dims[2], dims[3], dim_size);

    img = (float*)malloc(dim_size * sizeof(float));
    if (img == NULL) {
        printf("allocate memory failed!\n");
        return -1;
    }
    printf("allocate input memory size: %lu   addr: 0x%lx\n", dim_size * sizeof(float), (unsigned long)img);

    int N   = dims[0];
    int C   = dims[1];
    int H   = dims[2];
    int W   = dims[3];
    int chw = C * H * W;

    if (nchw_to_nhwc) {
        // convert from nchw to nhwc
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int idx = h * W * C + w * C + c;
                    f >> img[idx];
                    img[idx] = img[idx] / 255.0;
                }
            }
        }
    } else {
        for (int i = 0; i < chw; i++) {
            f >> img[i];
            img[i] = img[i] / 255.0;
        }
    }

    int offset = chw;
    for (int n = 1; n < N; ++n) {
        memcpy(img + offset, img, chw * sizeof(float));
        offset += chw;
    }

    f.close();
    return 0;
}

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
    // float* input_data_ptr = nullptr;
    unsigned char* input_data_ptr = nullptr;
    //auto input_dims             = input->GetBlobDesc().dims;
    std::vector<int> input_dims   = {1,683,1024,3};
    std::vector<int> mid_dims     = {1,641,360,3};
    std::vector<int> output_dims  = {1,641,360,3};
    auto input_format             = input->GetBlobDesc().data_format;
    if (DATA_FORMAT_NCHW == input_format) {
        // ret = ReadFromTxtToBatch(input_data_ptr, argv[2], input_dims, false);
        ret = ReadFromNchwtoNhwcU8FromTxt(input_data_ptr, argv[2], input_dims);
    } else if (DATA_FORMAT_NHWC == input_format) {
        // ret = ReadFromTxtToBatch(input_data_ptr, argv[2], {input_dims[0], input_dims[3], input_dims[1],
        // input_dims[2]}, false);
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
    // Mat input_mat(DEVICE_NAIVE, NCHW_FLOAT, input->GetBlobDesc().dims, input_data_ptr);
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
