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

#include "tnn/core/instance.h"
#include "tnn/core/tnn.h"
#include "test_common.h"
#include "tnn/utils/dims_vector_utils.h"

using namespace TNN_NS;
TNN net_;

int InitTNN(std::string config_file) {
    std::ifstream file_in;
    file_in.open(config_file, std::ios::in);

    std::string line_str;
    int error_code = -1;
    while (std::getline(file_in, line_str)) {
        if (line_str.at(0) == '#')
            continue;

        ModelConfig config;
        config.model_type = MODEL_TYPE_ATLAS;
        config.params.push_back(line_str);

        error_code = net_.Init(config);  // init the net

        if (error_code != 0) {
            continue;
        } else {
            break;
        }
    }

    file_in.close();
    return error_code;
}

/* read input from text files */
int ReadFromNchwtoNhwcFromTxt(unsigned char*& img, std::string file_path,
                              std::vector<int> dims) {
    printf("read from txt file! (%s)\n", file_path.c_str());
    std::ifstream f(file_path);
    int dim_size = DimsVectorUtils::Count(dims, 1);
    printf("\tdim:[%d,%d,%d,%d]  size:%d\n", dims[0], dims[1], dims[2], dims[3],
           dim_size);

    img = (unsigned char*)malloc(dim_size);
    if (img == NULL) {
        printf("allocate memory failed!\n");
        return -1;
    }

    std::shared_ptr<unsigned char> img_org(
        new unsigned char[dim_size], [](unsigned char* p) { delete[] p; });

    float tmp = 0;
    for (int i = 0; i < dim_size; i++) {
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

    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    struct timeval time_begin, time_end;
    float delta = 0;

    gettimeofday(&time1, NULL);
    int ret = InitTNN(argv[1]);
    if (CheckResult("init tnn", ret) != true)
        return -1;
    gettimeofday(&time2, NULL);
    delta = (time2.tv_sec - time1.tv_sec) * 1000.0 +
            (time2.tv_usec - time1.tv_usec) / 1000.0;
    printf("init tnn time cost: %g ms\n", delta);

    gettimeofday(&time1, NULL);
    Status error;
    auto instance_ = net_.CreateInst(network_config, error);
    if (CheckResult("create instance", error) != true)
        return -1;
    gettimeofday(&time2, NULL);
    delta = (time2.tv_sec - time1.tv_sec) * 1000.0 +
            (time2.tv_usec - time1.tv_usec) / 1000.0;
    printf("tnn create instance time cost: %g ms\n", delta);

    InputShapesMap input_shapemap;
    error = instance_->Reshape(input_shapemap);

    BlobMap input_blobs, output_blobs;
    error       = instance_->GetAllInputBlobs(input_blobs);
    Blob* input = input_blobs.begin()->second;
    printf("input data shape [ %d %d %d %d ]\n", input->GetBlobDesc().dims[0],
           input->GetBlobDesc().dims[1], input->GetBlobDesc().dims[2],
           input->GetBlobDesc().dims[3]);
    instance_->GetAllOutputBlobs(output_blobs);

    for (auto it = output_blobs.begin(); it != output_blobs.end(); ++it) {
        printf("output(%s) data shape [ %d %d %d %d ]\n", it->first.c_str(),
               it->second->GetBlobDesc().dims[0],
               it->second->GetBlobDesc().dims[1],
               it->second->GetBlobDesc().dims[2],
               it->second->GetBlobDesc().dims[3]);
    }

    // load input
    unsigned char* input_data_ptr = nullptr;
    ret = ReadFromNchwtoNhwcFromTxt(input_data_ptr, argv[2],
                                    input->GetBlobDesc().dims);
    if (CheckResult("load input data", ret) != true)
        return -1;
    int index = 10;
    printf("input_data_ptr[%d] = %d\n", index, (int)input_data_ptr[index]);

    // copy data to atlas buffer
    int bytes_per_elem = 1;
    if (input->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        bytes_per_elem = 4;
    } else if (input->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        bytes_per_elem = 2;
    } else if (input->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        bytes_per_elem = 1;
    }
    int input_data_size =
        DimsVectorUtils::Count(input->GetBlobDesc().dims, 1) * bytes_per_elem;
    int offset = 0;
    for (int i = 0; i < input->GetBlobDesc().dims[0]; ++i) {
        memcpy((char*)input->GetHandle().base + offset, input_data_ptr,
               input_data_size);
        offset += input_data_size;
    }

    if (input_data_ptr != nullptr)
        free(input_data_ptr);

    // Forward on atlas device.
    // Also check the running time.
    srand(102);
    std::vector<float> costs;
    const int loopcnt = 10;
    gettimeofday(&time1, &zone);
    for (int i = 0; i < loopcnt; ++i) {
        gettimeofday(&time_begin, NULL);
        ret = instance_->Forward();
        gettimeofday(&time_end, NULL);
        costs.push_back((time_end.tv_sec - time_begin.tv_sec) * 1000.0 +
                        (time_end.tv_usec - time_begin.tv_usec) / 1000.0);
    }
    gettimeofday(&time2, &zone);
    delta = (time2.tv_sec - time1.tv_sec) * 1000.0 +
            (time2.tv_usec - time1.tv_usec) / 1000.0;
    printf("time cost: %g ms\n", delta / (float)loopcnt);
    DisplayStats("", costs);

    // copy data from atlas buffer
    // then dump to files
    for (auto output : output_blobs) {
        float* output_data_ptr       = nullptr;
        std::vector<int> output_dims = output.second->GetBlobDesc().dims;
        int output_data_size = DimsVectorUtils::Count(output_dims) * sizeof(float);
        output_data_ptr      = (float*)malloc(output_data_size);
        memcpy(output_data_ptr, output.second->GetHandle().base, output_data_size);
        DumpDataToTxt(output_data_ptr, output_dims, "../dump_data/dump_" + output.second->GetBlobDesc().name + ".txt");
        // DumpDataToBin(output_data_ptr, output_dims, "../dump_data/dump_" + output.second->GetBlobDesc().name + ".bin");
        // SpiltResult(output_data_ptr, outout_dims);

        if (output_data_ptr != nullptr)
            free(output_data_ptr);
    }
    return 0;
}
