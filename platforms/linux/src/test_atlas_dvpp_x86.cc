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
    Blob* output = output_blobs.begin()->second;

    // load input
    char* input_data_ptr = nullptr;
    int jpeg_size        = 0;
    ret                  = ReadFromJpeg(input_data_ptr, argv[2], jpeg_size);
    if (CheckResult("load input data", ret) != true)
        return -1;

    // copy data to atlas buffer
    memcpy(input->GetHandle().base, input_data_ptr, jpeg_size);

    // set bytes_offset
    TNN_NS::BlobHandle blob_handle;
    blob_handle.base         = input->GetHandle().base;
    blob_handle.bytes_offset = jpeg_size + 8;  // add 8 bytes to check
    input->SetHandle(blob_handle);

    if (input_data_ptr != nullptr)
        free(input_data_ptr);

    srand(102);
    std::vector<float> costs;
    const int loopcnt = 128;
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
    float* output_data_ptr       = nullptr;
    std::vector<int> output_dims = output->GetBlobDesc().dims;
    int output_data_size = DimsVectorUtils::Count(output_dims) * sizeof(float);
    output_data_ptr      = (float*)malloc(output_data_size);
    memcpy(output_data_ptr, output->GetHandle().base, output_data_size);
    DumpDataToTxt(output_data_ptr, output_dims, "../dump_data/dump_data.txt");
    DumpDataToBin(output_data_ptr, output_dims, "../dump_data/dump_data.bin");
    // SpiltResult(output_data_ptr, outout_dims);

    if (output_data_ptr != nullptr)
        free(output_data_ptr);
    return 0;
}
