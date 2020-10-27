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

#include "tnn_lib.h"

#include <fstream>
#include <random>
#include <chrono>
#include "tnn/core/common.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/blob_converter.h"


TNNLib::TNNLib() {}

TNNLib::~TNNLib() {
    tnn_.DeInit();
}

int TNNLib::Init(const std::string& proto_file,
               const std::string& model_file, const std::string& device) {
    TNN_NS::ModelConfig model_config;
    {
        std::ifstream f(proto_file);
        std::string buffer;
        buffer = std::string((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());

        model_config.params.push_back(buffer);
    }
    {
        std::ifstream f(model_file);
        std::string buffer;
        buffer = std::string((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());

        model_config.params.push_back(buffer);
    }

    tnn_.Init(model_config);

    TNN_NS::Status error;

    TNN_NS::NetworkConfig cpu_network_config;
    if("ARM" == device) {
        cpu_network_config.device_type = TNN_NS::DEVICE_ARM;
    } else if("OPENCL" == device){
        cpu_network_config.device_type = TNN_NS::DEVICE_OPENCL;
    } else if ("HUAWEI_NPU" == device) {
        cpu_network_config.device_type = TNN_NS::DEVICE_HUAWEI_NPU;
    }
    instance_ = tnn_.CreateInst(cpu_network_config, error);
    return (int)error;

}

std::vector<float> TNNLib::Forward(void* sourcePixelscolor) {
    if (!instance_) {
        return {};
    }

    void* command_queue;
    instance_->GetCommandQueue((void**)&command_queue);

    TNN_NS::BlobMap input_blobs;
    instance_->GetAllInputBlobs(input_blobs);
    TNN_NS::Blob* input = input_blobs.begin()->second;
    TNN_NS::Mat input_mat(TNN_NS::DEVICE_ARM, TNN_NS::N8UC4, sourcePixelscolor);

    TNN_NS::BlobConverter input_blob_convert(input);
    TNN_NS::MatConvertParam input_convert_param;
    input_convert_param.scale = {1.0/ (255 * 0.229), 1.0/ (255 * 0.224), 1.0/(255 * 0.225), 0.0};
    input_convert_param.bias = {-0.485/0.229, -0.456/ 0.224, -0.406/0.225, 0.0};
    input_blob_convert.ConvertFromMat(input_mat, input_convert_param, command_queue);

    instance_->Forward();

    TNN_NS::BlobMap output_blobs;
    instance_->GetAllOutputBlobs(output_blobs);
    TNN_NS::Blob* output = output_blobs.begin()->second;

    int output_count = TNN_NS::DimsVectorUtils::Count(output->GetBlobDesc().dims);
    std::vector<float> result(output_count);
    TNN_NS::Mat output_mat(TNN_NS::DEVICE_ARM, TNN_NS::NCHW_FLOAT, result.data());

    TNN_NS::BlobConverter output_blob_convert(output);
    TNN_NS::MatConvertParam output_convert_param;
    output_blob_convert.ConvertToMat(output_mat, output_convert_param, command_queue);

    return result;
}
