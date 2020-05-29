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

#include "tnn_inst.h"
#include <random>
#include <string>
#include <sstream>
#include "tnn/core/common.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/dims_vector_utils.h"
#include "utils.h"

namespace arm_linux_demo {

TNNInst::~TNNInst() {
    tnn_.DeInit();
}

int TNNInst::Init(const std::string& proto_file,
                    const std::string& model_file) {
    TNN_NS::ModelConfig model_config;
    {
        std::ifstream proto_stream(proto_file);
        if (!proto_stream.is_open() || !proto_stream.good()) {
            printf("read proto_file failed!\n");
            return -1;
        }
        auto buffer = std::string((std::istreambuf_iterator<char>(proto_stream)),
                                  std::istreambuf_iterator<char>());
        model_config.params.push_back(buffer);
    }

    {
        std::ifstream model_stream(model_file);
        if (!model_stream.is_open() || !model_stream.good()) {
            printf("read model_file failed!\n");
            return -1;
        }
        auto buffer = std::string((std::istreambuf_iterator<char>(model_stream)),
                                  std::istreambuf_iterator<char>());
        model_config.params.push_back(buffer);
    }
    
    CHECK_TNN_STATUS(tnn_.Init(model_config));

    TNN_NS::NetworkConfig config;
    config.device_type = TNN_NS::DEVICE_ARM;
    TNN_NS::Status error;
    net_instance_      = tnn_.CreateInst(config, error);
    CHECK_TNN_STATUS(error);

    CHECK_TNN_STATUS(net_instance_->GetAllInputBlobs(input_blobs_));
    CHECK_TNN_STATUS(net_instance_->GetAllOutputBlobs(output_blobs_));

    return TNN_NS::RPD_OK;
}

std::vector<int> TNNInst::GetInputSize() const{
    return input_blobs_.begin()->second->GetBlobDesc().dims;
}

std::vector<int> TNNInst::GetOutputSize() const{
    return input_blobs_.begin()->second->GetBlobDesc().dims;
}

int TNNInst::Forward(TNN_NS::Mat &input_mat, TNN_NS::Mat &output_mat) {
    if (!net_instance_) {
        return TNN_NS::RPDERR_INST_ERR;
    }

    TNN_NS::BlobConverter input_blob_convert(input_blobs_.begin()->second);
    CHECK_TNN_STATUS(
        input_blob_convert.ConvertFromMat(input_mat, input_convert_param_, nullptr));

    CHECK_TNN_STATUS( net_instance_->Forward());

    TNN_NS::BlobConverter output_blob_convert(output_blobs_.begin()->second);
    CHECK_TNN_STATUS(
        output_blob_convert.ConvertToMat(output_mat, output_convert_param_, nullptr));

    return TNN_NS::RPD_OK;
}

}
