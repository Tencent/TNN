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

#include "rknpu_utils.h"

#include <tnn/interpreter/layer_resource.h>
#include <tnn/utils/dims_vector_utils.h>

#include <numeric>
#include <sstream>

#include "tnn/core/macro.h"

namespace tnn {

std::shared_ptr<rk::nn::Tensor> RknpuUtils::CreateRknnTensor(rk::nn::Graph *graph, const std::string &name,
                                                             const std::vector<int> &dims, const void *data,
                                                             const rk::nn::TensorRole role, const DataType type,
                                                             const rk::nn::DataLayoutType layout,
                                                             const rk::nn::QuantizationType qntType, const uint8_t bits,
                                                             const float scale, const uint32_t zero_point,
                                                             const int8_t fl) {
    auto attr  = std::make_shared<rk::nn::TensorAttr>();
    attr->name = name;
    for (auto dim : dims) {
        attr->dims.push_back((uint32_t)dim);
    }
    switch (type) {
        case DATA_TYPE_FLOAT:
            attr->precision = rk::nn::PrecisionType::FLOAT32;
            break;
        case DATA_TYPE_HALF:
            attr->precision = rk::nn::PrecisionType::FLOAT16;
            break;
        case DATA_TYPE_INT8:
            attr->precision = rk::nn::PrecisionType::INT8;
            break;
        case DATA_TYPE_INT32:
            attr->precision = rk::nn::PrecisionType::INT32;
            break;
        default:
            break;
    }

    attr->layout  = layout;
    attr->qntType = qntType;
    attr->role    = role;
    attr->qntBits = bits;
    attr->qntParamDFP.fl.push_back(fl);
    attr->qntParamAffineAsymmetric.zero_point.push_back(zero_point);
    attr->qntParamAffineAsymmetric.scale.push_back(scale);
    attr->qntParamSymmetric.scale.push_back(scale);
    return graph->CreateTensor(attr, (void *)data);
}

Status RknpuUtils::CalculateBroadcastSize(std::vector<int> &weight, EltwiseLayerResource *layer_res,
                                          std::vector<int> &input) {
    int input_count = DimsVectorUtils::Count(input, 1);
    if (weight.size() < 4) {
        weight             = {1, 1, 1, 1};
        int layer_res_size = layer_res->element_handle.GetDataCount();
        if (layer_res_size == 1) {
            // single element
            weight[1] = layer_res_size;
        } else if (layer_res_size == input[1]) {
            // channel broadcast
            weight[1] = layer_res_size;
        } else if (layer_res_size == input_count) {
            // element broadcast
            weight[1] = input[1];
            weight[2] = input[2];
            weight[3] = input[3];
        } else if (layer_res_size == input[3]) {
            weight[3] = input[3];
        } else {
            return Status(TNNERR_LAYER_ERR, "Error: unsupported broadcast type");
        }
        layer_res->element_shape = weight;
    }
    return TNN_OK;
}

std::string RknpuUtils::GetFileHash(ModelConfig &model_config) {
    std::string file_content = model_config.params[1] + model_config.params[0];
    int hash                 = 0;
    for (size_t i = 0; i < file_content.length(); ++i)
        hash = 65599 * hash + file_content.at(i);
    return std::to_string(hash ^ (hash >> 16));
}

Status RknpuUtils::GetPadType(rk::nn::PadType &rk_pad_type, int pad_type) {
    // rknpu pad mode
    if (pad_type == 0) {  // SAME_UPPER or SAME_LOWER
        rk_pad_type = rk::nn::PadType::SAME;
    } else if (pad_type == 1) {  // VALID
        rk_pad_type = rk::nn::PadType::VALID;
    } else if (pad_type == -1) {  // NOSET
        rk_pad_type = rk::nn::PadType::AUTO;
    } else {
        return Status(TNNERR_PARAM_ERR, "Error: ConvLayer dont support pad type");
    }
    return TNN_OK;
}

uint32_t RknpuUtils::CalcSize(rk::nn::PrecisionType type, std::vector<uint32_t> dims) {
    size_t type_size = 4;
    switch (type) {
        case rk::nn::PrecisionType::FLOAT32:
            type_size = 4;
            break;
        case rk::nn::PrecisionType::UINT8:
            type_size = 1;
            break;
        case rk::nn::PrecisionType::INT32:
            type_size = 4;
            break;
        case rk::nn::PrecisionType::INT64:
            type_size = 8;
            break;
        default:
            throw std::invalid_argument("Init: unknow intput or output data type!");
            break;
    }

    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<uint32_t>()) * type_size;
}

}  // namespace tnn
