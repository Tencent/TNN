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

#ifndef TNN_TOOLS__DYNAMIC_RANGE_DYNAMIC_RANGE_QUANTIZATION_H
#define TNN_TOOLS__DYNAMIC_RANGE_DYNAMIC_RANGE_QUANTIZATION_H

#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/interpreter/tnn/model_packer.h"

namespace TNN_NS {
class DynamicRangeQuantizer {
public:
    DynamicRangeQuantizer() = delete;
    DynamicRangeQuantizer(const std::shared_ptr<NetStructure>& net_structure,
                          const std::shared_ptr<NetResource>& net_resource);

    ~DynamicRangeQuantizer() {}

public:
    Status GetDynamicRangeQuantModel(std::shared_ptr<NetStructure>& net_structure,
                                     std::shared_ptr<NetResource>& net_resource);

private:
    Status QuantConvolution(std::shared_ptr<LayerInfo>& layer,
                            std::map<std::string, std::shared_ptr<LayerResource>>& resource_map,
                            std::map<std::string, std::shared_ptr<RawBuffer>>& constant_map);
    Status QuantLSTM(std::shared_ptr<LayerInfo>& layer,
                     std::map<std::string, std::shared_ptr<LayerResource>>& resource_map,
                     std::map<std::string, std::shared_ptr<RawBuffer>>& constant_map);
    Status QuantMatMul(std::shared_ptr<LayerInfo>& layer,
                       std::map<std::string, std::shared_ptr<LayerResource>>& resource_map,
                       std::map<std::string, std::shared_ptr<RawBuffer>>& constant_map);
    Status QuantInnerProduct(std::shared_ptr<LayerInfo>& layer,
                       std::map<std::string, std::shared_ptr<LayerResource>>& resource_map,
                       std::map<std::string, std::shared_ptr<RawBuffer>>& constant_map);
    Status QuantGatherEmbedding(std::shared_ptr<LayerInfo>& layer,
                       std::map<std::string, std::shared_ptr<LayerResource>>& resource_map,
                       std::map<std::string, std::shared_ptr<RawBuffer>>& constant_map);

    Status PerChannelQuant(RawBuffer& weight_buf, RawBuffer& quant_buf, RawBuffer& scale_buf, int num_kernel);
    Status PerTensorQuant(RawBuffer& weight_buf, RawBuffer& quant_buf, RawBuffer& scale_buf);

    std::shared_ptr<NetStructure> net_structure_ = nullptr;
    std::shared_ptr<NetResource> net_resource_   = nullptr;
    const int bits_                              = 8;
    const float threshold_                       = (float)(1 << (bits_ - 1)) - 1.0f;
};
}  // namespace TNN_NS

#endif  // TNN_TOOLS__DYNAMIC_RANGE_DYNAMIC_RANGE_QUANTIZATION_H
