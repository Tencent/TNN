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

#include "tnn/utils/dims_utils.h"
#include "torch/torch_base_converter.h"
#include "torch/torch_utils.h"

namespace TNN_CONVERTER {

DECLARE_TORCH_OP_CONVERTER(Pool);

std::string TorchPoolConverter::TNNOpType(const torch::jit::Node *node, bool quantized_model) {
    return "Pooling";
}

TNN_NS::ActivationType TorchPoolConverter::ActivationType(const torch::jit::Node *node) {
    return TNN_NS::ActivationType_None;
}
// adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
TNN_NS::Status TorchPoolConverter::exec(tnn::NetStructure &net_structure, tnn::NetResource &net_resource,
                                        const torch::jit::Node *node, bool quantized_mode) {
    auto cur_layer      = std::make_shared<TNN_NS::LayerInfo>();
    cur_layer->name     = node->output(0)->debugName();
    auto type_name      = TNNOpType(node, quantized_mode);
    auto layer_type     = TNN_NS::GlobalConvertLayerType(type_name);
    cur_layer->type     = layer_type;
    cur_layer->type_str = type_name;

    cur_layer->inputs.push_back(node->input(0)->debugName());
    cur_layer->outputs.push_back(node->output(0)->debugName());
    net_structure.layers.push_back(cur_layer);
    // parse param
    auto param                = new TNN_NS::PoolingLayerParam;
    cur_layer->param          = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type               = cur_layer->type_str;
    param->name               = cur_layer->name;
    param->quantized          = false;
    std::string torch_op_type = node->kind().toQualString();
    const auto &inputs        = node->inputs();
    std::string op_type       = node->kind().toUnqualString();
    if (op_type.find("adaptive") == std::string::npos) {
        const auto kernel_size = GetValue<std::vector<int64_t>>(inputs[1]);
        const auto stride      = GetValue<std::vector<int64_t>>(inputs[2]);
        const auto padding     = GetValue<std::vector<int64_t>>(inputs[3]);
        const auto dialation   = GetValue<std::vector<int64_t>>(inputs[4]);
        const auto ceil_mode   = GetValue<bool>(inputs[5]);

        param->pad_type       = -1;
        param->kernels_params = {(int)kernel_size[1], (int)kernel_size[0]};
        param->strides        = {(int)stride[1], (int)stride[0]};
        param->pads           = {(int)padding[1], (int)padding[1], (int)padding[0], (int)padding[0]};
        param->kernel_indexs  = {-1, -1};
        param->kernels        = {-1, -1};
        param->output_shape   = {-1, -1};
        param->ceil_mode      = ceil_mode;
    } else {
        const auto output_shape = GetValue<std::vector<int64_t>>(inputs[1]);
        param->is_adaptive_pool = 1;
        param->output_shape     = {(int)output_shape[1], (int)output_shape[0]};
        param->kernels_params   = {-1, -1};
        param->strides          = {1, 1};
        param->pads             = {0, 0, 0, 0};
        param->kernel_indexs    = {-1, -1};
        param->kernels          = {-1, -1};
    }
    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_TORCH_OP_CONVERTER(Pool, aten, adaptive_avg_pool2d);

}  // namespace TNN_CONVERTER
