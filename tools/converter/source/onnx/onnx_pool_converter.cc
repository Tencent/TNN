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

#include "onnx_utils.h"
#include "tools/converter/source/onnx/onnx_base_converter.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(Pooling);

std::string OnnxPoolingConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    if (node.op_type() == "GlobalAveragePool" || node.op_type() == "GlobalMaxPool" || node.op_type() == "AveragePool" ||
        node.op_type() == "MaxPool") {
        return "Pooling";
    }
    return "";
}

TNN_NS::ActivationType OnnxPoolingConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxPoolingConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                          const onnx::NodeProto &node,
                                          std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
                                          std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes,
                                          bool &quantized_model) {
    const std::string &onnx_op = node.op_type();
    auto param                 = new TNN_NS::PoolingLayerParam;
    auto cur_layer             = net_structure.layers.back();
    cur_layer->param           = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type                = cur_layer->type_str;
    param->name                = cur_layer->name;
    param->quantized           = false;
    // do not support adaptive pool for now
    param->is_adaptive_pool = 0;
    param->output_shape     = {-1, -1};
    if (onnx_op == "AveragePool" || onnx_op == "MaxPool") {
        auto auto_pad     = GetAttributeString(node, "auto_pad", "NOTSET");
        auto kernel_shape = GetAttributeIntVector(node, "kernel_shape");
        auto strides      = GetAttributeIntVector(node, "strides");
        auto pads         = GetAttributeIntVector(node, "pads");
        //计算输出时候采用的截断方式 0：floor 1：ceil
        int ceil_mode = GetAttributeInt(node, "ceil_mode", 0);

        int pad_type = -1;
        if (auto_pad == "SAME_UPPER") {
            pad_type = 0;
        } else if (auto_pad == "VALID") {
            pad_type = 1;
        } else if (auto_pad == "SAME_LOWER") {
            pad_type = 0;
            LOGE("SAME_LOWER is unsuported, change to SAME_UPPER\n");
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }

        param->pool_type = (onnx_op == "AveragePool") ? 1 : 0;

        bool is3d = false;
        if (kernel_shape.size() == 1) {
            param->kernels.push_back(kernel_shape[0]);
            param->kernels.push_back(kernel_shape[0]);
        } else if (kernel_shape.size() == 2) {
            param->kernels.push_back(kernel_shape[1]);
            param->kernels.push_back(kernel_shape[0]);
        } else if (kernel_shape.size() == 3) {
            is3d = true;
            param->kernels.push_back(kernel_shape[2]);
            param->kernels.push_back(kernel_shape[1]);
            param->kernels.push_back(kernel_shape[0]);
        }
        param->kernels_params = param->kernels;

        if (strides.size() == 1) {
            param->strides.push_back(strides[0]);
            param->strides.push_back(strides[0]);
        } else if (strides.size() == 2) {
            param->strides.push_back(strides[1]);
            param->strides.push_back(strides[0]);
        } else if (strides.size() == 3) {
            param->strides.push_back(strides[2]);
            param->strides.push_back(strides[1]);
            param->strides.push_back(strides[0]);
        }

        if (pads.size() == 1) {
            param->pads.push_back(pads[0]);
            param->pads.push_back(pads[0]);
            param->pads.push_back(pads[0]);
            param->pads.push_back(pads[0]);
        } else if (pads.size() == 2) {
            param->pads.push_back(pads[0]);
            param->pads.push_back(pads[0]);
            param->pads.push_back(pads[1]);
            param->pads.push_back(pads[1]);
        } else if (pads.size() == 4) {
            if (pads[0] == pads[2] && pads[1] == pads[3]) {
                param->pads.push_back(pads[1]);
                param->pads.push_back(pads[1]);
                param->pads.push_back(pads[0]);
                param->pads.push_back(pads[0]);
            } else if (pads[0] < pads[2] && pads[1] < pads[3]) {
                pad_type = 0;  // SAME UPPER
                param->pads.push_back(pads[0]);
                param->pads.push_back(pads[0]);
                param->pads.push_back(pads[1]);
                param->pads.push_back(pads[1]);
            } else {
                LOGE("SAME_LOWER is unsuported, change toSAME_UPPER \n");
                return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
            }
        } else if (pads.size() == 6) {
            if (pads[0] == pads[3] && pads[1] == pads[4] && pads[2] == pads[5]) {
                param->pads.push_back(pads[0]);
                param->pads.push_back(pads[1]);
                param->pads.push_back(pads[2]);
            } else if (pads[0] < pads[3] && pads[1] < pads[4] && pads[2] < pads[5]) {
                pad_type = 0;  // SAME UPPER
                param->pads.push_back(pads[0]);
                param->pads.push_back(pads[1]);
                param->pads.push_back(pads[2]);
            } else {
                LOGE("SAME_LOWER is unsuported, change toSAME_UPPER \n");
                return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
            }
        } else {
            if (auto_pad == "NOTSET" || auto_pad == "SAME_LOWER" || auto_pad == "SAME_UPPER" || auto_pad == "VALID") {
                if (kernel_shape.size() == 3) {
                    param->pads.push_back(0);
                    param->pads.push_back(0);
                    param->pads.push_back(0);
                } else {
                    param->pads.push_back(0);
                    param->pads.push_back(0);
                    param->pads.push_back(0);
                    param->pads.push_back(0);
                }
            } else {
                LOGE("OnnxPooling unsupport this type!\n");
                return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
            }
        }
        // kernel_h_index_in_input_node_size kernel_w_index_in_input_node_size
        // for runtime kernel size of global pool
        if (is3d) {
            param->kernel_indexs.push_back(-1);
            param->kernel_indexs.push_back(-1);
            param->kernel_indexs.push_back(-1);
        } else {
            param->kernel_indexs.push_back(-1);
            param->kernel_indexs.push_back(-1);
        }
        // pad type
        param->pad_type = pad_type;
        // ceil mode, 计算输出时候采用的截断方式 0：floor 1：ceil
        param->ceil_mode = ceil_mode;

    } else {
        param->pool_type = (onnx_op == "GlobalAveragePool") ? 1 : 0;
        param->kernels.push_back(0);
        param->kernels.push_back(0);
        param->kernels_params = param->kernels;
        param->strides.push_back(1);
        param->strides.push_back(1);
        param->pads.push_back(0);
        param->pads.push_back(0);
        param->pads.push_back(0);
        param->pads.push_back(0);
        param->kernel_indexs.push_back(-1);
        param->kernel_indexs.push_back(-1);
        param->pad_type  = -1;
        param->ceil_mode = 0;
    }

    return TNN_NS::TNN_CONVERT_OK;
}
REGISTER_CONVERTER(Pooling, MaxPool);
REGISTER_CONVERTER(Pooling, AveragePool);
REGISTER_CONVERTER(Pooling, GlobalMaxPool);
REGISTER_CONVERTER(Pooling, GlobalAveragePool);

}  // namespace TNN_CONVERTER
