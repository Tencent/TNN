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

#include "onnx/onnx_utils.h"
#include "tnn/interpreter/tnn/objseri.h"
#include "tools/converter/source/onnx/onnx_base_converter.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(Resize);

std::string OnnxResizeConverter::TNNOpType(const onnx::NodeProto &node, bool quantized_model) {
    return "Upsample";
}

TNN_NS::ActivationType OnnxResizeConverter::ActivationType(const onnx::NodeProto &node) {
    return TNN_NS::ActivationType_None;
}

TNN_NS::Status OnnxResizeConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                         const onnx::NodeProto &node,
                                         std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
                                         std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes,
                                         bool &quantized_model) {
    const std::string &onnx_op = node.op_type();
    auto param                 = new TNN_NS::UpsampleLayerParam;
    auto cur_layer             = net_structure.layers.back();
    cur_layer->param           = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type                = cur_layer->type_str;
    param->name                = cur_layer->name;
    param->quantized           = false;
    param->mode                = 0;

    std::string coordinate_transformation_mode =
        GetAttributeString(node, "coordinate_transformation_mode", "half_pexel");
    int align_corners = 0;
    if (coordinate_transformation_mode == "half_pixel" || coordinate_transformation_mode == "pytorch_half_pixel") {
        align_corners = 0;
    } else if (coordinate_transformation_mode == "align_corners") {
        align_corners = 1;
    } else {
        LOGE("resize: coordinate_transformation_mode(%s) is not supported, result may be different.\n",
             coordinate_transformation_mode.c_str());
        return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
    }
    param->align_corners = align_corners;

    std::vector<float> scales;
    std::vector<int> sizes;
    if (node.input_size() > 2) {
        std::string scales_name = node.input(2);
        if (proxy_initializers_map.find(scales_name) != proxy_initializers_map.end()) {
            const auto scales_tensor = proxy_initializers_map[scales_name];
            const auto scales_dims   = std::vector<int>(scales_tensor->dims().begin(), scales_tensor->dims().end());
            void *raw_data_ptr       = GetDataFromTensor(*scales_tensor, onnx::TensorProto_DataType_FLOAT);
            const int count          = TNN_NS::DimsVectorUtils::Count(scales_dims);
            for (int i = 0; i < count; ++i) {
                scales.push_back(*((float *)raw_data_ptr + i));
            }
        }
    }

    if (node.input_size() > 3) {
        std::string sizes_name = node.input(3);
        if (proxy_initializers_map.find(sizes_name) != proxy_initializers_map.end()) {
            const auto sizes_tensor = proxy_initializers_map[sizes_name];
            const auto sizes_dims   = std::vector<int>(sizes_tensor->dims().begin(), sizes_tensor->dims().end());
            void *raw_data_ptr      = GetDataFromTensor(*sizes_tensor, onnx::TensorProto_DataType_INT64);
            const int count         = TNN_NS::DimsVectorUtils::Count(sizes_dims);
            for (int i = 0; i < count; ++i) {
                sizes.push_back(*((int64_t *)raw_data_ptr + i));
            }
        }
    }

    // parse mode
    auto model = GetAttributeString(node, "mode", "nearest");
    if ("nearest" == model) {
        param->mode = 1;
    } else if ("bilinear" == model || "linear" == model) {
        param->mode = 2;
    } else if ("trilinear" == model) {
        LOGE("Onnx Converter: do not support resize trilinear mode\n");
        return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
    }
    if (!sizes.empty()) {
        param->scales.push_back(0.0f);
        param->scales.push_back(0.0f);
        if (sizes.size() == 4) {
            param->dims.push_back(sizes[3]);
            param->dims.push_back(sizes[2]);
        } else {
            LOGE("Onnx Converter: get wrong sizes\n");
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }
    } else {
        if (scales.empty()) {
            LOGE("Onnx Converter: get wrong scales\n");
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }
        float scale_h = 0.0;
        float scale_w = 0.0;
        if (scales.size() == 2) {
            scale_w = scales[1];
        } else if (scales.size() == 3) {
            scale_h = scales[1];
            scale_w = scales[2];
        } else if (scales.size() == 4) {
            scale_h = scales[2];
            scale_w = scales[3];
        }
        param->scales.push_back(scale_w);
        param->scales.push_back(scale_h);
    }
    cur_layer->inputs.resize(1);
    cur_layer->inputs[0] = node.input(0);

    return TNN_NS::TNN_CONVERT_OK;
}

REGISTER_CONVERTER(Resize, Resize);

}  // namespace TNN_CONVERTER
