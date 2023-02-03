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
#include "flatbuffers/flexbuffers.h"
#include "tflite_op_converter.h"
#include "tflite_utils.h"

namespace TNN_CONVERTER {
DECLARE_OP_CONVERTER(Custom);

std::string TFLiteCustomConverter::TNNOpType(tflite::BuiltinOperator op_code, bool quantized_model) {
    return "DetectionPostProcess";
}
tflite::ActivationFunctionType TFLiteCustomConverter::ActivationType(
    const std::unique_ptr<tflite::OperatorT> &tf_lite_operator, tflite::BuiltinOperator op_code) {
    return tflite::ActivationFunctionType_NONE;
}

TNN_NS::Status TFLiteCustomConverter::exec(TNN_NS::NetStructure &net_structure, TNN_NS::NetResource &net_resource,
                                           const std::unique_ptr<tflite::OperatorT> &tf_lite_operator,
                                           const std::vector<std::unique_ptr<tflite::TensorT>> &tf_lite_tensors,
                                           const std::vector<std::unique_ptr<tflite::BufferT>> &tf_lite_model_buffer,
                                           const std::vector<std::unique_ptr<tflite::OperatorCodeT>> &tf_lite_op_set,
                                           bool quantized_model) {
    TNN_NS::DetectionPostProcessLayerParam *param = new TNN_NS::DetectionPostProcessLayerParam;
    auto cur_layer                                = net_structure.layers.back();
    cur_layer->param                              = std::shared_ptr<TNN_NS::LayerParam>(param);
    param->type                                   = cur_layer->type_str;
    param->name                                   = cur_layer->name;
    param->quantized                              = false;
    auto &custom_op_code                          = tf_lite_op_set[tf_lite_operator->opcode_index]->custom_code;
    assert(custom_op_code == "TFLite_Detection_PostProcess");
    const uint8_t *custom_option_buffer_data_ptr = tf_lite_operator->custom_options.data();
    const auto size                              = tf_lite_operator->custom_options.size();
    const flexbuffers::Map &flex_buffers_map     = flexbuffers::GetRoot(custom_option_buffer_data_ptr, size).AsMap();
    param->max_detections                        = flex_buffers_map["max_detections"].AsInt32();
    param->max_classes_per_detection             = flex_buffers_map["max_classes_per_detection"].AsInt32();
    if (flex_buffers_map["detections_per_class"].IsNull()) {
        param->detections_per_class = 100;
    } else {
        param->detections_per_class = flex_buffers_map["detections_per_class"].AsInt32();
    }
    if (flex_buffers_map["use_regular_nms"].IsNull()) {
        param->use_regular_nms = false;
    } else {
        param->use_regular_nms = flex_buffers_map["use_regular_nms"].AsBool();
    }
    param->nms_score_threshold = flex_buffers_map["nms_score_threshold"].AsFloat();
    param->nms_iou_threshold   = flex_buffers_map["nms_iou_threshold"].AsFloat();
    param->num_classes         = flex_buffers_map["num_classes"].AsInt32();
    param->center_size_encoding.push_back(flex_buffers_map["y_scale"].AsFloat());
    param->center_size_encoding.push_back(flex_buffers_map["x_scale"].AsFloat());
    param->center_size_encoding.push_back(flex_buffers_map["h_scale"].AsFloat());
    param->center_size_encoding.push_back(flex_buffers_map["w_scale"].AsFloat());
    if (tf_lite_operator->inputs.size() == 3) {
        auto layer_resource        = std::make_shared<TNN_NS::DetectionPostProcessLayerResource>();
        layer_resource->name       = cur_layer->name;
        const auto &anchors_tensor = tf_lite_tensors[tf_lite_operator->inputs[2]];
        auto anchors_data_ptr =
            reinterpret_cast<const float *>(tf_lite_model_buffer[anchors_tensor->buffer]->data.data());
        if (anchors_data_ptr != nullptr) {
            param->has_anchors        = true;
            auto anchors_tensor_shape = anchors_tensor->shape;
            auto anchors_size         = Count(anchors_tensor_shape);
            assert(anchors_tensor_shape.size() == 2);
            param->num_anchors               = anchors_tensor_shape[0];
            param->anchors_coord_num         = anchors_tensor_shape[1];
            TNN_NS::RawBuffer anchors_handle = TNN_NS::RawBuffer(anchors_size * sizeof(float));
            anchors_handle.SetDataType(TNN_NS::DATA_TYPE_FLOAT);
            anchors_handle.SetBufferDims(anchors_tensor_shape);
            ::memcpy(anchors_handle.force_to<float *>(), anchors_data_ptr, anchors_size * sizeof(float));
            layer_resource->anchors_handle = anchors_handle;
        }
        net_resource.resource_map[cur_layer->name] = layer_resource;
    }
    cur_layer->inputs.resize(2);
    cur_layer->inputs[0] = tf_lite_tensors[tf_lite_operator->inputs[0]]->name;
    cur_layer->inputs[1] = tf_lite_tensors[tf_lite_operator->inputs[1]]->name;
    assert(tf_lite_operator->outputs.size() == 4);
    return TNN_NS::TNN_CONVERT_OK;
}

using namespace tflite;
REGISTER_CONVERTER(Custom, BuiltinOperator_CUSTOM);

}  // namespace TNN_CONVERTER
