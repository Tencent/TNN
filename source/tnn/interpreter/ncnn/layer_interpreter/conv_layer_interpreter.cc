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

#include "tnn/interpreter/ncnn/layer_interpreter/abstract_layer_interpreter.h"
#include "tnn/interpreter/ncnn/ncnn_layer_type.h"
#include "tnn/interpreter/ncnn/ncnn_param_utils.h"

namespace TNN_NS {

namespace ncnn {

    DECLARE_LAYER_INTERPRETER(Conv);

    REGISTER_LAYER_INTERPRETER(Conv, Convolution);
    REGISTER_LAYER_INTERPRETER(Conv, ConvolutionDepthWise);

    Status ConvLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                LayerParam** param) {
        type = ConvertNCNNLayerType(type_name);

        auto& p = param_dict;

        int num_output = GetInt(p, 0, 0);

        int kernel_w = GetInt(p, 1, 0);
        int kernel_h = GetInt(p, 11, kernel_w);

        int dilation_w = GetInt(p, 2, 1);
        int dilation_h = GetInt(p, 12, dilation_w);

        int stride_w = GetInt(p, 3, 1);
        int stride_h = GetInt(p, 13, stride_w);

        int pad_left    = GetInt(p, 4, 0);
        int pad_right   = GetInt(p, 15, pad_left);
        int pad_top     = GetInt(p, 14, pad_left);
        int pad_bottom  = GetInt(p, 16, pad_top);
        float pad_value = GetFloat(p, 18, 0.f);

        int bias_term        = GetInt(p, 5, 0);
        int weight_data_size = GetInt(p, 6, 0);

        int group = GetInt(p, 7, 1);

        int int8_scale_term = GetInt(p, 8, 0);

        int activation_type    = GetInt(p, 9, 0);
        auto activation_params = GetFloatList(p, 10);

        int impl_type = GetInt(p, 17, 0);

        ConvLayerParam* layer_param = new ConvLayerParam();
        *param                      = layer_param;

        // group
        layer_param->group = group;

        // input and output channel
        layer_param->input_channel  = 0;
        layer_param->output_channel = num_output;

        // kernels
        layer_param->kernels.push_back(kernel_w);
        layer_param->kernels.push_back(kernel_h);

        // strides
        layer_param->strides.push_back(stride_w);
        layer_param->strides.push_back(stride_h);

        // pads
        layer_param->pads.push_back(pad_left);
        layer_param->pads.push_back(pad_right);
        layer_param->pads.push_back(pad_top);
        layer_param->pads.push_back(pad_bottom);

        // bias
        layer_param->bias = bias_term;

        // padding type
        if (pad_left == -233 && pad_top == -233 && pad_right == -233 && pad_bottom == -233) {
            layer_param->pad_type = 0;  // SAME
        } else if (pad_left == -234 && pad_top == -234 && pad_right == -234 && pad_bottom == -234) {
            // SAME LOWER
            return Status(TNNERR_INVALID_NETCFG, "ncnn conv padding mode same_lower is not supported now");
        } else {
            layer_param->pad_type = -1;  // DEFAULT
        }

        // dailations
        layer_param->dialations.push_back(dilation_w);
        layer_param->dialations.push_back(dilation_h);

        // activation
        layer_param->activation_type = activation_type;

        // weight_data_size
        layer_param->weight_data_size = weight_data_size;

        return TNN_OK;
    }

    Status ConvLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                   LayerResource** resource) {
        ConvLayerResource* layer_res = new ConvLayerResource();
        *resource                    = layer_res;

        auto param = std::dynamic_pointer_cast<ConvLayerParam>(info->param);
        if (!param) {
            return Status(TNNERR_LAYER_ERR, "conv layer param is nil: ConvLayerParam");
        }

        RawBuffer weights;
        deserializer.GetRaw(weights, param->weight_data_size);

        // LOGDT("conv model %d %.6f\n", "ncnn", param->weight_data_size, weights.force_to<float *>()[0]);

        layer_res->filter_format = OIHW;
        layer_res->filter_handle = weights;

        if (param->bias) {
            RawBuffer bias;
            deserializer.GetRawSimple(bias, param->output_channel);
            layer_res->bias_handle = bias;
        }

        // if (weights.GetDataType() == DATA_TYPE_INT8) {
        //     // quantized
        //     RawBuffer scale;
        //     deserializer.GetRaw(scale);
        //     layer_res->scale_handle = scale;
        // }
        // if (int8_scale_term)
        // {
        //     weight_data_int8_scales = mb.load(num_output, 1);
        //     bottom_blob_int8_scale = mb.load(1, 1)[0];
        // }

        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
