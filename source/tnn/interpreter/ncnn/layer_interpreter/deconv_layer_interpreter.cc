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

    DECLARE_LAYER_INTERPRETER(Deconv);

    REGISTER_LAYER_INTERPRETER(Deconv, Deconvolution);
    REGISTER_LAYER_INTERPRETER(Deconv, DeconvolutionDepthWise);

    Status DeconvLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                  LayerParam** param) {
        type = ConvertNCNNLayerType(type_name);

        ConvLayerParam* layer_param = new ConvLayerParam();
        *param                      = layer_param;

        auto& p = param_dict;

        int num_output = GetInt(p, 0, 0);
        // input and output channel
        layer_param->input_channel  = 0;
        layer_param->output_channel = num_output;

        // kernels
        int kernel_w = GetInt(p, 1, 0);
        int kernel_h = GetInt(p, 11, kernel_w);
        layer_param->kernels.push_back(kernel_w);
        layer_param->kernels.push_back(kernel_h);

        // strides
        int stride_w = GetInt(p, 3, 1);
        int stride_h = GetInt(p, 13, stride_w);
        layer_param->strides.push_back(stride_w);
        layer_param->strides.push_back(stride_h);

        // pads
        int pad_left   = GetInt(p, 4, 0);
        int pad_right  = GetInt(p, 15, pad_left);
        int pad_top    = GetInt(p, 14, pad_left);
        int pad_bottom = GetInt(p, 16, pad_top);
        layer_param->pads.push_back(pad_left);
        layer_param->pads.push_back(pad_right);
        layer_param->pads.push_back(pad_top);
        layer_param->pads.push_back(pad_bottom);

        // dailations
        int dilation_w = GetInt(p, 2, 1);
        int dilation_h = GetInt(p, 12, dilation_w);
        layer_param->dialations.push_back(dilation_w);
        layer_param->dialations.push_back(dilation_h);

        // bias
        int bias_term                 = GetInt(p, 5, 0);
        int weight_data_size          = GetInt(p, 6, 0);
        layer_param->bias             = bias_term;
        layer_param->weight_data_size = weight_data_size;

        // group
        int group          = GetInt(p, 7, 1);
        layer_param->group = group;

        int int8_scale_term = GetInt(p, 8, 0);

        // activation
        int activation_type          = GetInt(p, 9, 0);
        auto activation_params       = GetFloatList(p, 10);
        layer_param->activation_type = activation_type;

        int output_pad_right  = GetInt(p, 18, 0);
        int output_pad_bottom = GetInt(p, 19, 0);
        int output_w          = GetInt(p, 20, 0);
        int output_h          = GetInt(p, 21, 0);

        if (output_h != 0 || output_w != 0) {
            return Status(TNNERR_INVALID_NETCFG, "ncnn deconv with output hw is not supported now");
        }

        // padding type
        layer_param->pad_type = -1;  // DEFAULT
        if (output_pad_right != 0 || output_pad_bottom != 0) {
            // deconv exchange pad_right and pad_left because of output_padding
            layer_param->pad_type = 3;
        }

        return TNN_OK;
    }

    Status DeconvLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                     LayerResource** resource) {
        ConvLayerResource* layer_res = new ConvLayerResource();
        *resource                    = layer_res;

        auto param = std::dynamic_pointer_cast<ConvLayerParam>(info->param);
        if (!param) {
            return Status(TNNERR_LAYER_ERR, "layer param is nil: ConvLayerParam");
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
