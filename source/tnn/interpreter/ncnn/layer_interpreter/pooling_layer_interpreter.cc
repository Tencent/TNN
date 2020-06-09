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

    DECLARE_LAYER_INTERPRETER(Pooling);

    REGISTER_LAYER_INTERPRETER(Pooling, Pooling);

    static std::map<int, int> global_ncnn_pad_type_map = {
        {0, -1},  // ncnn full padding not supported
        {1, -1},  // ncnn valid padding -> rpn default padding
        {2, 0},   // tf same padding
        {3, 0},   // onnx same_lower
    };

    Status PoolingLayerInterpreter::InterpretProto(std::string type_name, str_dict param_dict, LayerType& type,
                                                   LayerParam** param) {
        PoolingLayerParam* layer_param = new PoolingLayerParam();
        *param                         = layer_param;

        type = ConvertNCNNLayerType(type_name);

        auto& p = param_dict;

        int pooling_type = GetInt(p, 0, 0);

        int kernel_w = GetInt(p, 1, 0);
        int kernel_h = GetInt(p, 11, kernel_w);

        int stride_w = GetInt(p, 2, 1);
        int stride_h = GetInt(p, 2, stride_w);

        int pad_left   = GetInt(p, 3, 0);
        int pad_right  = GetInt(p, 14, pad_left);
        int pad_top    = GetInt(p, 13, pad_left);
        int pad_bottom = GetInt(p, 15, pad_top);

        int global_pooling = GetInt(p, 4, 0);
        int pad_mod        = GetInt(p, 5, 0);

        if (global_pooling == 1) {
            kernel_w   = 0;
            kernel_h   = 0;
            pad_left   = 0;
            pad_right  = 0;
            pad_top    = 0;
            pad_bottom = 0;
            pad_mod    = 1;
        }

        layer_param->pool_type = pooling_type;

        layer_param->kernels_params.push_back(kernel_w);
        layer_param->kernels_params.push_back(kernel_h);
        layer_param->kernels.push_back(kernel_w);
        layer_param->kernels.push_back(kernel_h);

        layer_param->strides.push_back(stride_w);
        layer_param->strides.push_back(stride_h);

        layer_param->pads.push_back(pad_left);
        layer_param->pads.push_back(pad_right);
        layer_param->pads.push_back(pad_top);
        layer_param->pads.push_back(pad_bottom);
        layer_param->pad_type  = global_ncnn_pad_type_map[pad_mod];
        layer_param->ceil_mode = -1;

        layer_param->kernel_indexs.push_back(-1);
        layer_param->kernel_indexs.push_back(-1);

        if (pad_mod == 0) {
            layer_param->ceil_mode = 1;
        }

        /* ncnn same_lower padding:
            pad right and bottom first;
            we pads left and top as default.
            */
        if (pad_mod == 3) {
            return Status(TNNERR_INVALID_NETCFG, "ncnn pool mod 3 SAME_LOWER is not supported now");
        }

        return TNN_OK;
    }

    Status PoolingLayerInterpreter::InterpretResource(Deserializer& deserializer, std::shared_ptr<LayerInfo> info,
                                                      LayerResource** resource) {
        return TNN_OK;
    }

}  // namespace ncnn

}  // namespace TNN_NS
