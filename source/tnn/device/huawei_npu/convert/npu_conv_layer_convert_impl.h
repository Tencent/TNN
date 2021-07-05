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

#include "graph/attr_value.h"
#include "graph/op/all_ops.h"
#include "graph/op/nn_defs.h"
#include "npu_base_layer_convert.h"
#include "npu_utils.h"
#ifndef TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_CONVERT_NPU_CONV_LAYER_IMPL_H_
#define TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_CONVERT_NPU_CONV_LAYER_IMPL_H_

namespace TNN_NS {

class NpuConvImplLayer : public NpuBaseLayer {
public:
    NpuConvImplLayer(LayerType layer_type) : NpuBaseLayer(layer_type){};
    virtual ~NpuConvImplLayer() {}

protected:
    Status ObtainParam() {
        auto param = dynamic_cast<ConvLayerParam *>(param_);
        CHECK_PARAM_NULL(param);
        stride_w_ = param->strides[0];
        stride_h_ = param->strides[1];

        dilation_w_ = param->dialations[0];
        dilation_h_ = param->dialations[1];

        kernel_w_       = param->kernels[0];
        kernel_h_       = param->kernels[1];
        group_          = param->group;
        output_channel_ = param->output_channel;
        pad_w_begin_    = param->pads[0];
        pad_w_end_      = param->pads[1];
        pad_h_begin_    = param->pads[2];
        pad_h_end_      = param->pads[3];
        pad_type_       = param->pad_type;

        activation_type_ = param->activation_type;

        return TNN_OK;
    }
    std::vector<shared_ptr<ge::Operator>> weight_ops_;
    int stride_w_;
    int stride_h_;

    int dilation_w_;
    int dilation_h_;

    int kernel_w_;
    int kernel_h_;

    int pad_w_begin_;
    int pad_w_end_;
    int pad_h_begin_;
    int pad_h_end_;

    int group_;
    int output_channel_;
    int pad_type_;

    int activation_type_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_CONVERT_NPU_CONV_LAYER_CONVERT_IMPL_H_
