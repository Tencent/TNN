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
        stride_w = param->strides[0];
        stride_h = param->strides[1];

        dilation_w = param->dialations[0];
        dilation_h = param->dialations[1];

        kernel_w       = param->kernels[0];
        kernel_h       = param->kernels[1];
        group          = param->group;
        output_channel = param->output_channel;
        pad_w_begin    = param->pads[0];
        pad_w_end      = param->pads[1];
        pad_h_begin    = param->pads[2];
        pad_h_end      = param->pads[3];
        pad_type       = param->pad_type;

        return TNN_OK;
    }
    std::vector<shared_ptr<ge::Operator>> weight_ops_;
    int stride_w;
    int stride_h;

    int dilation_w;
    int dilation_h;

    int kernel_w;
    int kernel_h;

    int pad_w_begin;
    int pad_w_end;
    int pad_h_begin;
    int pad_h_end;

    int group;
    int output_channel;
    int pad_type;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_HUAWEI_NPU_CONVERT_NPU_CONV_LAYER_CONVERT_IMPL_H_
