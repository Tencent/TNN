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

#ifndef TNN_SOURCE_TNN_DEVICE_DIRECTX_ACC_DIRECTX_CONV_LAYER_ACC_IMPL_H_
#define TNN_SOURCE_TNN_DEVICE_DIRECTX_ACC_DIRECTX_CONV_LAYER_ACC_IMPL_H_

#include "tnn/device/directx/acc/directx_layer_acc.h"
#include "tnn/device/directx/directx_memory.h"

namespace TNN_NS {

namespace directx {

struct ConvParam {
    int input_channel;
    int output_channel;
    int kernel_h;
    int kernel_w;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;

    int pad_type;
    int group;
    int has_bias;
    int activation_type;
};

enum ConvType { CT_CONV_COMMON = 0, CT_CONV_1x1, CT_CONV_DEPTHWISE, CT_CONV_WINOGRAD };

class DirectXConvLayerAccImpl : public DirectXLayerAcc {
public:
    DirectXConvLayerAccImpl();
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~DirectXConvLayerAccImpl() override;

#if TNN_PROFILE
    virtual double GetFlops() override;
#endif

protected:
    Status AllocateWeightsBias(LayerResource *resource);

private:
    Status ConvertWeights(float *weights_data_ptr);

protected:
    ConvParam conv_params_ = {0};
    shared_ptr<DirectXMemory> weights_;
    shared_ptr<DirectXMemory> bias_;
    ConvType conv_type_;
    bool is_channel_blocking_ = false;
};

}  // namespace directx
}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_DIRECTX_ACC_DIRECTX_CONV_LAYER_ACC_IMPL_H_
