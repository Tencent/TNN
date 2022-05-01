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

#include "tnn/device/directx/acc/directx_layer_acc.h"
#include "tnn/device/directx/acc/convolution/directx_conv_layer_1x1_acc.h"
#include "tnn/device/directx/acc/convolution/directx_conv_layer_acc_impl.h"
#include "tnn/device/directx/acc/convolution/directx_conv_layer_common_acc.h"
#include "tnn/device/directx/acc/convolution/directx_conv_layer_depthwise_acc.h"
#include "tnn/device/directx/acc/convolution/directx_conv_layer_winograd_acc.h"

namespace TNN_NS {

namespace directx {

class DirectXConvLayerAcc : public DirectXLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~DirectXConvLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

#if TNN_PROFILE
    virtual double GetFlops() override;
#endif

private:
    std::shared_ptr<DirectXConvLayerAccImpl> conv_acc_implement_ = nullptr;
};

Status DirectXConvLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret        = DirectXLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param);

    if (DirectXConvLayerDepthwiseAcc::IsPrefered(conv_param, inputs, outputs)) {
        conv_acc_implement_ = std::make_shared<DirectXConvLayerDepthwiseAcc>();
    } else if (DirectXConvLayer1x1Acc::IsPrefered(conv_param, inputs, outputs)) {
        conv_acc_implement_ = std::make_shared<DirectXConvLayer1x1Acc>();
    } else if (DirectXConvLayerWinogradAcc::IsPrefered(conv_param, inputs, outputs)) {
        conv_acc_implement_ = std::make_shared<DirectXConvLayerWinogradAcc>();
    } else if (DirectXConvLayerCommonAcc::IsPrefered(conv_param, inputs, outputs)) {
        conv_acc_implement_ = std::make_shared<DirectXConvLayerCommonAcc>();
    }

    if (conv_acc_implement_ == nullptr)
        return Status(TNNERR_DX_ACC_INIT_ERR, "directx conv acc is not implemented");

    return conv_acc_implement_->Init(context, conv_param, resource, inputs, outputs);
}

DirectXConvLayerAcc::~DirectXConvLayerAcc() {}

Status DirectXConvLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = DirectXLayerAcc::Reshape(inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    if (conv_acc_implement_ == nullptr)
        return Status(TNNERR_DX_ACC_INIT_ERR, "this type conv acc is not implemented");

    return conv_acc_implement_->Reshape(inputs, outputs);
}

Status DirectXConvLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (conv_acc_implement_ == nullptr)
        return Status(TNNERR_DX_LAYER_ERR, "this type conv acc is not implemented");

    return conv_acc_implement_->DoForward(inputs, outputs);
}

#if TNN_PROFILE
double DirectXConvLayerAcc::GetFlops(){
    if (conv_acc_implement_ == nullptr) {
        LOGE("this type conv acc is not implemented");
        return 0.;
    }

    return conv_acc_implement_->GetFlops();
}
#endif


REGISTER_DIRECTX_ACC(Conv, LAYER_CONVOLUTION)
REGISTER_DIRECTX_LAYOUT(LAYER_CONVOLUTION, DATA_FORMAT_NCHW);
REGISTER_DIRECTX_LAYOUT(LAYER_CONVOLUTION, DATA_FORMAT_NHC4W4);

} // namespace directx
}  // namespace TNN_NS
