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

#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_1x1_acc.h"
#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_acc_impl.h"
#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_common_acc.h"
#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_depthwise_acc.h"
#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_winograd_acc.h"
#include "tnn/device/opencl/acc/opencl_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {

class OpenCLConvLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLConvLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    std::shared_ptr<OpenCLConvLayerAccImpl> conv_acc_implement_ = nullptr;
};

Status OpenCLConvLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv_param = dynamic_cast<ConvLayerParam *>(param);

    if (OpenCLConvLayerDepthwiseAcc::IsPrefered(conv_param, inputs, outputs)) {
        conv_acc_implement_ = std::make_shared<OpenCLConvLayerDepthwiseAcc>();
    } else if (OpenCLConvLayer1x1Acc::IsPrefered(conv_param, inputs, outputs)) {
        conv_acc_implement_ = std::make_shared<OpenCLConvLayer1x1Acc>();
    } else if (OpenCLConvLayerWinogradAcc::IsPrefered(conv_param, inputs, outputs)) {
        conv_acc_implement_ = std::make_shared<OpenCLConvLayerWinogradAcc>();
    } else if (OpenCLConvLayerCommonAcc::IsPrefered(conv_param, inputs, outputs)) {
        conv_acc_implement_ = std::make_shared<OpenCLConvLayerCommonAcc>();
    }

    if (conv_acc_implement_ == nullptr)
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "this type conv acc is not implemented");

    return conv_acc_implement_->Init(context, conv_param, resource, inputs, outputs);
}

OpenCLConvLayerAcc::~OpenCLConvLayerAcc() {}

Status OpenCLConvLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (conv_acc_implement_ == nullptr)
        return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "this type conv acc is not implemented");

    return conv_acc_implement_->Reshape(inputs, outputs);
}

Status OpenCLConvLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (conv_acc_implement_ == nullptr)
        return Status(TNNERR_OPENCL_ACC_FORWARD_ERROR, "this type conv acc is not implemented");

    return conv_acc_implement_->Forward(inputs, outputs);
}

REGISTER_OPENCL_ACC(Conv, LAYER_CONVOLUTION)

}  // namespace TNN_NS
