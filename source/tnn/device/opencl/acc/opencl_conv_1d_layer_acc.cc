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

#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_acc_impl.h"
#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_common_acc.h"
#include "tnn/device/opencl/acc/opencl_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {

class OpenCLConvolution1DLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLConvolution1DLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    std::shared_ptr<OpenCLConvLayerAccImpl> conv_acc_implement_ = nullptr;
};

Status OpenCLConvolution1DLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                         const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    ConvLayerParam *conv1d_param = dynamic_cast<ConvLayerParam *>(param);
    ConvLayerParam *conv2d_param = new ConvLayerParam(*conv1d_param);

    // Fill up the parameters of conv1d to conv2d.
    conv2d_param->kernels.insert(conv2d_param->kernels.begin(), 1);
    conv2d_param->strides.insert(conv2d_param->strides.begin(), 1);
    conv2d_param->dialations.insert(conv2d_param->dialations.begin(), 1);
    conv2d_param->pads.insert(conv2d_param->pads.begin(), 2, 0);

    conv_acc_implement_ = std::make_shared<OpenCLConvLayerCommonAcc>();

    if (conv_acc_implement_ == nullptr)
        return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "this type conv acc is not implemented");

    return conv_acc_implement_->Init(context, conv2d_param, resource, inputs, outputs);
}

OpenCLConvolution1DLayerAcc::~OpenCLConvolution1DLayerAcc() {}

Status OpenCLConvolution1DLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    if (conv_acc_implement_ == nullptr)
        return Status(TNNERR_OPENCL_ACC_RESHAPE_ERROR, "this type conv acc is not implemented");

    return conv_acc_implement_->Reshape(inputs, outputs);
}

Status OpenCLConvolution1DLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    if (conv_acc_implement_ == nullptr)
        return Status(TNNERR_OPENCL_ACC_FORWARD_ERROR, "this type conv acc is not implemented");

    return conv_acc_implement_->Forward(inputs, outputs);
}

REGISTER_OPENCL_ACC(Convolution1D, LAYER_CONVOLUTION_1D)
REGISTER_OPENCL_LAYOUT(LAYER_CONVOLUTION_1D, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
