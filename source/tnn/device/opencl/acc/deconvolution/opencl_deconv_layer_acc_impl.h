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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_DECONV_LAYER_ACC_IMPL_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_DECONV_LAYER_ACC_IMPL_H_

#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_acc_impl.h"
#include "tnn/device/opencl/opencl_memory.h"
namespace TNN_NS {

enum DeConvType { CT_DECONV_COMMON = 0, CT_DECONV_DEPTHWISE };

class OpenCLDeconvLayerAccImpl : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLDeconvLayerAccImpl() override;

#if TNN_PROFILE
    virtual double GetFlops() override;
#endif

private:
    virtual void SetExtraKernelParameters(uint32_t idx, const std::vector<Blob *> &inputs,
                                          const std::vector<Blob *> &outputs);
    Status ConvertWeights(float *weights_data_ptr);

    std::string GenerateTuneKernelKey(OpenCLExecuteUnit &unit);

protected:
    OpenCLConvParam deconv_params_;
    shared_ptr<OpenCLMemory> ocl_weights_;
    shared_ptr<OpenCLMemory> ocl_bias_;
    DeConvType deconv_type_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_DECONV_LAYER_ACC_IMPL_H_
