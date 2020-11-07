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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_CONV_LAYER_ACC_IMPL_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_CONV_LAYER_ACC_IMPL_H_

#include "tnn/device/opencl/acc/opencl_layer_acc.h"

namespace TNN_NS {

struct OpenCLConvParam {
    int input_channel;
    int output_channel;
    int kernel_x;
    int kernel_y;
    int pad_x;
    int pad_y;
    int stride_x;
    int stride_y;
    int dilation_x;
    int dilation_y;

    int pad_type;
    int group;
    int has_bias;
    int activation_type;
};

enum ConvType { CT_CONV_COMMON = 0, CT_CONV_1x1, CT_CONV_DEPTHWISE };

class OpenCLConvLayerAccImpl : public OpenCLLayerAcc {
public:
    OpenCLConvLayerAccImpl();
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLConvLayerAccImpl() override;

#if TNN_PROFILE
    virtual double GetFlops() override;
#endif

protected:
    Status AllocateWeightsBias(LayerResource *resource);
    std::vector<uint32_t> Conv2dCommonLocalWS2D(std::vector<uint32_t> &gws, const uint32_t max_workgroup_size,
                                                const uint32_t subgroup_size = 0);
    std::vector<uint32_t> Conv2dCommonLocalWS3DGeneral(std::vector<uint32_t> &gws, const uint32_t kernel_size,
                                                const uint32_t max_workgroup_size);

    std::vector<uint32_t> Conv2dCommonLocalWS3DKernel3x3(std::vector<uint32_t> &gws, const uint32_t kernel_size,
                                                const uint32_t max_workgroup_size);

private:
    Status ConvertWeights(float *weights_data_ptr);

protected:
    OpenCLConvParam conv_params_ = {0};
    shared_ptr<OpenCLMemory> ocl_weights_;
    shared_ptr<OpenCLMemory> ocl_bias_;
    ConvType conv_type_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_CONV_LAYER_ACC_IMPL_H_
