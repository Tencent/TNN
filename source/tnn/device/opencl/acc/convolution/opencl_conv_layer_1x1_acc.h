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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_CONV_LAYER_1X1_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_CONV_LAYER_1X1_ACC_H_

#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_acc_impl.h"
#include "tnn/device/opencl/opencl_memory.h"
namespace TNN_NS {

class OpenCLConvLayer1x1Acc : public OpenCLConvLayerAccImpl {
public:
    static bool IsPrefered(const ConvLayerParam *param, const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs);

    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLConvLayer1x1Acc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    std::vector<uint32_t> Conv2d1x1LocalWS3D(std::vector<uint32_t> &gws, const uint32_t max_workgroup_size);

    bool stride_is_1_ = false;
    bool width_blocking_is_1_ = false;
    bool run_local_work_ = false;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_ACC_OPENCL_CONV_LAYER_1X1_ACC_H_
