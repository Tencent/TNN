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

#ifndef TNN_SOURCE_TNN_DEVICE_DIRECTX_ACC_DIRECTX_BINARY_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_DIRECTX_ACC_DIRECTX_BINARY_LAYER_ACC_H_

#include "tnn/device/directx/acc/directx_layer_acc.h"

namespace TNN_NS {

namespace directx {

class DirectXBinaryLayerAcc : public DirectXLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~DirectXBinaryLayerAcc() override;

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob = false) override;

#if TNN_PROFILE
    virtual double GetBandwidth() override;
#endif

private:

    Status ConvertParam(float *bias_data_ptr, std::vector<int> param_dims);

    Status CalcStrides(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    std::string kernel_name_ = "";
    MultidirBroadcastLayerParam broadcast_param_;

private:
    std::shared_ptr<DirectXMemory> binary_params_ = nullptr;
    std::shared_ptr<ID3D11Buffer> const_buffer_;

    std::vector<int> param_dims_ = {};

    unsigned int input_a_stride_[6];
    unsigned int input_b_stride_[6];
    unsigned int output_dim_[6];

    size_t output_dims_size_;

};

#define DECLARE_DIRECTX_BINARY_ACC(type_string)                                                                        \
    class DirectX##type_string##LayerAcc : public DirectXBinaryLayerAcc {                                              \
    public:                                                                                                            \
        virtual Status Init(Context *context, LayerParam *param, LayerResource *resource,                              \
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;           \
        virtual ~DirectX##type_string##LayerAcc() override;                                                            \
    }

} // namespace directx

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_DIRECTX_ACC_DIRECTX_BINARY_LAYER_ACC_H_
