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

#ifndef TNN_SOURCE_TNN_DEVICE_X86_ACC_X86_DECONV_LAYER_ACC_H
#define TNN_SOURCE_TNN_DEVICE_X86_ACC_X86_DECONV_LAYER_ACC_H

#include <vector>

#include "tnn/core/blob.h"
#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/device/x86/x86_device.h"

namespace TNN_NS {

// class X86DeconvLayerAcc : public X86LayerAcc {
// public:
//     virtual ~X86DeconvLayerAcc(){};
    
//     Status Init(Context *context, LayerParam *param, LayerResource *resource,
//                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    
//     virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
//     virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

// protected:
//     bool do_im2col_ = true;
//     RawBuffer col_buffer_;
// private:
//     size_t col_offset_;
//     size_t weight_offset_;
//     size_t conv_in_width_;
//     size_t conv_in_height_;
//     size_t conv_in_channels_;
//     size_t conv_out_channles_;
//     size_t conv_out_spatial_dim_;
//     size_t kernel_dim_;
//     size_t conv_in_offset_;
//     size_t output_offset_;
// };

class X86DeconvLayerAcc : public X86LayerAcc {
public:
    virtual ~X86DeconvLayerAcc(){};
    
    Status Init(Context *context, LayerParam *param, LayerResource *resource,
                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

protected:
    std::shared_ptr<X86LayerAcc> conv_acc_impl_ = nullptr;
    std::shared_ptr<LayerResource> conv_acc_f32_resource_ = nullptr;
};

}   // namespace TNN_NS
#endif