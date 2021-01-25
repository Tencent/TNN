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

#ifndef TNN_SOURCE_TNN_DEVICE_X86_X86_INNER_PRODUCT_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_X86_X86_INNER_PRODUCT_LAYER_ACC_H_

#include "tnn/device/x86/acc/x86_layer_acc.h"

namespace TNN_NS {

class X86InnerProductLayerAcc : public X86LayerAcc {
public:
    virtual ~X86InnerProductLayerAcc();

    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status allocateBufferWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status allocateBufferBias(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    RawBuffer buffer_weight_;
    RawBuffer buffer_bias_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_X86_X86_INNER_PRODUCT_LAYER_ACC_H_
