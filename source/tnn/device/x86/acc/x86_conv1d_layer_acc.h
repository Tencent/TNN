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

#ifndef TNN_SOURCE_TNN_DEVICE_X86_ACC_X86_CONV1D_LAYER_ACC_H
#define TNN_SOURCE_TNN_DEVICE_X86_ACC_X86_CONV1D_LAYER_ACC_H

#include <vector>

#include "tnn/core/blob.h"
#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/device/x86/x86_device.h"

namespace TNN_NS {

class X86Conv1DLayerAcc : public X86LayerAcc {
public:
    virtual ~X86Conv1DLayerAcc(){};
    
    Status Init(Context *context, LayerParam *param, LayerResource *resource,
                const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

protected:
    std::shared_ptr<X86LayerAcc> conv_acc_impl_ = nullptr;
    std::shared_ptr<LayerResource> conv_acc_f32_resource_ = nullptr;
};

}   // namespace TNN_NS
#endif