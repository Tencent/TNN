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

#ifndef TNN_SOURCE_TNN_DEVICE_X86_X86_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_X86_X86_LAYER_ACC_H_

#include <vector>

#include "tnn/core/abstract_layer_acc.h"
#include "tnn/device/x86/x86_device.h"
#include "tnn/device/x86/x86_util.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/device/x86/acc/compute/jit/utils/cpu_isa.h"

namespace TNN_NS {
using namespace x86;

// @brief x86 layer acc
class X86LayerAcc : public AbstractLayerAcc {
public:
    // @brief virtual destructor
    virtual ~X86LayerAcc();

    virtual Status Init(Context* context, LayerParam* param, LayerResource* resource,
                        const std::vector<Blob*> &inputs, const std::vector<Blob*> &outputs);
    
    virtual Status Reshape(const std::vector<Blob*> &inputs, const std::vector<Blob*> &outputs);

    virtual Status Forward(const std::vector<Blob*> &inputs, const std::vector<Blob*> &outputs);

    virtual Status DoForward(const std::vector<Blob*> &inputs, const std::vector<Blob*> &outputs);

    // @brief allocate or update constant blobs if constant resource change
    // Note: this func may cost much time, call this func only when necessary.
    virtual Status ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob = false);

#if TNN_PROFILE
    Timer timer;
#endif

protected:
    LayerParam* param_          = nullptr;
    LayerResource* resource_    = nullptr;
    X86Context *context_           = nullptr;
    x86_isa_t arch_;

private:
    // @brief return device layer acc support data format
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type);
};

#define DECLARE_X86_ACC(type_string, layer_type)                                                                   \
    class X86##type_string##LayerAcc : public X86LayerAcc {                                                        \
    public:                                                                                                        \
        virtual ~X86##type_string##LayerAcc(){};                                                                   \
        virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;  \
    }

#define REGISTER_X86_ACC(type_string, layer_type)                                                               \
    X86TypeLayerAccRegister<TypeLayerAccCreator<X86##type_string##LayerAcc>> g_x86_##layer_type##_acc_register( \
        layer_type);                                                                                            \

} // TNN_NS

#endif // TNN_SOURCE_TNN_DEVICE_X86_X86_LAYER_ACC_H_
