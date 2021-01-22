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

#ifndef TNN_SOURCE_TNN_DEVICE_X86_ACC_X86_ARG_MAX_OR_MIN_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_X86_ACC_X86_ARG_MAX_OR_MIN_LAYER_ACC_H_

#include <vector>

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/device/x86/x86_device.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

typedef struct x86_arg_max_or_min_opeartor {
public:
    virtual Status Init() {
        return TNN_OK;
    }

    virtual void operator()(const int idx, const float val){}

    int get_idx() {
        return guard_idx;
    }

protected:
    float guard_val;
    int guard_idx;
} X86_ARG_MAX_OR_MIN_OP;

typedef struct x86_arg_max_operator : X86_ARG_MAX_OR_MIN_OP {
    Status Init() {
        guard_val = -FLT_MAX;
        guard_idx = 0;
        return TNN_OK;
    }

    void operator()(const int idx, const float val) {
        guard_idx = val > guard_val ? idx : guard_idx;
        guard_val = val > guard_val ? val : guard_val;
    }
} X86_ARG_MAX_OP;

typedef struct x86_arg_min_operator : X86_ARG_MAX_OR_MIN_OP {
    Status Init() {
        guard_val = FLT_MAX;
        guard_idx = 0;
        return TNN_OK;
    }

    void operator()(const int idx, const float val) {
        guard_idx = val < guard_val ? idx : guard_idx;
        guard_val = val < guard_val ? val : guard_val;
    }
} X86_ARG_MIN_OP;

class X86ArgMaxOrMinLayerAcc : public X86LayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource* resource, const std::vector<Blob*> &inputs,
                        const std::vector<Blob *> &outputs) override;
    
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    std::shared_ptr<X86_ARG_MAX_OR_MIN_OP> op_;
    int num_, channels_, stride_;
};

}

#endif