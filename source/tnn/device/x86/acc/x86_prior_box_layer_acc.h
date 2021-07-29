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

#ifndef TNN_SOURCE_TNN_DEVICE_X86_ACC_PRIOR_BOX_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_X86_ACC_PRIOR_BOX_LAYER_ACC_H_

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/device/x86/x86_device.h"

namespace TNN_NS {

// @brief conv layer cpu acc
class X86PriorBoxLayerAcc : public X86LayerAcc {
    // @brief virtual destrcutor
    virtual ~X86PriorBoxLayerAcc();
    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

private:
    template <typename T>
    inline void set_value(const int N, const T alpha, T *Y) {
        if (alpha == 0) {
            memset(Y, 0, sizeof(T) * N);  // NOLINT(caffe/alt_fn)
            return;
        }
        for (int i = 0; i < N; ++i) {
            Y[i] = alpha;
        }
    }

    template <typename T>
    void compute(Blob *output_blob, T *output_data, PriorBoxLayerParam *param, int layer_height, int layer_width,
                 int img_height, int img_width, float step_h, float step_w);
};
}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_X86_ACC_PRIOR_BOX_LAYER_ACC_H_
