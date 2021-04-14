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

#ifndef TNN_SOURCE_TNN_DEVICE_X86_X86_PERMUTE_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_X86_X86_PERMUTE_LAYER_ACC_H_

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/device/x86/x86_device.h"

namespace TNN_NS {

class X86PermuteLayerAcc : public X86LayerAcc {
    virtual ~X86PermuteLayerAcc();

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    /**
     * @brief Compute the volume of a slice; i.e., the product of dimensions
     *        among a range of axes.
     *
     * @param dimes the dimensions
     *
     * @param start_axis The first axis to include in the slice.
     *
     */
    inline int count(std::vector<int> dimes, int start_axis) const {
        const int end_axis = int(dimes.size());
        ASSERT(start_axis <= end_axis);
        ASSERT(start_axis >= 0);
        ASSERT(end_axis >= 0);
        int count = 1;
        for (int i = start_axis; i < end_axis; ++i) {
            count *= dimes[i];
        }
        return count;
    };
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_X86_X86_PERMUTE_LAYER_ACC_H_
