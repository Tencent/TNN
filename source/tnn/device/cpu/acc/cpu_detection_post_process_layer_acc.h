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

#ifndef TNN_SOURCE_TNN_DEVICE_CPU_ACC_CPU_DETECTION_POST_PROCESS_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_CPU_ACC_CPU_DETECTION_POST_PROCESS_LAYER_ACC_H_

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/device/cpu/cpu_device.h"
#include "tnn/utils/bbox_util.h"

namespace TNN_NS {

// @brief Detection Output layer cpu acc
class CpuDetectionPostProcessLayerAcc : public CpuLayerAcc {
    // @brief virtual destrcutor
    virtual ~CpuDetectionPostProcessLayerAcc();

    /**
     * @brief input or output blobs reshape.
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return reshape result
     */
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    /**
     * @brief layer forward
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return execution result
     */
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_CPU_ACC_CPU_DETECTION_POST_PROCESS_LAYER_ACC_H_
