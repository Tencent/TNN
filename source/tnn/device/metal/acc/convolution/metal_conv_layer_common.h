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

#ifndef TNN_SOURCE_TNN_DEVICE_METAL_METAL_CONV_LAYER_ACC_COMMON_H_
#define TNN_SOURCE_TNN_DEVICE_METAL_METAL_CONV_LAYER_ACC_COMMON_H_

#include "tnn/device/metal/acc/metal_layer_acc.h"

namespace TNN_NS {

// @brief conv layer metal acc
class MetalConvLayerCommon : public MetalLayerAcc {
public:
    virtual ~MetalConvLayerCommon();

    Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    /**
     * @brief layer forward
     * @param param    convolution para
     * @param inputs    input blobs
     * @param outputs   output blobs
     * @return implement is prefered
     */
    static bool isPrefered(ConvLayerParam *param, const std::vector<Blob *> &inputs,
                           const std::vector<Blob *> &outputs);

    /**
     * @brief allocate MTLBuffer for weigths
     * @param inputs    input blobs
     * @param outputs   output blobs
     */
    virtual Status AllocateBufferWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    /**
     * @brief allocate MTLBuffer for biase
     * @param inputs    input blobs
     * @param outputs   output blobs
     */
    virtual Status AllocateBufferBias(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

    virtual Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
public:
    virtual std::string KernelName(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status ComputeThreadSize(const std::vector<Blob *> &inputs,
                                     const std::vector<Blob *> &outputs,
                                     MTLSize &size);
protected:
    id<MTLBuffer> buffer_weight_ = nil;
    id<MTLBuffer> buffer_bias_   = nil;
    bool is_channel_4x_ = false;
    int bias_datatype_bytes_ = 0;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_METAL_METAL_CONV_LAYER_ACC_COMMON_H_
