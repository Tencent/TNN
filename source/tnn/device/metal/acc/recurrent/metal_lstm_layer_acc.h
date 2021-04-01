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

#ifndef TNN_SOURCE_TNN_DEVICE_METAL_METAL_LSTM_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_METAL_METAL_LSTM_LAYER_ACC_H_

#include "tnn/device/metal/acc/metal_layer_acc.h"

namespace TNN_NS {

// @brief conv layer metal acc
class MetalLSTMLayerAcc : public MetalLayerAcc {
public:
    Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                const std::vector<Blob *> &outputs);

    // @brief virtual destrcutor
    virtual ~MetalLSTMLayerAcc();

    virtual Status AllocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
protected:
    virtual Status AllocateBufferWeights(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status AllocateBufferBias(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status AllocateBufferStates(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
protected:
    // weight buffer
    id<MTLBuffer> buffer_wi_ = nil;
    id<MTLBuffer> buffer_wh_ = nil;
    id<MTLBuffer> buffer_bias_ = nil;
    // initial state buffer
    id<MTLBuffer> buffer_h0_ = nil;
    id<MTLBuffer> buffer_c0_ = nil;
    // gates buffer
    id<MTLBuffer> buffer_gates_ = nil;
    virtual bool UseNaiveConstantBlobs() {return true;}
private:
    virtual std::vector<DataFormat> SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type);

};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_METAL_METAL_LSTM_LAYER_ACC_H_
