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

#ifndef TNN_SOURCE_TNN_DEVICE_ARM_ARM_LSTM_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_ARM_ARM_LSTM_LAYER_ACC_H_

#include "tnn/device/arm/acc/arm_layer_acc.h"

namespace TNN_NS {

class ArmLSTMONNXLayerAcc : public ArmLayerAcc {
public:
    virtual ~ArmLSTMONNXLayerAcc();

    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs);

    virtual Status DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:
    Status AllocateBufferWeightInput(Blob *weight_i);
    Status AllocateBufferWeightRecurrent(Blob *weight_r);
    Status AllocateBufferBias(Blob *bias);

    RawBuffer buffer_weight_input_;
    RawBuffer buffer_weight_recurrent_;
    RawBuffer buffer_bias_;

    int direction_;
    int num_directions_;
    int input_size_;
    int hidden_size_;
    int seq_len_;

private:
    Status LstmSingleDirection(const float *x, float *y, const float *w, const float *r, const float *b, float *h_t,
                               float *c_t, const int batch_size, int reverse);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_ARM_ARM_LSTM_LAYER_ACC_H_
