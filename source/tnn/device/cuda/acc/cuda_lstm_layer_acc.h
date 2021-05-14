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

#ifndef TNN_SOURCE_TNN_DEVICE_CUDA_ACC_CUDA_LSTM_LAYER_ACC_H_
#define TNN_SOURCE_TNN_DEVICE_CUDA_ACC_CUDA_LSTM_LAYER_ACC_H_

#include "tnn/device/cuda/acc/cuda_layer_acc.h"

namespace TNN_NS {

class CudaLSTMONNXLayerAcc : public CudaLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual ~CudaLSTMONNXLayerAcc();
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

protected:

    int num_layers_;
    int seq_length_;
    int hidden_size_;
    int input_size_;
    bool bidirectional_;

    cudnnRNNAlgo_t rnn_algo_;

    cudnnRNNDescriptor_t rnn_desc_;
    cudnnFilterDescriptor_t w_desc_;
    cudnnPersistentRNNPlan_t rnn_plan_;
    cudnnDropoutDescriptor_t dropout_desc_;

    cudnnTensorDescriptor_t *x_desc_, *y_desc_;
    cudnnTensorDescriptor_t hx_desc_, cx_desc_;
    cudnnTensorDescriptor_t hy_desc_, cy_desc_;

    float * hx_ = nullptr;
    float * hy_ = nullptr;
    float * cx_ = nullptr;
    float * cy_ = nullptr;

    void * workspace_ = nullptr;
    void * dropout_state_ = nullptr;
    size_t workspace_size_ = 0;
    float* weights_ = nullptr;
    
};

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_DEVICE_CUDA_ACC_CUDA_LSTM_LAYER_ACC_H_