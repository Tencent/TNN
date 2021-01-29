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

    int m_num_layers;
    int m_seq_length;
    int m_hidden_size;
    int m_input_size;
    bool m_bidirectional;

    cudnnRNNAlgo_t m_rnn_algo;

    cudnnRNNDescriptor_t m_rnn_desc;
    cudnnFilterDescriptor_t m_w_desc;
    cudnnPersistentRNNPlan_t m_rnn_plan;
    cudnnDropoutDescriptor_t m_dropout_desc;

    cudnnTensorDescriptor_t *m_x_desc, *m_y_desc;
    cudnnTensorDescriptor_t m_hx_desc, m_cx_desc;
    cudnnTensorDescriptor_t m_hy_desc, m_cy_desc;

    float * m_hx;
    float * m_hy;
    float * m_cx;
    float * m_cy;

    void * m_workspace;
    void * m_dropout_state;
    size_t m_workspace_size;
    const float* m_weights;
    
};

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_DEVICE_CUDA_ACC_CUDA_LSTM_LAYER_ACC_H_