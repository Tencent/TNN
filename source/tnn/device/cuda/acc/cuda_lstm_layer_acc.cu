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

#include "tnn/device/cuda/acc/cuda_lstm_layer_acc.h"

#include <memory>

#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/core/macro.h"

namespace TNN_NS {

using perf_t = cudnnConvolutionFwdAlgoPerf_t;

Status CudaLSTMONNXLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    CudaLayerAcc::Init(context, param, resource, inputs, outputs);

    LSTMONNXLayerParam * lstm_param = dynamic_cast<LSTMONNXLayerParam *>(param_);
    printf("Init lstm hidden_size:%d direction:%d\n", lstm_param->hidden_size, lstm_param->direction);

    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;

    this->m_rnn_algo = CUDNN_RNN_ALGO_STANDARD;
    // this->m_rnn_algo = CUDNN_RNN_ALGO_PERSIST_DYNAMIC;
    // this->m_rnn_algo = CUDNN_RNN_ALGO_PERSIST_STATIC;

    CUDNN_CHECK(cudnnCreateRNNDescriptor(&this->m_rnn_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&this->m_w_desc));
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&this->m_dropout_desc));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&this->m_hx_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&this->m_cx_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&this->m_hy_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&this->m_cy_desc));
   
    unsigned long long seed = 1337ull; // Pick a seed.
    float dropout = 0;
    size_t stateSize;

    CUDNN_CHECK(cudnnDropoutGetStatesSize(context_->cudnn_handle_, &stateSize));
    RETURN_ON_NEQ(device_->Allocate(&m_dropout_state, stateSize), TNN_OK);
    CUDNN_CHECK(cudnnSetDropoutDescriptor(this->m_dropout_desc, 
                               context_->cudnn_handle_,
                               dropout, 
                               m_dropout_state, 
                               stateSize, 
                               seed));

    return this->Reshape(inputs, outputs);
}

CudaLSTMONNXLayerAcc::~CudaLSTMONNXLayerAcc(){
    if (m_dropout_state) {
        device_->Free(m_dropout_state);
    }
}

Status CudaLSTMONNXLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    DimsVector output_dims = outputs[0]->GetBlobDesc().dims;
    LSTMONNXLayerParam * lstm_param = dynamic_cast<LSTMONNXLayerParam *>(param_);
    printf("reshape lstm hidden_size:%d direction:%d\n", lstm_param->hidden_size, lstm_param->direction);

    // free the last init resources 
    if (m_x_desc && m_seq_length > 0) {
        for (int i = 0; i < this->m_seq_length; i++) {CUDNN_CHECK(cudnnDestroyTensorDescriptor(this->m_x_desc[i])); }
        free(m_x_desc);
        m_x_desc = nullptr;
    }
    if (m_y_desc && m_seq_length > 0) {
        for (int i = 0; i < this->m_seq_length; i++) {CUDNN_CHECK(cudnnDestroyTensorDescriptor(this->m_y_desc[i])); }
        free(m_y_desc);
        m_y_desc = nullptr;
    }

    this->m_hidden_size = lstm_param->hidden_size;
    this->m_num_layers = 1;
    this->m_input_size = DimsVectorUtils::Count(input_dims, 2); // input dimension
    this->m_bidirectional = lstm_param->direction >= 2 ? true : false;

    // currently one onnx lstm layer only compute one time, so num_layers = 1
    this->m_seq_length = input_dims[0];
    int batch_size = input_dims[1];

    printf("Reshape batchsize:%d seq_length:%d num_layers:%d input_size:%d hidden_size:%d, direction:%d\n", 
                batch_size, m_seq_length, m_num_layers, m_input_size, m_hidden_size, lstm_param->direction);

    CUDNN_CHECK(cudnnSetRNNDescriptor_v6(context_->cudnn_handle_,
                                       this->m_rnn_desc,
                                       this->m_hidden_size, 
                                       this->m_num_layers, 
                                       this->m_dropout_desc,
                                       CUDNN_LINEAR_INPUT, 
                                       this->m_bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, 
                                       CUDNN_LSTM, 
                                       this->m_rnn_algo,
                                       CUDNN_DATA_FLOAT));

    this->m_x_desc = (cudnnTensorDescriptor_t*)malloc(this->m_seq_length * sizeof(cudnnTensorDescriptor_t));
    this->m_y_desc = (cudnnTensorDescriptor_t*)malloc(this->m_seq_length * sizeof(cudnnTensorDescriptor_t));

    
    int dimA[3];
    int strideA[3];

    for (int i = 0; i < this->m_seq_length; i++) {
        CUDNN_CHECK( cudnnCreateTensorDescriptor(&(this->m_x_desc[i])) );
        CUDNN_CHECK( cudnnCreateTensorDescriptor(&(this->m_y_desc[i])) );

        dimA[0] = batch_size;
        dimA[1] = this->m_input_size;
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(this->m_x_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));

        dimA[0] = batch_size;
        dimA[1] = this->m_hidden_size * (this->m_bidirectional ? 2 : 1);
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(this->m_y_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
    }
   
   
    dimA[0] = m_num_layers * (m_bidirectional ? 2 : 1);
    dimA[1] = batch_size;
    dimA[2] = m_hidden_size;

    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;

    CUDNN_CHECK(cudnnSetTensorNdDescriptor(m_hx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(m_cx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(m_hy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(m_cy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

   
    size_t weightsSize;
    CUDNN_CHECK(cudnnGetRNNParamsSize(context_->cudnn_handle_, m_rnn_desc, m_x_desc[0], &weightsSize, CUDNN_DATA_FLOAT));

    int dimW[3];   
    dimW[0] =  weightsSize / sizeof(float);
    dimW[1] = 1;
    dimW[2] = 1;

    printf("lstm weightsSize:%lu\n", weightsSize);
      
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(m_w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));   

    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(context_->cudnn_handle_, m_rnn_desc, m_seq_length, m_x_desc, &m_workspace_size));

    if (this->m_workspace_size > 0) {
        RETURN_ON_NEQ(device_->Allocate(&m_workspace, m_workspace_size), TNN_OK);
    }
    if (this->m_workspace == NULL) {
        return Status(TNNERR_LAYER_ERR, "LSTM allocate workspace failed.");
    }

    if (inputs.size() < 4) {
        return Status(TNNERR_LAYER_ERR, "LSTM has invalid inputs");
    }

    // h,c initialize according to batch_size
    size_t h_size = DimsVectorUtils::Count(inputs[1]->GetBlobDesc().dims);
    size_t c_size = DimsVectorUtils::Count(inputs[2]->GetBlobDesc().dims); 
    // float* h = inputs[1]->GetHandle().base +  inputs[1]->GetHandle().bytes_offset;
    // float* c = inputs[2]->GetHandle().base +  inputs[2]->GetHandle().bytes_offset;

    // h_size = this->m_input_nodes[1]->count();
    // c_size = this->m_input_nodes[2]->count();
    // h = this->m_input_nodes[1]->data();
    // c = this->m_input_nodes[2]->data();
    // m_weights = this->m_input_nodes[3]->data();


    // this->m_hx = (float *)(this->m_mngr->myalloc(h_size * sizeof(float) * batch_size , false));
    // this->m_hy = (float *)(this->m_mngr->myalloc(h_size * sizeof(float) * batch_size , false));
    // this->m_cx = (float *)(this->m_mngr->myalloc(c_size * sizeof(float) * batch_size , false));
    // this->m_cy = (float *)(this->m_mngr->myalloc(c_size * sizeof(float) * batch_size , false));

    // CUDA_CHECK(cudaMemset(this->m_hx , 0, batch_size * h_size * sizeof(float)));
    // CUDA_CHECK(cudaMemset(this->m_hy , 0, batch_size * h_size * sizeof(float)));
    // CUDA_CHECK(cudaMemset(this->m_cx , 0, batch_size * c_size * sizeof(float)));
    // CUDA_CHECK(cudaMemset(this->m_cy , 0, batch_size * c_size * sizeof(float)));


    // set lstm algo persist plan 
    if (this->m_rnn_algo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
      // Note: This step is expensive. Once completed the plan can be reused so long as the descriptor
      //       batchsize or datatype don't change.
      CUDNN_CHECK(cudnnCreatePersistentRNNPlan(this->m_rnn_desc, batch_size, CUDNN_DATA_FLOAT, &this->m_rnn_plan));
      CUDNN_CHECK(cudnnSetPersistentRNNPlan(this->m_rnn_desc, this->m_rnn_plan));
    }
   

    return TNN_OK;
}

Status CudaLSTMONNXLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

REGISTER_CUDA_ACC(LSTMONNX, LAYER_LSTMONNX);

}  // namespace TNN_NS
