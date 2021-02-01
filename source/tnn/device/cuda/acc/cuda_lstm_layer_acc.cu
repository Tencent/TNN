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

#include "tnn/core/macro.h"
#include "tnn/device/cuda/acc/cuda_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

/* 
    CUDNN LSTM Weight Storage Format:
        Concat(
            [4, hidden_size, input_size],   // ifco Weight For input 
            [4, hidden_size, hidden_size],  // ifco Weight For reccurent 
            *[4, hidden_size, input_size],  // ifco Backward Weight For input, only exists in bidirection mode
            *[4, hidden_size, hidden_size], // ifco Backward Weight For reccurent, only exists in bidirection mode
            [4, hidden_size],               // ifco Bias for input
            [4, hidden_size],               // ifco Bias for reccurent
            *[4, hidden_size],               // ifco Backward Bias for input, only exists in bidirection mode
            *[4, hidden_size],               // ifco Backward Bias for recurent, only exists in bidirection mode
        )
*/
Status PackONNXWeightsToCUDNNFormat(Blob * W, Blob * R, Blob* B, 
                                    const int directions, const int hidden_size, const int input_size, 
                                    float * cudnn_weight_ptr)
{
    // 1. Check blob volumn
    if (DimsVectorUtils::Count(W->GetBlobDesc().dims) != directions * 4 * hidden_size * input_size) {
        LOGE("Blob W has invalid volumn\n");
        return  TNNERR_LAYER_ERR;
    }

    if (DimsVectorUtils::Count(R->GetBlobDesc().dims) != directions * 4 * hidden_size * hidden_size) {
        LOGE("Blob R has invalid volumn\n");
        return  TNNERR_LAYER_ERR;
    }

    if (DimsVectorUtils::Count(B->GetBlobDesc().dims) != directions * 8 * hidden_size) {
        LOGE("Blob B has invalid volumn\n");
        return  TNNERR_LAYER_ERR;
    }

    const int gate_offset[4] = {0, 2, 3, 1}; // IOFC -> IFCO

    // [num_directions, 4*hidden_size, input_size].
    float * W_ptr = (float*)(((char*)W->GetHandle().base) + W->GetHandle().bytes_offset);
    // [num_directions, 4*hidden_size, hidden_size].
    float * R_ptr = (float*)(((char*)R->GetHandle().base) + R->GetHandle().bytes_offset);
    // [num_directions, 8*hidden_size].
    float * B_ptr = (float*)(((char*)B->GetHandle().base) + B->GetHandle().bytes_offset);

    size_t offset = 0;
    for(int dire = 0; dire < directions; dire++) {
        // W
        for(int g=0;g<4;g++) {
            CUDA_CHECK(cudaMemcpy(cudnn_weight_ptr + offset, 
                                  W_ptr + (dire * 4 + gate_offset[g]) * hidden_size * input_size,
                                  hidden_size * input_size * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            offset += hidden_size * input_size;
        }
        // R
        for(int g=0;g<4;g++) {
            CUDA_CHECK(cudaMemcpy(cudnn_weight_ptr + offset, 
                                  R_ptr + (dire * 4 + gate_offset[g]) * hidden_size * hidden_size,
                                  hidden_size * hidden_size * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            offset += hidden_size * hidden_size;
        }
    }

    for(int dire = 0; dire < directions; dire++) {
        // WB
        for(int g=0;g<4;g++) {
            CUDA_CHECK(cudaMemcpy(cudnn_weight_ptr + offset, 
                                  B_ptr + (dire * 8 + gate_offset[g]) * hidden_size,
                                  hidden_size * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            offset += hidden_size;
        }
        // RB
        for(int g=0;g<4;g++) {
            CUDA_CHECK(cudaMemcpy(cudnn_weight_ptr + offset, 
                                  B_ptr + (dire * 8 + 4 + gate_offset[g]) * hidden_size,
                                  hidden_size * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            offset += hidden_size;
        }
    }

    return TNN_OK;
}

Status CudaLSTMONNXLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
        const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    CudaLayerAcc::Init(context, param, resource, inputs, outputs);

    rnn_algo_ = CUDNN_RNN_ALGO_STANDARD;
    // rnn_algo_ = CUDNN_RNN_ALGO_PERSIST_DYNAMIC;
    // rnn_algo_ = CUDNN_RNN_ALGO_PERSIST_STATIC;

    CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnn_desc_));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc_));
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&hx_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&cx_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&hy_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&cy_desc_));
   
    unsigned long long seed = 1337ull; // Pick a seed.
    float dropout = 0;
    size_t stateSize;

    CUDNN_CHECK(cudnnDropoutGetStatesSize(context_->cudnn_handle_, &stateSize));
    RETURN_ON_NEQ(device_->Allocate(&dropout_state_, stateSize), TNN_OK);
    CUDNN_CHECK(cudnnSetDropoutDescriptor(dropout_desc_, 
                               context_->cudnn_handle_,
                               dropout, 
                               dropout_state_, 
                               stateSize, 
                               seed));

    return this->Reshape(inputs, outputs);
}

CudaLSTMONNXLayerAcc::~CudaLSTMONNXLayerAcc(){
    CUDNN_CHECK(cudnnDestroyRNNDescriptor(rnn_desc_));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc_));
    CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(hx_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(cx_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(hy_desc_));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(cy_desc_));

    if (dropout_state_) {
        device_->Free(dropout_state_);
        dropout_state_ = nullptr;
    }

    if (x_desc_ && seq_length_ > 0) {
        for (int i = 0; i < seq_length_; i++) {CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc_[i])); }
        free(x_desc_);
        x_desc_ = nullptr;
    }
    if (y_desc_ && seq_length_ > 0) {
        for (int i = 0; i < seq_length_; i++) {CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc_[i])); }
        free(y_desc_);
        y_desc_ = nullptr;
    }

    if (hx_) {
        device_->Free(hx_);
        hx_ = nullptr;
    }
    if (hy_) {
        device_->Free(hy_);
        hy_ = nullptr;
    }
    if (cx_) {
        device_->Free(cx_);
        cx_ = nullptr;
    }
    if (cy_) {
        device_->Free(cy_);
        cy_ = nullptr;
    }
    if (workspace_) {
        device_->Free(workspace_);
        workspace_= nullptr;
        workspace_size_ = 0;
    }

    if (rnn_algo_ == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
        cudnnDestroyPersistentRNNPlan(rnn_plan_);
    }
}

Status CudaLSTMONNXLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    DimsVector input_dims  = inputs[0]->GetBlobDesc().dims;
    LSTMONNXLayerParam * lstm_param = dynamic_cast<LSTMONNXLayerParam *>(param_);

    if (inputs.size() < 4) {
        return Status(TNNERR_LAYER_ERR, "LSTM has invalid inputs");
    }

    // free the last init resources 
    if (x_desc_ && seq_length_ > 0) {
        for (int i = 0; i < seq_length_; i++) {CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc_[i])); }
        free(x_desc_);
        x_desc_ = nullptr;
    }
    if (y_desc_ && seq_length_ > 0) {
        for (int i = 0; i < seq_length_; i++) {CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc_[i])); }
        free(y_desc_);
        y_desc_ = nullptr;
    }

    hidden_size_ = lstm_param->hidden_size;
    num_layers_ = 1;
    input_size_ = DimsVectorUtils::Count(input_dims, 2); // input dimension
    bidirectional_ = lstm_param->direction >= 2 ? true : false;

    // currently one onnx lstm layer only compute one time, so num_layers = 1
    seq_length_ = input_dims[0];
    int batch_size = input_dims[1];

    CUDNN_CHECK(cudnnSetRNNDescriptor_v6(context_->cudnn_handle_,
                                       rnn_desc_,
                                       hidden_size_, 
                                       num_layers_, 
                                       dropout_desc_,
                                       CUDNN_LINEAR_INPUT, 
                                       bidirectional_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, 
                                       CUDNN_LSTM, 
                                       rnn_algo_,
                                       CUDNN_DATA_FLOAT));

    // xy initialize
    x_desc_ = (cudnnTensorDescriptor_t*)malloc(seq_length_ * sizeof(cudnnTensorDescriptor_t));
    y_desc_ = (cudnnTensorDescriptor_t*)malloc(seq_length_ * sizeof(cudnnTensorDescriptor_t));

    
    int dimA[3];
    int strideA[3];

    for (int i = 0; i < seq_length_; i++) {
        CUDNN_CHECK( cudnnCreateTensorDescriptor(&(x_desc_[i])) );
        CUDNN_CHECK( cudnnCreateTensorDescriptor(&(y_desc_[i])) );

        dimA[0] = batch_size;
        dimA[1] = input_size_;
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(x_desc_[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));

        dimA[0] = batch_size;
        dimA[1] = hidden_size_ * (bidirectional_ ? 2 : 1);
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(y_desc_[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
    }
   
   
    // hc initialize
    dimA[0] = num_layers_ * (bidirectional_ ? 2 : 1);
    dimA[1] = batch_size;
    dimA[2] = hidden_size_;

    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;

    CUDNN_CHECK(cudnnSetTensorNdDescriptor(hx_desc_, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(cx_desc_, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(hy_desc_, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(cy_desc_, CUDNN_DATA_FLOAT, 3, dimA, strideA));


    size_t hc_size_in_bytes = (bidirectional_ ? 2 : 1) * batch_size * hidden_size_ * sizeof(float);
    RETURN_ON_NEQ(device_->ReAllocate((void **)&hx_, hc_size_in_bytes), TNN_OK);
    RETURN_ON_NEQ(device_->ReAllocate((void **)&hy_, hc_size_in_bytes), TNN_OK);
    RETURN_ON_NEQ(device_->ReAllocate((void **)&cx_, hc_size_in_bytes), TNN_OK);
    RETURN_ON_NEQ(device_->ReAllocate((void **)&cy_, hc_size_in_bytes), TNN_OK);

    CUDA_CHECK(cudaMemset(hy_, 0, hc_size_in_bytes));
    CUDA_CHECK(cudaMemset(cy_, 0, hc_size_in_bytes));

    if (inputs.size() >= 6) {
        // [num_directions, batch_size, hidden_size].
        float * h0_ptr = (float*)(((char*)inputs[4]->GetHandle().base) + inputs[4]->GetHandle().bytes_offset);
        float * c0_ptr = (float*)(((char*)inputs[5]->GetHandle().base) + inputs[5]->GetHandle().bytes_offset);
        CUDA_CHECK(cudaMemcpy(hx_, h0_ptr, hc_size_in_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(cx_, c0_ptr, hc_size_in_bytes, cudaMemcpyDeviceToDevice));
    } else {
        CUDA_CHECK(cudaMemset(hx_, 0, hc_size_in_bytes));
        CUDA_CHECK(cudaMemset(cx_, 0, hc_size_in_bytes));
    }
   
    // weight initialize
    size_t weightsSize;
    CUDNN_CHECK(cudnnGetRNNParamsSize(context_->cudnn_handle_, rnn_desc_, x_desc_[0], &weightsSize, CUDNN_DATA_FLOAT));

    int dimW[3];   
    dimW[0] =  weightsSize / sizeof(float);
    dimW[1] = 1;
    dimW[2] = 1;

    CUDNN_CHECK(cudnnSetFilterNdDescriptor(w_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));   

    RETURN_ON_NEQ(device_->ReAllocate((void **)&weights_, weightsSize), TNN_OK);
    RETURN_ON_NEQ(PackONNXWeightsToCUDNNFormat(inputs[1], inputs[2], inputs[3], 
                                               num_layers_ * (bidirectional_ ? 2 : 1), hidden_size_, input_size_,
                                               (float*)weights_), 
                  TNN_OK);

    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(context_->cudnn_handle_, rnn_desc_, seq_length_, x_desc_, &workspace_size_));

    if (workspace_size_ > 0) {
        RETURN_ON_NEQ(device_->ReAllocate(&workspace_, workspace_size_), TNN_OK);
    }

    // set lstm algo persist plan 
    if (rnn_algo_ == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
      // Note: This step is expensive. Once completed the plan can be reused so long as the descriptor
      CUDNN_CHECK(cudnnCreatePersistentRNNPlan(rnn_desc_, batch_size, CUDNN_DATA_FLOAT, &rnn_plan_));
      CUDNN_CHECK(cudnnSetPersistentRNNPlan(rnn_desc_, rnn_plan_));
    }

    return TNN_OK;
}

Status CudaLSTMONNXLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    float * bottom_data = (float*)(((char*)inputs[0]->GetHandle().base) + inputs[0]->GetHandle().bytes_offset);
    float * top_data    = (float*)(((char*)outputs[0]->GetHandle().base) + outputs[0]->GetHandle().bytes_offset);

    CUDNN_CHECK(cudnnRNNForwardInference(context_->cudnn_handle_,
                                         rnn_desc_, 
                                         seq_length_,
                                         x_desc_, 
                                         bottom_data, 
                                         hx_desc_,
                                         hx_, 
                                         cx_desc_,
                                         cx_, 
                                         w_desc_,
                                         weights_,
                                         y_desc_,
                                         top_data,
                                         hy_desc_, 
                                         hy_,
                                         cy_desc_, 
                                         cy_,
                                         workspace_,
                                         workspace_size_));
    return TNN_OK;
}

REGISTER_CUDA_ACC(LSTMONNX, LAYER_LSTMONNX);

}  // namespace TNN_NS
