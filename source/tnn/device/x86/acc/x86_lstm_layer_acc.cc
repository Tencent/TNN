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

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/device/x86/acc/x86_lstm_layer_acc.h"
#include "tnn/device/x86/acc/Float4.h"
#include "tnn/utils/omp_utils.h"
namespace TNN_NS {

static void X86LSTMActivate(const float *gates, float *h_t, float *c_t, float *y, int len) {
    int len_vec  = len / 4 * 4;
    OMP_PARALLEL_FOR_GUIDED_
    for (int i = 0; i < len_vec; i += 4) {
        Float4x4 vec = Float4x4::ld4u(gates + i * 4);
        Float4 I, O, F, C;
        vec.get_lane(I, 0);
        vec.get_lane(O, 1);
        vec.get_lane(F, 2);
        vec.get_lane(C, 3);

        I = Float4::sigmoid(I);
        O = Float4::sigmoid(O);
        F = Float4::sigmoid(F);
        C = Float4::tanh(C);

        Float4 cell2_vec = F * Float4::loadu(c_t + i) + I * C;
        Float4 h_vec = O * Float4::tanh(cell2_vec);
        Float4::saveu(c_t + i, cell2_vec);
        Float4::saveu(h_t + i, h_vec);
        Float4::saveu(y + i, h_vec);
    }
    for (int i = len_vec; i < len; i++) {
        float I = gates[i * 4];
        float O = gates[i * 4 + 1];
        float F = gates[i * 4 + 2];
        float C = gates[i * 4 + 3];

        I = 1.f / (1.f + exp(-I));
        F = 1.f / (1.f + exp(-F));
        O = 1.f / (1.f + exp(-O));
        C = tanh(C);

        float cell2 = F * c_t[i] + I * C;
        float H = O * tanh(cell2);
        c_t[i] = cell2;
        h_t[i] = H;
        y[i] = H;
    }
}

Status X86LSTMONNXLayerAcc::LSTMOneDirection(const float *x, float *y, const float *w, const float *r,
                              const float *b, float *h_t, float *c_t, int seq_len, int batch_size,
                              int input_size, int hidden_size, int reverse) {
    int k_c = conv_gemm_conf_.K_c_;
    int n_block = conv_gemm_conf_.n_block_;

    // sgemm for weight tensor
    // weights: [4*hidden_size, input_size]
    // inputs: [seq_len, batch, input_size]
    int K = input_size;
    int N = seq_len * batch_size;
    int M = 4 * hidden_size;

    // two temp buf: gemm_buf and gates_buf
    size_t gemm_buf_size = ROUND_UP(k_c * ROUND_UP(N, n_block) * sizeof(float), 32);
    size_t gates_buf_size = ROUND_UP(N * M * sizeof(float), 32);
    size_t workspace_size = gemm_buf_size + gates_buf_size;
    float *workspace = reinterpret_cast<float *>(context_->GetSharedWorkSpace(workspace_size));
    float *gemm_buf = workspace;
    float *gates_buf = workspace + gemm_buf_size / sizeof(float);

    RawBuffer fake_bias(N * sizeof(float));
    float *fake_bias_ptr = fake_bias.force_to<float *>();
    conv_sgemm_tn_col_major_prepack_a(M, N, K, w, K, x, K, gates_buf, M,
            fake_bias_ptr, ActivationType_None, gemm_buf, conv_gemm_conf_);
    
    for (int t = 0; t < seq_len; t++) {
        int ti = reverse ? seq_len - 1 - t : t;
        auto gates_t = gates_buf +  ti * batch_size * 4 * hidden_size;
        auto y_t = y + ti * batch_size * hidden_size;

        // add bias
        OMP_PARALLEL_FOR_GUIDED_
        for (int i = 0; i < batch_size; i++) {
            auto gates_b = gates_t + i * 4 * hidden_size;
            for (int j = 0; j < hidden_size; j++) {
                auto gates_j = gates_b + j * 4;
                auto bias_j = b + j * 4;
                Float4::saveu(gates_j, Float4::loadu(gates_j) + Float4::loadu(bias_j));
            }
        }

        // sgemm for recurrence weight
        // weights: [4*hidden_size, hidden_size]
        // inputs: [batch, hidden_size]
        K = hidden_size;
        N = batch_size;
        M = 4 * hidden_size;
        conv_sgemm_tn_col_major_prepack_a(M, N, K, r, K, h_t, K, gates_t, M,
                nullptr, ActivationType_None, gemm_buf, conv_gemm_conf_);

        // activation for h_t, c_t, output
        X86LSTMActivate(gates_t, h_t, c_t, y_t, batch_size * hidden_size);
    }
    return TNN_OK;
}

X86LSTMONNXLayerAcc::~X86LSTMONNXLayerAcc() {}

Status X86LSTMONNXLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                     const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = X86LayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);

    if (inputs.size() < 4) {
        return Status(TNNERR_LAYER_ERR, "LSTM has invalid inputs");
    }

    RETURN_ON_NEQ(allocateBufferWeight(inputs, outputs), TNN_OK);
    RETURN_ON_NEQ(allocateBufferBias(inputs, outputs), TNN_OK);

    return TNN_OK;
}

Status X86LSTMONNXLayerAcc::allocateBufferWeight(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    // weights for gates, [num_direction, 4 * hidden_size, input_size]
    auto w_dims = inputs[1]->GetBlobDesc().dims;
    int w_direction_size = DimsVectorUtils::Count(w_dims, 1);
    float *w_ptr = (float *)((char*)(inputs[1]->GetHandle().base) + inputs[1]->GetHandle().bytes_offset);

    // recurrence weights, [num_direction, 4 * hidden_size, hidden_size]
    auto r_dims = inputs[2]->GetBlobDesc().dims;
    int r_direction_size = DimsVectorUtils::Count(r_dims, 1);
    float *r_ptr = (float *)((char*)(inputs[2]->GetHandle().base) + inputs[2]->GetHandle().bytes_offset);

    int k_c = conv_gemm_conf_.K_c_;
    int m_block = conv_gemm_conf_.m_block_;
    // gate weights
    int K = w_dims[2];
    int M = w_dims[1];
    size_t w_pack_size = ROUND_UP(K, k_c) * ROUND_UP(M, m_block);
    // align pointer of packed weights, since gemm use aligned load for input A
    RawBuffer w_temp_buffer(w_dims[0] * w_pack_size * sizeof(float), 32);
    
    // before conv_pack, trans from 4 * hidden_size to hidden_size * 4
    size_t trans_size = MAX(w_direction_size, r_direction_size);
    RawBuffer trans_buf(trans_size * sizeof(float));
    int hidden_size = w_dims[1] / 4;
    float *trans_ptr = trans_buf.force_to<float *>();

    for (int d = 0; d < w_dims[0]; d++) {
        float *w_src = w_ptr + d * w_direction_size;
        float *w_dst = w_temp_buffer.force_to<float *>() + d * w_pack_size;

        // transpose
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < hidden_size; j++) {
                auto trans_dst = trans_ptr + j * 4 * w_dims[2] + i * w_dims[2];
                auto trans_src = w_src + i * hidden_size * w_dims[2] + j * w_dims[2];
                memcpy(trans_dst, trans_src, w_dims[2] * sizeof(float));
            }
        }

        conv_pack_col_a_t(M, K, trans_ptr, K, w_dst, conv_gemm_conf_);
    }

    // recurrence weights
    K = r_dims[2];
    M = r_dims[1];
    size_t r_pack_size = ROUND_UP(K, k_c) * ROUND_UP(M, m_block);
    RawBuffer r_temp_buffer(r_dims[0] * r_pack_size * sizeof(float), 32);
    for (int d = 0; d < r_dims[0]; d++) {
        float *r_src = r_ptr + d * r_direction_size;
        float *r_dst = r_temp_buffer.force_to<float *>() + d * r_pack_size;

        // transpose
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < hidden_size; j++) {
                auto trans_dst = trans_ptr + j * 4 * r_dims[2] + i * r_dims[2];
                auto trans_src = r_src + i * hidden_size * r_dims[2] + j * r_dims[2];
                memcpy(trans_dst, trans_src, r_dims[2] * sizeof(float));
            }
        }

        conv_pack_col_a_t(M, K, trans_ptr, K, r_dst, conv_gemm_conf_);
    }

    w_temp_buffer.SetDataType(DATA_TYPE_FLOAT);
    r_temp_buffer.SetDataType(DATA_TYPE_FLOAT);
    buffer_w_ = w_temp_buffer;
    buffer_r_ = r_temp_buffer;

    return TNN_OK;
}

Status X86LSTMONNXLayerAcc::allocateBufferBias(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    // bias for gate and recurrence, [num_directions, 8*hidden_size]
    auto b_dims = inputs[3]->GetBlobDesc().dims;
    int hidden_size = b_dims[1] / 8;
    int bias_size = hidden_size * 4;
    RawBuffer b_temp_buffer(b_dims[0] * bias_size * sizeof(float));

    float *b_ptr = (float *)((char*)(inputs[3]->GetHandle().base) + inputs[3]->GetHandle().bytes_offset);

    for (int d = 0; d < b_dims[0]; d++) {
        float *b_d = b_ptr + d * b_dims[1];
        float *wb_d = b_d;
        float *rb_d = b_d + 4 * hidden_size;
        float *b_dst = b_temp_buffer.force_to<float *>() + d * bias_size;

        // add bias and transpose to hidden_size * 4
        for (int i = 0; i < hidden_size; i++) {
            b_dst[i * 4 + 0] = wb_d[i + 0 * hidden_size] + rb_d[i + 0 * hidden_size];
            b_dst[i * 4 + 1] = wb_d[i + 1 * hidden_size] + rb_d[i + 1 * hidden_size];
            b_dst[i * 4 + 2] = wb_d[i + 2 * hidden_size] + rb_d[i + 2 * hidden_size];
            b_dst[i * 4 + 3] = wb_d[i + 3 * hidden_size] + rb_d[i + 3 * hidden_size];
        }
    }
    b_temp_buffer.SetDataType(DATA_TYPE_FLOAT);
    buffer_b_ = b_temp_buffer;

    return TNN_OK;
}

Status X86LSTMONNXLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<LSTMONNXLayerParam *>(param_);
    int num_directions = layer_param->direction >=2 ? 2 : 1;
    
    bool reverse = false;
    if (inputs.size() < 4) {
        return Status(TNNERR_LAYER_ERR, "LSTM has invalid inputs");
    }
    Blob * blob_h0 = nullptr;
    Blob * blob_c0 = nullptr;
    
    if (inputs.size() >= 6) {
        blob_h0 = inputs[4];
        blob_c0 = inputs[5];
    }
    
    const auto input_dims = inputs[0]->GetBlobDesc().dims;
    const auto T = input_dims[0]; // length of sequence
    const auto batch = input_dims[1];  // batch_size
    const auto input_size = DimsVectorUtils::Count(input_dims, 2); // input dimension
    const auto output_dims = outputs[0]->GetBlobDesc().dims;
    const auto hidden_size = layer_param->hidden_size; // output dimension
    // block size for gemm
    int k_c = conv_gemm_conf_.K_c_;
    int m_block = conv_gemm_conf_.m_block_;
    
    //X shape [sequence batch_size input_size]
    float *x = (float *)((char*)(inputs[0]->GetHandle().base) + inputs[0]->GetHandle().bytes_offset);
    
    //Y shape [sequence batch_size num_directions *hidden_size]
    float *y = (float *)((char*)(outputs[0]->GetHandle().base) + outputs[0]->GetHandle().bytes_offset);
    
    //W[iofc], weight tensor for the gates, shape [num_directions, 4*hidden_size, input_size]
    float *w = (float *)buffer_w_.force_to<float *>();
    auto w_dims = inputs[1]->GetBlobDesc().dims;
    size_t w_pack_size = ROUND_UP(w_dims[2], k_c) * ROUND_UP(w_dims[1], m_block);

    //R[iofc], recurrence weight tensor, shape [num_directions, 4*hidden_size, hidden_size]
    float *r = (float *)buffer_r_.force_to<float *>();
    auto r_dims = inputs[2]->GetBlobDesc().dims;
    size_t r_pack_size = ROUND_UP(r_dims[2], k_c) * ROUND_UP(r_dims[1], m_block);
    
    //B[iofc] Concatenation of [Wb[iofc], Rb[iofc]], [num_directions, 8*hidden_size]
    float *b = (float *)buffer_b_.force_to<float *>();
    
    //initial_h, initial value of the hidden, If not specified - assumed to be 0. shape [num_directions, batch_size, hidden_size]
    auto h_t = (float *)((char*)(outputs[1]->GetHandle().base) + outputs[1]->GetHandle().bytes_offset);
    //initial_c, initial value of the cell, If not specified - assumed to be 0. shape [num_directions, batch_size, hidden_size]
    auto c_t = (float *)((char*)(outputs[2]->GetHandle().base) + outputs[2]->GetHandle().bytes_offset);

    if (inputs.size() >= 6) {
        auto h_0 = (float *)((char*)(blob_h0->GetHandle().base) + blob_h0->GetHandle().bytes_offset);
        auto c_0 = (float *)((char*)(blob_c0->GetHandle().base) + blob_c0->GetHandle().bytes_offset);
        memcpy((void *)h_t, h_0, num_directions * batch * hidden_size * sizeof(float));
        memcpy((void *)c_t, c_0, num_directions * batch * hidden_size * sizeof(float));
    } else {
        memset((void *)h_t, 0, num_directions * batch * hidden_size * sizeof(float));
        memset((void *)c_t, 0, num_directions * batch * hidden_size * sizeof(float));
    }
    
    if (layer_param->direction == 0 || layer_param->direction == 1) {
        return LSTMOneDirection(x, y, w, r, b, h_t, c_t, T, batch, input_size, hidden_size, layer_param->direction);
    } else if (layer_param->direction == 2) {
        //Y shape [num_directions sequence batch_size hidden_size]
        auto y_temp = std::shared_ptr<float>(new float[num_directions*T*batch*hidden_size], [](float* p) { delete[] p; });
        auto y0 = y_temp.get();
        auto y1 = y0 + T * batch * hidden_size;
        LSTMOneDirection(x, y0, w, r, b, h_t, c_t, T, batch, input_size, hidden_size, 0);
        
        auto w1 = w + w_pack_size;
        auto r1 = r + r_pack_size;
        auto b1 = b + 4 * hidden_size;
        auto h_t1 = h_t + batch * hidden_size;
        auto c_t1 = c_t + batch * hidden_size;
        LSTMOneDirection(x, y1, w1, r1, b1, h_t1, c_t1, T, batch, input_size, hidden_size, 1);
        
        //transpose [num_directions sequence batch_size hidden_size] to [sequence batch_size num_directions*hidden_size]
        for (int i = 0; i < T*batch; i++) {
            auto y0_data = y0 + i * hidden_size;
            auto y1_data = y1 + i * hidden_size;
            auto y_data = y + i * num_directions * hidden_size;

            memcpy(y_data, y0_data, hidden_size * sizeof(float));
            memcpy(y_data + hidden_size, y1_data, hidden_size * sizeof(float));
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "LSTMONNX has invalid direction param");
    }

    return TNN_OK;
}

REGISTER_X86_ACC(LSTMONNX, LAYER_LSTMONNX);
}  // namespace TNN_NS
