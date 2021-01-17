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

#include "cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(LSTMONNX, LAYER_LSTMONNX);

Status CpuLSTMONNXLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuLSTMONNXLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<LSTMONNXLayerParam *>(param_);
    
    bool reverse = false;
    if (inputs.size() < 4) {
        return Status(TNNERR_LAYER_ERR, "LSTM has invalid inputs");
    }
    Blob * blob_W = inputs[1];
    Blob * blob_R = inputs[2];
    Blob * blob_B = inputs[3];
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
    const auto output_size = DimsVectorUtils::Count(output_dims, 2); // output dimension
    
    //X shape [sequence batch_size input_size]
    int x_page_size = DimsVectorUtils::Count(input_dims, 1);
    float *x_data = (float *)((char*)(inputs[0]->GetHandle().base) + inputs[0]->GetHandle().bytes_offset);
    
    //Y shape [sequence batch_size output_size]
    int y_page_size = DimsVectorUtils::Count(output_dims, 1);
    float *y_data = (float *)((char*)(outputs[0]->GetHandle().base) + outputs[0]->GetHandle().bytes_offset);
    
    //W[iofc], weight tensor for the gates, shape [num_directions, 4*hidden_size, input_size]
    int w_page_size = DimsVectorUtils::Count(blob_W->GetBlobDesc().dims) / 4;
    float *w_x_I = (float *)((char*)(blob_W->GetHandle().base) + blob_W->GetHandle().bytes_offset);
    float *w_x_O = w_x_I + w_page_size;
    float *w_x_F = w_x_O + w_page_size;
    float *w_x_C = w_x_F + w_page_size;
    
    //R[iofc], recurrence weight tensor, shape [num_directions, 4*hidden_size, hidden_size]
    int r_page_size = DimsVectorUtils::Count(blob_R->GetBlobDesc().dims) / 4;
    float *r_x_I = (float *)((char*)(blob_R->GetHandle().base) + blob_R->GetHandle().bytes_offset);
    float *r_x_O = r_x_I + r_page_size;
    float *r_x_F = r_x_O + r_page_size;
    float *r_x_C = r_x_F + r_page_size;
    
    //B[iofc] Concatenation of [Wb[iofc], Rb[iofc]], [num_directions, 8*hidden_size]
    int b_page_size = DimsVectorUtils::Count(blob_B->GetBlobDesc().dims) / 2 / 4;
    float *b_w_I = (float *)((char*)(blob_B->GetHandle().base) + blob_B->GetHandle().bytes_offset);
    float *b_w_O = b_w_I + b_page_size;
    float *b_w_F = b_w_O + b_page_size;
    float *b_w_C = b_w_F + b_page_size;
    
    float *b_r_I = b_w_C + b_page_size;
    float *b_r_O = b_r_I + b_page_size;
    float *b_r_F = b_r_O + b_page_size;
    float *b_r_C = b_r_F + b_page_size;
    
    //initial_h, initial value of the hidden, If not specified - assumed to be 0. shape [num_directions, batch_size, hidden_size]
    float *h_0 = (float *)((char*)(blob_h0->GetHandle().base) + blob_h0->GetHandle().bytes_offset);
    auto h_t = std::shared_ptr<float>((float*)calloc(batch * output_size, sizeof(float)), [](float* p) { free(p); });
    if (h_0) {
        memcpy((void *)h_t.get(), h_0, batch * output_size * sizeof(float));
    }
    
    //initial_c, initial value of the cell, If not specified - assumed to be 0. shape [num_directions, batch_size, hidden_size]
    auto c_t = std::shared_ptr<float>((float*)calloc(batch * output_size, sizeof(float)), [](float* p) { free(p); });
    float *c_0 = (float *)((char*)(blob_c0->GetHandle().base) + blob_c0->GetHandle().bytes_offset);
    if (c_0) {
        memcpy((void *)c_t.get(), c_0, batch * output_size * sizeof(float));
    }
    
    //temp gates, shape [hidden_size, 4]
    auto gates = std::shared_ptr<float>(new float[output_size * 4], [](float* p) { delete[] p; });
    
    for (int t = 0; t < T; t++) {
        int ti = reverse ? T - 1 - t : t;

        const float* x_t = x_data + ti * x_page_size;
        float* y_t = y_data + ti *y_page_size;
        
        for (int b = 0; b < batch; b++) {
            const float* x_t_b = x_t + b * input_size;
            float* h_t_b = h_t.get() + b * output_size;
            float* c_t_b = c_t.get() + b * output_size;
            //float*gates_b = (float *)gates.get() + b * output_size * 4;
            
            for (int q = 0; q < output_size; q++) {
                auto gates_data = (float *)gates.get() + q * 4;
                
                //W weights
                auto w_x_I_o = w_x_I + q * input_size;
                auto w_x_O_o = w_x_O + q * input_size;
                auto w_x_F_o = w_x_F + q * input_size;
                auto w_x_C_o = w_x_C + q * input_size;

                auto r_x_I_o = r_x_I + q * output_size;
                auto r_x_O_o = r_x_O + q * output_size;
                auto r_x_F_o = r_x_F + q * output_size;
                auto r_x_C_o = r_x_C + q * output_size;
                
                //bias
                float I = b_w_I[q] + b_r_I[q];
                float O = b_w_O[q] + b_r_O[q];
                float F = b_w_F[q] + b_r_F[q];
                float C = b_w_C[q] + b_r_C[q];

                for (int i = 0; i < input_size; i++) {
                    I += w_x_I_o[i] * x_t_b[i];
                    O += w_x_O_o[i] * x_t_b[i];
                    F += w_x_F_o[i] * x_t_b[i];
                    C += w_x_C_o[i] * x_t_b[i];
                }

                for (int i = 0; i < output_size; i++) {
                    I += r_x_I_o[i] * h_t_b[i];
                    O += r_x_O_o[i] * h_t_b[i];
                    F += r_x_F_o[i] * h_t_b[i];
                    C += r_x_C_o[i] * h_t_b[i];
                }

                gates_data[0] = I;
                gates_data[1] = O;
                gates_data[2] = F;
                gates_data[3] = C;
            }
            
            float* output_data = y_t + b *output_size;
            for (int q = 0; q < output_size; q++) {
                const auto gates_data = (float *)gates.get() + q * 4;

                float I = gates_data[0];
                float O = gates_data[1];
                float F = gates_data[2];
                float C = gates_data[3];

                I = 1.f / (1.f + exp(-I));
                F = 1.f / (1.f + exp(-F));
                O = 1.f / (1.f + exp(-O));
                C = tanh(C);

                float cell2 = F * c_t_b[q] + I * C;
                float H = O * tanh(cell2);
                c_t_b[q] = cell2;
                h_t_b[q] = H;
                output_data[q] = H;
            }
        }
        

    }
  
    return TNN_OK;
}

REGISTER_CPU_ACC(LSTMONNX, LAYER_LSTMONNX);
}  // namespace TNN_NS
