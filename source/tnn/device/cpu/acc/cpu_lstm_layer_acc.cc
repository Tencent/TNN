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
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

class CpuLSTMONNXLayerAcc : public CpuLayerAcc {
public:
    virtual ~CpuLSTMONNXLayerAcc(){};
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs);
    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
    virtual Status Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

private:
    std::shared_ptr<float> w_ = nullptr;
    std::shared_ptr<float> r_ = nullptr;
    std::shared_ptr<float> b_ = nullptr;
};

static Status LSTM_Single(const float *x, float *y, const float *w, const float *r, const float *b,
                          float *h_t, float *c_t,
                          const int T, const int batch_size, const int input_size, const int hidden_size, int reverse) {
    //num_directions = 1 for all below
    //X shape [sequence batch_size input_size]
    const int x_page_size = batch_size * input_size;
    
    //Y shape [sequence batch_size num_directions * hidden_size]
    const int y_page_size = batch_size * hidden_size;
    
    //W[iofc], weight tensor for the gates, shape [num_directions, 4*hidden_size, input_size]
    const int w_page_size = hidden_size * input_size;
    auto w_x_I = w;
    auto w_x_O = w_x_I + w_page_size;
    auto w_x_F = w_x_O + w_page_size;
    auto w_x_C = w_x_F + w_page_size;
    
    //R[iofc], recurrence weight tensor, shape [num_directions, 4*hidden_size, hidden_size]
    int r_page_size = hidden_size * hidden_size;
    auto r_x_I = r;
    auto r_x_O = r_x_I + r_page_size;
    auto r_x_F = r_x_O + r_page_size;
    auto r_x_C = r_x_F + r_page_size;
    
    //B[iofc] Concatenation of [Wb[iofc], Rb[iofc]], [num_directions, 8*hidden_size]
    int b_page_size = hidden_size;
    auto b_w_I = b;
    auto b_w_O = b_w_I + b_page_size;
    auto b_w_F = b_w_O + b_page_size;
    auto b_w_C = b_w_F + b_page_size;
    
    auto b_r_I = b_w_C + b_page_size;
    auto b_r_O = b_r_I + b_page_size;
    auto b_r_F = b_r_O + b_page_size;
    auto b_r_C = b_r_F + b_page_size;
    
    //temp gates, shape [hidden_size, 4]
    auto gates = std::shared_ptr<float>(new float[hidden_size * 4], [](float* p) { delete[] p; });
    
    for (int t = 0; t < T; t++) {
        int ti = reverse ? T - 1 - t : t;

        const float* x_t = x + ti * x_page_size;
        float* y_t = y + ti *y_page_size;
        
        for (int b = 0; b < batch_size; b++) {
            const float* x_t_b = x_t + b * input_size;
            float* h_t_b = h_t + b * hidden_size;
            float* c_t_b = c_t + b * hidden_size;
            //float*gates_b = (float *)gates.get() + b * output_size * 4;
            
            for (int q = 0; q < hidden_size; q++) {
                auto gates_data = (float *)gates.get() + q * 4;
                
                //W weights
                auto w_x_I_o = w_x_I + q * input_size;
                auto w_x_O_o = w_x_O + q * input_size;
                auto w_x_F_o = w_x_F + q * input_size;
                auto w_x_C_o = w_x_C + q * input_size;

                auto r_x_I_o = r_x_I + q * hidden_size;
                auto r_x_O_o = r_x_O + q * hidden_size;
                auto r_x_F_o = r_x_F + q * hidden_size;
                auto r_x_C_o = r_x_C + q * hidden_size;
                
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

                for (int i = 0; i < hidden_size; i++) {
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
            
            float* output_data = y_t + b *hidden_size;
            for (int q = 0; q < hidden_size; q++) {
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

Status CpuLSTMONNXLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto status = CpuLayerAcc::Init(context, param, resource, inputs, outputs);

    if (runtime_model_ == RUNTIME_MODE_CONST_FOLD) {
        return TNN_OK;
    }

    auto get_blob_data = [&](Blob *blob, float *result) -> Status {
        const int data_size = DimsVectorUtils::Count(blob->GetBlobDesc().dims);
        if (blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            float *src_ptr = (float *)((char *)(blob->GetHandle().base) + blob->GetHandle().bytes_offset);
            memcpy(result, src_ptr, data_size * sizeof(float));
        } else if (blob->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            fp16_t *src_ptr = (fp16_t *)((char *)(blob->GetHandle().base) + blob->GetHandle().bytes_offset);
            ConvertFromHalfToFloat(src_ptr, result, data_size);
        } else {
            return Status(TNNERR_LAYER_ERR, "data type not support in LSTM");
        }

        return TNN_OK;
    };

    Blob *blob_W = inputs[1];
    if (blob_W->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        const int blob_W_dims_count = DimsVectorUtils::Count(blob_W->GetBlobDesc().dims);
        std::shared_ptr<float> w(new float[blob_W_dims_count], [](float *p) { delete[] p; });
        RETURN_ON_NEQ(get_blob_data(blob_W, w.get()), TNN_OK);
        w_ = w;
    }

    Blob *blob_R = inputs[2];
    if (blob_R->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        const int blob_R_dims_count = DimsVectorUtils::Count(blob_R->GetBlobDesc().dims);
        std::shared_ptr<float> r(new float[blob_R_dims_count], [](float *p) { delete[] p; });
        RETURN_ON_NEQ(get_blob_data(blob_R, r.get()), TNN_OK);
        r_ = r;
    }

    Blob *blob_B = inputs[3];
    if (blob_B->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        const int blob_B_dims_count = DimsVectorUtils::Count(blob_B->GetBlobDesc().dims);
        std::shared_ptr<float> b(new float[blob_B_dims_count], [](float *p) { delete[] p; });
        RETURN_ON_NEQ(get_blob_data(blob_B, b.get()), TNN_OK);
        b_ = b;
    }

    return TNN_OK;
}

Status CpuLSTMONNXLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuLSTMONNXLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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
    
    //X shape [sequence batch_size input_size]
    float *x = (float *)((char*)(inputs[0]->GetHandle().base) + inputs[0]->GetHandle().bytes_offset);
    
    //Y shape [sequence batch_size num_directions *hidden_size]
    float *y = (float *)((char*)(outputs[0]->GetHandle().base) + outputs[0]->GetHandle().bytes_offset);
    
    //W[iofc], weight tensor for the gates, shape [num_directions, 4*hidden_size, input_size]
    float *w = inputs[1]->GetBlobDesc().data_type != DATA_TYPE_HALF
                   ? (float *)((char *)(inputs[1]->GetHandle().base) + inputs[1]->GetHandle().bytes_offset)
                   : w_.get();

    //R[iofc], recurrence weight tensor, shape [num_directions, 4*hidden_size, hidden_size]
    float *r = inputs[2]->GetBlobDesc().data_type != DATA_TYPE_HALF
                   ? (float *)((char *)(inputs[2]->GetHandle().base) + inputs[2]->GetHandle().bytes_offset)
                   : r_.get();

    //B[iofc] Concatenation of [Wb[iofc], Rb[iofc]], [num_directions, 8*hidden_size]
    float *b = inputs[3]->GetBlobDesc().data_type != DATA_TYPE_HALF
                   ? (float *)((char *)(inputs[3]->GetHandle().base) + inputs[3]->GetHandle().bytes_offset)
                   : b_.get();

    //initial_h, initial value of the hidden, If not specified - assumed to be 0. shape [num_directions, batch_size, hidden_size]
    auto h_t = (float *)((char*)(outputs[1]->GetHandle().base) + outputs[1]->GetHandle().bytes_offset);
    if (blob_h0 != nullptr){
        auto h_0 = (float *)((char*)(blob_h0->GetHandle().base) + blob_h0->GetHandle().bytes_offset);
        if (h_0) {
            memcpy((void *)h_t, h_0, num_directions * batch * hidden_size * sizeof(float));
        }
    } else {
        memset(h_t, 0, num_directions * batch * hidden_size * sizeof(float));
    }
    
    //initial_c, initial value of the cell, If not specified - assumed to be 0. shape [num_directions, batch_size, hidden_size]
    auto c_t = (float *)((char*)(outputs[2]->GetHandle().base) + outputs[2]->GetHandle().bytes_offset);
    if (blob_c0 != nullptr){
        auto c_0 = (float *)((char*)(blob_c0->GetHandle().base) + blob_c0->GetHandle().bytes_offset);
        if (c_0) {
            memcpy((void *)c_t, c_0, num_directions * batch * hidden_size * sizeof(float));
        }
    } else {
        memset(c_t, 0, num_directions * batch * hidden_size * sizeof(float));
    }
    
    if (layer_param->direction == 0 || layer_param->direction == 1) {
        return LSTM_Single(x, y, w, r, b, h_t, c_t, T, batch, input_size, hidden_size, layer_param->direction);
    } else if (layer_param->direction == 2) {
        //Y shape [num_directions sequence batch_size hidden_size]
        auto y_temp = std::shared_ptr<float>(new float[num_directions*T*batch*hidden_size], [](float* p) { delete[] p; });
        auto y0 = y_temp.get();
        auto y1 = y0 + T * batch * hidden_size;
        LSTM_Single(x, y0, w, r, b, h_t, c_t, T, batch, input_size, hidden_size, 0);
        
        auto w1 = w + 4*hidden_size*input_size;
        auto r1 = r + 4*hidden_size*hidden_size;
        auto b1 = b + 8*hidden_size;
        auto h_t1 = h_t + batch*hidden_size;
        auto c_t1 = c_t + batch*hidden_size;
        LSTM_Single(x, y1, w1, r1, b1, h_t1, c_t1, T, batch, input_size, hidden_size, 1);
        
        //transpose [num_directions sequence batch_size hidden_size] to [sequence batch_size num_directions*hidden_size]
        for (int i = 0; i < T*batch; i++) {
            auto y0_data = y0 + i*hidden_size;
            auto y1_data = y1 + i*hidden_size;
            auto y_data = y + i*num_directions*hidden_size;

            memcpy(y_data, y0_data, hidden_size * sizeof(float));
            memcpy(y_data + hidden_size, y1_data, hidden_size * sizeof(float));
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "LSTMONNX has invalid direction param");
    }

  
    return TNN_OK;
}

REGISTER_CPU_ACC(LSTMONNX, LAYER_LSTMONNX);
}  // namespace TNN_NS
