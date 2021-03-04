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

#include "arm_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(LSTMONNX, LAYER_LSTMONNX);

static Status LstmSingle(const float *x, float *y, const float *w, const float *r, const float *b, float *h_t,
                         float *c_t, const int seq_len, const int batch_size, const int input_size,
                         const int hidden_size, int reverse) {
    // num_directions = 1 for all below
    // X shape [sequence batch_size input_size]
    const int x_page_size = batch_size * input_size;

    // Y shape [sequence batch_size num_directions * hidden_size]
    const int y_page_size = batch_size * hidden_size;

    // W[iofc], weight tensor for the gates, shape [num_directions, 4*hidden_size, input_size]
    const int w_page_size = hidden_size * input_size;
    auto w_x_I            = w;
    auto w_x_O            = w_x_I + w_page_size;
    auto w_x_F            = w_x_O + w_page_size;
    auto w_x_C            = w_x_F + w_page_size;

    // R[iofc], recurrence weight tensor, shape [num_directions, 4*hidden_size, hidden_size]
    int r_page_size = hidden_size * hidden_size;
    auto r_x_I      = r;
    auto r_x_O      = r_x_I + r_page_size;
    auto r_x_F      = r_x_O + r_page_size;
    auto r_x_C      = r_x_F + r_page_size;

    // B[iofc] Concatenation of [Wb[iofc], Rb[iofc]], [num_directions, 8*hidden_size]
    int b_page_size = hidden_size;
    auto b_w_I      = b;
    auto b_w_O      = b_w_I + b_page_size;
    auto b_w_F      = b_w_O + b_page_size;
    auto b_w_C      = b_w_F + b_page_size;

    auto b_r_I = b_w_C + b_page_size;
    auto b_r_O = b_r_I + b_page_size;
    auto b_r_F = b_r_O + b_page_size;
    auto b_r_C = b_r_F + b_page_size;

    // temp gates, shape [hidden_size, 4]
    RawBuffer gates = RawBuffer(seq_len * batch_size * hidden_size * 4 * sizeof(float));
    auto gates_ptr  = gates.force_to<float *>();

    for (int t = 0; t < seq_len * batch_size; t++) {
        const float *x_t = x + t * input_size;
        float *gates_t   = gates_ptr + t * hidden_size * 4;
        for (int q = 0; q < hidden_size; q++) {
            auto gates_data = gates_t + q * 4;

            // W weights
            auto w_x_I_o = w_x_I + q * input_size;
            auto w_x_O_o = w_x_O + q * input_size;
            auto w_x_F_o = w_x_F + q * input_size;
            auto w_x_C_o = w_x_C + q * input_size;

            // bias
            float I = b_w_I[q] + b_r_I[q];
            float O = b_w_O[q] + b_r_O[q];
            float F = b_w_F[q] + b_r_F[q];
            float C = b_w_C[q] + b_r_C[q];

            for (int i = 0; i < input_size; i++) {
                I += w_x_I_o[i] * x_t[i];
                O += w_x_O_o[i] * x_t[i];
                F += w_x_F_o[i] * x_t[i];
                C += w_x_C_o[i] * x_t[i];
            }

            gates_data[0] = I;
            gates_data[1] = O;
            gates_data[2] = F;
            gates_data[3] = C;
        }
    }

    for (int t = 0; t < seq_len; t++) {
        int ti = reverse ? seq_len - 1 - t : t;

        float *y_t     = y + ti * y_page_size;
        float *gates_t = gates_ptr + ti * batch_size * hidden_size * 4;

        for (int b = 0; b < batch_size; b++) {
            float *h_t_b   = h_t + b * hidden_size;
            float *gates_b = gates_t + b * hidden_size * 4;

            for (int q = 0; q < hidden_size; q++) {
                auto gates_data = gates_b + q * 4;

                // R weights
                auto r_x_I_o = r_x_I + q * hidden_size;
                auto r_x_O_o = r_x_O + q * hidden_size;
                auto r_x_F_o = r_x_F + q * hidden_size;
                auto r_x_C_o = r_x_C + q * hidden_size;

                // bias
                float I = gates_data[0];
                float O = gates_data[1];
                float F = gates_data[2];
                float C = gates_data[3];

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
        }

        for (int b = 0; b < batch_size; b++) {
            float *h_t_b       = h_t + b * hidden_size;
            float *c_t_b       = c_t + b * hidden_size;
            float *output_data = y_t + b * hidden_size;
            float *gates_b     = gates_t + b * hidden_size * 4;
            for (int q = 0; q < hidden_size; q++) {
                const auto gates_data = gates_b + q * 4;

                float I = gates_data[0];
                float O = gates_data[1];
                float F = gates_data[2];
                float C = gates_data[3];

                I = 1.f / (1.f + exp(-I));
                F = 1.f / (1.f + exp(-F));
                O = 1.f / (1.f + exp(-O));
                C = tanh(C);

                float cell2    = F * c_t_b[q] + I * C;
                float H        = O * tanh(cell2);
                c_t_b[q]       = cell2;
                h_t_b[q]       = H;
                output_data[q] = H;
            }
        }
    }

    return TNN_OK;
}

Status ArmLSTMONNXLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<LSTMONNXLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    int num_directions = layer_param->direction >= 2 ? 2 : 1;

    if (inputs.size() < 4) {
        return Status(TNNERR_LAYER_ERR, "LSTM has invalid inputs");
    }
    Blob *blob_input = inputs[0];
    Blob *blob_W     = inputs[1];
    Blob *blob_R     = inputs[2];
    Blob *blob_B     = inputs[3];
    Blob *blob_h0    = (inputs.size() >= 6) ? inputs[4] : nullptr;
    Blob *blob_c0    = (inputs.size() >= 6) ? inputs[5] : nullptr;

    if (outputs.size() < 3) {
        return Status(TNNERR_LAYER_ERR, "LSTM has invalid outputs");
    }
    Blob *blob_output = outputs[0];
    Blob *blob_ht     = outputs[1];
    Blob *blob_ct     = outputs[2];

    const auto input_dims = blob_input->GetBlobDesc().dims;
    const auto seq_len    = input_dims[0];
    const auto batch      = input_dims[1];
    const auto input_size = DimsVectorUtils::Count(input_dims, 2);

    const auto output_dims = blob_output->GetBlobDesc().dims;
    const auto hidden_size = layer_param->hidden_size;

    // X shape [sequence batch_size input_size]
    float *x = reinterpret_cast<float *>(GetBlobHandlePtr(blob_input->GetHandle()));

    // Y shape [sequence batch_size num_directions *hidden_size]
    float *y = reinterpret_cast<float *>(GetBlobHandlePtr(blob_output->GetHandle()));

    // W[iofc], weight tensor for the gates, shape [num_directions, 4*hidden_size, input_size]
    float *w = reinterpret_cast<float *>(GetBlobHandlePtr(blob_W->GetHandle()));

    // R[iofc], recurrence weight tensor, shape [num_directions, 4*hidden_size, hidden_size]
    float *r = reinterpret_cast<float *>(GetBlobHandlePtr(blob_R->GetHandle()));

    // B[iofc] Concatenation of [Wb[iofc], Rb[iofc]], [num_directions, 8*hidden_size]
    float *b = reinterpret_cast<float *>(GetBlobHandlePtr(blob_B->GetHandle()));

    // Initial value of the hidden. If not specified, assumed to be 0.
    // shape [num_directions, batch_size, hidden_size]
    auto h_t = reinterpret_cast<float *>(GetBlobHandlePtr(blob_ht->GetHandle()));
    if (blob_h0) {
        auto h_0 = reinterpret_cast<float *>(GetBlobHandlePtr(blob_h0->GetHandle()));
        memcpy((void *)h_t, h_0, num_directions * batch * hidden_size * sizeof(float));
    }

    // Initial value of the cell. If not specified, assumed to be 0.
    // shape [num_directions, batch_size, hidden_size]
    auto c_t = reinterpret_cast<float *>(GetBlobHandlePtr(blob_ct->GetHandle()));
    if (blob_c0) {
        auto c_0 = reinterpret_cast<float *>(GetBlobHandlePtr(blob_c0->GetHandle()));
        memcpy((void *)c_t, c_0, num_directions * batch * hidden_size * sizeof(float));
    } else {
        memset((void *)c_t, 0, num_directions * batch * hidden_size * sizeof(float));
    }

    if (layer_param->direction == 0 || layer_param->direction == 1) {
        return LstmSingle(x, y, w, r, b, h_t, c_t, seq_len, batch, input_size, hidden_size, layer_param->direction);
    } else if (layer_param->direction == 2) {
        // Y shape [num_directions sequence batch_size hidden_size]
        RawBuffer y_temp = RawBuffer(num_directions * seq_len * batch * hidden_size * sizeof(float));
        auto y0          = y_temp.force_to<float *>();
        auto y1          = y0 + seq_len * batch * hidden_size;
        LstmSingle(x, y0, w, r, b, h_t, c_t, seq_len, batch, input_size, hidden_size, 0);

        auto w1   = w + 4 * hidden_size * input_size;
        auto r1   = r + 4 * hidden_size * hidden_size;
        auto b1   = b + 8 * hidden_size;
        auto h_t1 = h_t + batch * hidden_size;
        auto c_t1 = c_t + batch * hidden_size;
        LstmSingle(x, y1, w1, r1, b1, h_t1, c_t1, seq_len, batch, input_size, hidden_size, 1);

        // transpose [num_directions sequence batch_size hidden_size] to [sequence batch_size
        // num_directions*hidden_size]
        for (int i = 0; i < seq_len * batch; i++) {
            auto y0_data = y0 + i * hidden_size;
            auto y1_data = y1 + i * hidden_size;
            auto y_data  = y + i * num_directions * hidden_size;

            memcpy(y_data, y0_data, hidden_size * sizeof(float));
            memcpy(y_data + hidden_size, y1_data, hidden_size * sizeof(float));
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "LSTMONNX has invalid direction param");
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(LSTMONNX, LAYER_LSTMONNX);
REGISTER_ARM_LAYOUT(LAYER_LSTMONNX, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
