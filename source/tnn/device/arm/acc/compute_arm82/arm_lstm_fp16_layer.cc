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

#include "tnn/device/arm/acc/Half8.h"
#include "tnn/device/arm/acc/arm_lstm_layer_acc.h"
#include "tnn/device/arm/acc/compute/compute.h"
#include "tnn/device/arm/acc/compute/gemm_function.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

#if TNN_ARM82

static void LstmActivate(const int count, const fp16_t *g_ptr, fp16_t *c_ptr, fp16_t *h_ptr, fp16_t *o_ptr) {
#ifdef TNN_ARM82_USE_NEON
    OMP_PARALLEL_FOR_
    for (int q = 0; q < count - 7; q += 8) {
        Half8x4 gates_iofc = Half8x4::ld4(g_ptr + q * 4);
        Half8 I, O, F, C;
        gates_iofc.get_lane(I, 0);
        gates_iofc.get_lane(O, 1);
        gates_iofc.get_lane(F, 2);
        gates_iofc.get_lane(C, 3);

        I = Half8::sigmoid(I);
        F = Half8::sigmoid(F);
        O = Half8::sigmoid(O);
        C = Half8::tanh(C);

        Half8 cell2 = F * Half8::load(c_ptr + q) + I * C;
        Half8 H     = O * Half8::tanh(cell2);
        Half8::save(c_ptr + q, cell2);
        Half8::save(h_ptr + q, H);
        Half8::save(o_ptr + q, H);
    }
    int remain = count % 8;
    int offset = count / 8 * 8;
    g_ptr += offset * 4;
    c_ptr += offset;
    h_ptr += offset;
    o_ptr += offset;
    if (remain) {
        Half8x4 gates_iofc = Half8x4::ld4(g_ptr);
        Half8 I, O, F, C;
        gates_iofc.get_lane(I, 0);
        gates_iofc.get_lane(O, 1);
        gates_iofc.get_lane(F, 2);
        gates_iofc.get_lane(C, 3);

        I = Half8::sigmoid(I);
        F = Half8::sigmoid(F);
        O = Half8::sigmoid(O);
        C = Half8::tanh(C);

        Half8 c_old;
        for (int r = 0; r < remain; ++r) {
            c_old.set_lane(c_ptr[r], r);
        }
        Half8 cell2 = F * c_old + I * C;
        Half8 H     = O * Half8::tanh(cell2);
        for (int r = 0; r < remain; ++r) {
            c_ptr[r] = cell2[r];
            h_ptr[r] = H[r];
            o_ptr[r] = H[r];
        }
    }
#else
    for (int q = 0; q < count; ++q) {
        const auto gates_data = g_ptr + q * 4;

        float I = (float)gates_data[0];
        float O = (float)gates_data[1];
        float F = (float)gates_data[2];
        float C = (float)gates_data[3];

        I = 1.f / (1.f + exp(-I));
        F = 1.f / (1.f + exp(-F));
        O = 1.f / (1.f + exp(-O));
        C = tanh(C);

        float cell2 = F * (float)c_ptr[q] + I * C;
        float H     = O * tanh(cell2);
        c_ptr[q]    = (fp16_t)cell2;
        h_ptr[q]    = (fp16_t)H;
        o_ptr[q]    = (fp16_t)H;
    }
#endif  // TNN_ARM82_USE_NEON
}

Status ArmLSTMONNXLayerAcc::LstmSingleDirection(const fp16_t *x, fp16_t *y, const fp16_t *w, const fp16_t *r,
                                                const fp16_t *b, fp16_t *h_t, fp16_t *c_t, const int batch_size,
                                                int reverse) {
    const int input_size  = input_size_;
    const int hidden_size = hidden_size_;
    const int seq_len     = seq_len_;

    int gates_count      = seq_len * batch_size * hidden_size * 4;
    int input_pack_count = MAX(seq_len * batch_size * input_size, batch_size * hidden_size);
    auto workspace =
        context_->GetSharedWorkSpace((gates_count + input_pack_count) * sizeof(fp16_t) + NEON_KERNEL_EXTRA_LOAD);
    auto gates_ptr      = reinterpret_cast<fp16_t *>(workspace);
    auto input_pack_ptr = gates_ptr + gates_count;
    for (int i = 0; i < seq_len * batch_size; ++i) {
        fp16_t *gates_i = gates_ptr + i * hidden_size * 4;
        memcpy(gates_i, b, hidden_size * 4 * sizeof(fp16_t));
    }

    GemmHalfPackA(seq_len * batch_size, hidden_size * 4, input_size, x, input_pack_ptr, input_size, w, hidden_size * 4,
                  gates_ptr, hidden_size * 4);

    for (int t = 0; t < seq_len; ++t) {
        int ti          = reverse ? seq_len - 1 - t : t;
        fp16_t *y_t     = y + ti * batch_size * hidden_size;
        fp16_t *gates_t = gates_ptr + ti * batch_size * hidden_size * 4;

        GemmHalfPackA(batch_size, hidden_size * 4, hidden_size, h_t, input_pack_ptr, hidden_size, r, hidden_size * 4,
                      gates_t, hidden_size * 4);

        LstmActivate(batch_size * hidden_size, gates_t, c_t, h_t, y_t);
    }

    return TNN_OK;
}

// [4, hidden_size, input_size] -> transpose -> [input_size, hidden_size, 4]
// [input_size, hidden_size * 4] -> PackB_16 -> [hidden_size * 4 / 16, input_size, 16]
static void TransposeAndPackWeight(const fp16_t *src, fp16_t *dst, int input_size, int hidden_size) {
    RawBuffer tmp_transpose = RawBuffer(input_size * hidden_size * 4 * sizeof(fp16_t));
    fp16_t *src_transpose   = tmp_transpose.force_to<fp16_t *>();
    const fp16_t *vsrc[4];
    vsrc[0]   = src;
    vsrc[1]   = vsrc[0] + input_size * hidden_size;
    vsrc[2]   = vsrc[1] + input_size * hidden_size;
    vsrc[3]   = vsrc[2] + input_size * hidden_size;
    int count = 0;
    for (int i = 0; i < input_size; ++i) {
        for (int h = 0; h < hidden_size; ++h) {
            src_transpose[count++] = (fp16_t)vsrc[0][h * input_size + i];
            src_transpose[count++] = (fp16_t)vsrc[1][h * input_size + i];
            src_transpose[count++] = (fp16_t)vsrc[2][h * input_size + i];
            src_transpose[count++] = (fp16_t)vsrc[3][h * input_size + i];
        }
    }
    PackB_16(input_size, hidden_size * 4, src_transpose, hidden_size * 4, dst);
}

Status ArmLSTMONNXLayerAcc::AllocateBufferWeightInputHalf(Blob *weight_i) {
    // W[iofc], weight tensor for the gates, shape [num_directions, 4*hidden_size, input_size]
    fp16_t *weight_i_ptr = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(weight_i->GetHandle()));

    int weight_page       = input_size_ * ROUND_UP(4 * hidden_size_, 16);
    int weight_byte_count = num_directions_ * weight_page * sizeof(fp16_t);
    buffer_weight_input_  = RawBuffer(weight_byte_count + NEON_KERNEL_EXTRA_LOAD);
    for (int dir = 0; dir < num_directions_; ++dir) {
        fp16_t *buffer_ptr = buffer_weight_input_.force_to<fp16_t *>() + dir * weight_page;
        TransposeAndPackWeight(weight_i_ptr, buffer_ptr, input_size_, hidden_size_);
        weight_i_ptr += 4 * hidden_size_ * input_size_;
    }

    return TNN_OK;
}

Status ArmLSTMONNXLayerAcc::AllocateBufferWeightRecurrentHalf(Blob *weight_r) {
    // R[iofc], recurrence weight tensor, shape [num_directions, 4*hidden_size, hidden_size]
    fp16_t *weight_r_ptr = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(weight_r->GetHandle()));

    int weight_page          = hidden_size_ * ROUND_UP(4 * hidden_size_, 16);
    int weight_byte_count    = num_directions_ * weight_page * sizeof(fp16_t);
    buffer_weight_recurrent_ = RawBuffer(weight_byte_count + NEON_KERNEL_EXTRA_LOAD);
    for (int dir = 0; dir < num_directions_; ++dir) {
        fp16_t *buffer_ptr = buffer_weight_recurrent_.force_to<fp16_t *>() + dir * weight_page;
        TransposeAndPackWeight(weight_r_ptr, buffer_ptr, hidden_size_, hidden_size_);
        weight_r_ptr += 4 * hidden_size_ * hidden_size_;
    }

    return TNN_OK;
}

Status ArmLSTMONNXLayerAcc::AllocateBufferBiasHalf(Blob *bias) {
    // B[iofc] Concatenation of [Wb[iofc], Rb[iofc]], [num_directions, 8*hidden_size]
    fp16_t *bias_ptr = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(bias->GetHandle()));

    int bias_count       = num_directions_ * 4 * hidden_size_;
    int bias_byte_count  = bias_count * sizeof(fp16_t);
    buffer_bias_         = RawBuffer(bias_byte_count);
    auto buffer_bias_ptr = buffer_bias_.force_to<fp16_t *>();
    for (int d = 0; d < num_directions_; ++d) {
        auto src_d = bias_ptr + d * 8 * hidden_size_;
        auto dst_d = buffer_bias_ptr + d * 4 * hidden_size_;
        for (int i = 0; i < hidden_size_; ++i) {
            dst_d[i * 4 + 0] = (fp16_t)(src_d[i + 0 * hidden_size_] + src_d[i + 4 * hidden_size_]);
            dst_d[i * 4 + 1] = (fp16_t)(src_d[i + 1 * hidden_size_] + src_d[i + 5 * hidden_size_]);
            dst_d[i * 4 + 2] = (fp16_t)(src_d[i + 2 * hidden_size_] + src_d[i + 6 * hidden_size_]);
            dst_d[i * 4 + 3] = (fp16_t)(src_d[i + 3 * hidden_size_] + src_d[i + 7 * hidden_size_]);
        }
    }

    return TNN_OK;
}

Status ArmLSTMONNXLayerAcc::ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    const auto batch          = inputs[0]->GetBlobDesc().dims[1];
    const auto direction      = direction_;
    const auto num_directions = num_directions_;
    const auto seq_len        = seq_len_;
    const auto input_size     = input_size_;
    const auto hidden_size    = hidden_size_;

    // X shape [sequence batch_size input_size]
    fp16_t *x = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(inputs[0]->GetHandle()));

    // Y shape [sequence batch_size num_directions *hidden_size]
    fp16_t *y = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(outputs[0]->GetHandle()));

    // Initial states. If not specified, assumed to be 0.
    // shape [num_directions, batch_size, hidden_size]
    auto h_t = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(outputs[1]->GetHandle()));
    auto c_t = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(outputs[2]->GetHandle()));
    if (inputs.size() >= 6) {
        auto h_0 = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(inputs[4]->GetHandle()));
        memcpy((void *)h_t, h_0, num_directions * batch * hidden_size * sizeof(fp16_t));
        auto c_0 = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(inputs[5]->GetHandle()));
        memcpy((void *)c_t, c_0, num_directions * batch * hidden_size * sizeof(fp16_t));
    } else {
        memset((void *)h_t, 0, num_directions * batch * hidden_size * sizeof(fp16_t));
        memset((void *)c_t, 0, num_directions * batch * hidden_size * sizeof(fp16_t));
    }

    fp16_t *w = buffer_weight_input_.force_to<fp16_t *>();
    fp16_t *r = buffer_weight_recurrent_.force_to<fp16_t *>();
    fp16_t *b = buffer_bias_.force_to<fp16_t *>();

    if (direction == 0 || direction == 1) {
        return LstmSingleDirection(x, y, w, r, b, h_t, c_t, batch, direction);
    } else if (direction == 2) {
        // Y shape [num_directions sequence batch_size hidden_size]
        RawBuffer y_temp = RawBuffer(num_directions * seq_len * batch * hidden_size * sizeof(fp16_t));
        auto y0          = y_temp.force_to<fp16_t *>();
        auto y1          = y0 + seq_len * batch * hidden_size;
        LstmSingleDirection(x, y0, w, r, b, h_t, c_t, batch, 0);

        auto w1   = w + ROUND_UP(4 * hidden_size, 16) * input_size;
        auto r1   = r + ROUND_UP(4 * hidden_size, 16) * hidden_size;
        auto b1   = b + 4 * hidden_size;
        auto h_t1 = h_t + batch * hidden_size;
        auto c_t1 = c_t + batch * hidden_size;
        LstmSingleDirection(x, y1, w1, r1, b1, h_t1, c_t1, batch, 1);

        // transpose [num_directions sequence batch_size hidden_size] to [sequence batch_size
        // num_directions*hidden_size]
        for (int i = 0; i < seq_len * batch; i++) {
            auto y0_data = y0 + i * hidden_size;
            auto y1_data = y1 + i * hidden_size;
            auto y_data  = y + i * num_directions * hidden_size;

            memcpy(y_data, y0_data, hidden_size * sizeof(fp16_t));
            memcpy(y_data + hidden_size, y1_data, hidden_size * sizeof(fp16_t));
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "LSTMONNX has invalid direction param");
    }

    return TNN_OK;
}

#endif  // TNN_ARM82
}  // namespace TNN_NS
