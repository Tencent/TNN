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

#include "tnn/device/arm/acc/arm_lstm_layer_acc.h"

#include "tnn/device/arm/acc/compute/compute.h"
#include "tnn/device/arm/acc/compute/gemm_function.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

static void LstmActivate(const int count, const float *g_ptr, float *c_ptr, float *h_ptr, float *o_ptr) {
    OMP_PARALLEL_FOR_
    for (int q = 0; q < count - 3; q += 4) {
        Float4x4 gates_iofc = Float4x4::ld4(g_ptr + q * 4);
        Float4 I, O, F, C;
        gates_iofc.get_lane(I, 0);
        gates_iofc.get_lane(O, 1);
        gates_iofc.get_lane(F, 2);
        gates_iofc.get_lane(C, 3);

        I = Float4::sigmoid(I);
        F = Float4::sigmoid(F);
        O = Float4::sigmoid(O);
        C = Float4::tanh(C);

        Float4 cell2 = F * Float4::load(c_ptr + q) + I * C;
        Float4 H     = O * Float4::tanh(cell2);
        Float4::save(c_ptr + q, cell2);
        Float4::save(h_ptr + q, H);
        Float4::save(o_ptr + q, H);
    }
    int remain = count % 4;
    int offset = count / 4 * 4;
    g_ptr += offset * 4;
    c_ptr += offset;
    h_ptr += offset;
    o_ptr += offset;
    if (remain) {
        Float4x4 gates_iofc = Float4x4::ld4(g_ptr);
        Float4 I, O, F, C;
        gates_iofc.get_lane(I, 0);
        gates_iofc.get_lane(O, 1);
        gates_iofc.get_lane(F, 2);
        gates_iofc.get_lane(C, 3);

        I = Float4::sigmoid(I);
        F = Float4::sigmoid(F);
        O = Float4::sigmoid(O);
        C = Float4::tanh(C);

        Float4 c_old;
        for (int r = 0; r < remain; ++r) {
            c_old.set_lane(c_ptr[r], r);
        }
        Float4 cell2 = F * c_old + I * C;
        Float4 H     = O * Float4::tanh(cell2);
        for (int r = 0; r < remain; ++r) {
            c_ptr[r] = cell2[r];
            h_ptr[r] = H[r];
            o_ptr[r] = H[r];
        }
    }
}

Status ArmLSTMONNXLayerAcc::LstmSingleDirection(const float *x, float *y, const float *w, const float *r,
                                                const float *b, float *h_t, float *c_t, const int batch_size,
                                                int reverse) {
    const int input_size  = input_size_;
    const int hidden_size = hidden_size_;
    const int seq_len     = seq_len_;

    int gates_count      = seq_len * batch_size * hidden_size * 4;
    int input_pack_count = MAX(seq_len * batch_size * input_size, batch_size * hidden_size);
    auto workspace =
        context_->GetSharedWorkSpace((gates_count + input_pack_count) * sizeof(float) + NEON_KERNEL_EXTRA_LOAD);
    auto gates_ptr      = reinterpret_cast<float *>(workspace);
    auto input_pack_ptr = gates_ptr + gates_count;
    for (int i = 0; i < seq_len * batch_size; ++i) {
        float *gates_i = gates_ptr + i * hidden_size * 4;
        memcpy(gates_i, b, hidden_size * 4 * sizeof(float));
    }

    GemmFloatPackA(seq_len * batch_size, hidden_size * 4, input_size, x, input_pack_ptr, input_size, w, hidden_size * 4,
                   gates_ptr, hidden_size * 4);

    for (int t = 0; t < seq_len; ++t) {
        int ti         = reverse ? seq_len - 1 - t : t;
        float *y_t     = y + ti * batch_size * hidden_size;
        float *gates_t = gates_ptr + ti * batch_size * hidden_size * 4;

        GemmFloatPackA(batch_size, hidden_size * 4, hidden_size, h_t, input_pack_ptr, hidden_size, r, hidden_size * 4,
                       gates_t, hidden_size * 4);

        LstmActivate(batch_size * hidden_size, gates_t, c_t, h_t, y_t);
    }

    return TNN_OK;
}

// [4, hidden_size, input_size] -> transpose -> [input_size, hidden_size, 4]
// [input_size, hidden_size * 4] -> PackB_8 -> [hidden_size * 4 / 8, input_size, 8]
static void TransposeAndPackWeight(const float *src, float *dst, int input_size, int hidden_size) {
    RawBuffer tmp_transpose = RawBuffer(input_size * hidden_size * 4 * sizeof(float));
    float *src_transpose    = tmp_transpose.force_to<float *>();
    const float *vsrc[4];
    vsrc[0]   = src;
    vsrc[1]   = vsrc[0] + input_size * hidden_size;
    vsrc[2]   = vsrc[1] + input_size * hidden_size;
    vsrc[3]   = vsrc[2] + input_size * hidden_size;
    int count = 0;
    for (int i = 0; i < input_size; ++i) {
        for (int h = 0; h < hidden_size; ++h) {
            src_transpose[count++] = vsrc[0][h * input_size + i];
            src_transpose[count++] = vsrc[1][h * input_size + i];
            src_transpose[count++] = vsrc[2][h * input_size + i];
            src_transpose[count++] = vsrc[3][h * input_size + i];
        }
    }
    PackB_8(input_size, hidden_size * 4, src_transpose, hidden_size * 4, dst);
}

ArmLSTMONNXLayerAcc::~ArmLSTMONNXLayerAcc() {}

Status ArmLSTMONNXLayerAcc::AllocateBufferWeightInput(Blob *weight_i) {
    // W[iofc], weight tensor for the gates, shape [num_directions, 4*hidden_size, input_size]
    float *weight_i_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(weight_i->GetHandle()));

    int weight_page       = input_size_ * ROUND_UP(4 * hidden_size_, 8);
    int weight_byte_count = num_directions_ * weight_page * sizeof(float);
    buffer_weight_input_  = RawBuffer(weight_byte_count + NEON_KERNEL_EXTRA_LOAD);
    for (int dir = 0; dir < num_directions_; ++dir) {
        float *buffer_ptr = buffer_weight_input_.force_to<float *>() + dir * weight_page;
        TransposeAndPackWeight(weight_i_ptr, buffer_ptr, input_size_, hidden_size_);
        weight_i_ptr += 4 * hidden_size_ * input_size_;
    }

    return TNN_OK;
}

Status ArmLSTMONNXLayerAcc::AllocateBufferWeightRecurrent(Blob *weight_r) {
    // R[iofc], recurrence weight tensor, shape [num_directions, 4*hidden_size, hidden_size]
    float *weight_r_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(weight_r->GetHandle()));

    int weight_page          = hidden_size_ * ROUND_UP(4 * hidden_size_, 8);
    int weight_byte_count    = num_directions_ * weight_page * sizeof(float);
    buffer_weight_recurrent_ = RawBuffer(weight_byte_count + NEON_KERNEL_EXTRA_LOAD);
    for (int dir = 0; dir < num_directions_; ++dir) {
        float *buffer_ptr = buffer_weight_recurrent_.force_to<float *>() + dir * weight_page;
        TransposeAndPackWeight(weight_r_ptr, buffer_ptr, hidden_size_, hidden_size_);
        weight_r_ptr += 4 * hidden_size_ * hidden_size_;
    }

    return TNN_OK;
}

Status ArmLSTMONNXLayerAcc::AllocateBufferBias(Blob *bias) {
    // B[iofc] Concatenation of [Wb[iofc], Rb[iofc]], [num_directions, 8*hidden_size]
    float *bias_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(bias->GetHandle()));

    int bias_count       = num_directions_ * 4 * hidden_size_;
    int bias_byte_count  = bias_count * sizeof(float);
    buffer_bias_         = RawBuffer(bias_byte_count);
    auto buffer_bias_ptr = buffer_bias_.force_to<float *>();
    for (int d = 0; d < num_directions_; ++d) {
        auto src_d = bias_ptr + d * 8 * hidden_size_;
        auto dst_d = buffer_bias_ptr + d * 4 * hidden_size_;
        for (int i = 0; i < hidden_size_; ++i) {
            dst_d[i * 4 + 0] = src_d[i + 0 * hidden_size_] + src_d[i + 4 * hidden_size_];
            dst_d[i * 4 + 1] = src_d[i + 1 * hidden_size_] + src_d[i + 5 * hidden_size_];
            dst_d[i * 4 + 2] = src_d[i + 2 * hidden_size_] + src_d[i + 6 * hidden_size_];
            dst_d[i * 4 + 3] = src_d[i + 3 * hidden_size_] + src_d[i + 7 * hidden_size_];
        }
    }

    return TNN_OK;
}

Status ArmLSTMONNXLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                 const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    auto layer_param = dynamic_cast<LSTMONNXLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    direction_      = layer_param->direction;
    num_directions_ = direction_ >= 2 ? 2 : 1;
    hidden_size_    = layer_param->hidden_size;

    if (inputs.size() < 4) {
        return Status(TNNERR_LAYER_ERR, "LSTM has invalid inputs");
    }
    if (outputs.size() < 3) {
        return Status(TNNERR_LAYER_ERR, "LSTM has invalid outputs");
    }
    seq_len_    = inputs[0]->GetBlobDesc().dims[0];
    input_size_ = DimsVectorUtils::Count(inputs[0]->GetBlobDesc().dims, 2);

    auto input_data_type = inputs[0]->GetBlobDesc().data_type;
    if (input_data_type == DATA_TYPE_FLOAT) {
        RETURN_ON_NEQ((AllocateBufferWeightInput(inputs[1])), TNN_OK);
        RETURN_ON_NEQ((AllocateBufferWeightRecurrent(inputs[2])), TNN_OK);
        RETURN_ON_NEQ(AllocateBufferBias(inputs[3]), TNN_OK);
    }
#if TNN_ARM82
    else if (input_data_type == DATA_TYPE_HALF) {
        RETURN_ON_NEQ((AllocateBufferWeightInputHalf(inputs[1])), TNN_OK);
        RETURN_ON_NEQ((AllocateBufferWeightRecurrentHalf(inputs[2])), TNN_OK);
        RETURN_ON_NEQ(AllocateBufferBiasHalf(inputs[3]), TNN_OK);
    }
#endif  // TNN_ARM82
    else {
        LOGE("ARM LSTM not support data type: %d\n", input_data_type);
        return Status(TNNERR_LAYER_ERR, "ARM LSTM not support data type");
    }
    return TNN_OK;
}

template <typename T>
Status ArmLSTMONNXLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    const auto batch          = inputs[0]->GetBlobDesc().dims[1];
    const auto direction      = direction_;
    const auto num_directions = num_directions_;
    const auto seq_len        = seq_len_;
    const auto input_size     = input_size_;
    const auto hidden_size    = hidden_size_;

    // X shape [sequence batch_size input_size]
    T *x = reinterpret_cast<T *>(GetBlobHandlePtr(inputs[0]->GetHandle()));

    // Y shape [sequence batch_size num_directions *hidden_size]
    T *y = reinterpret_cast<T *>(GetBlobHandlePtr(outputs[0]->GetHandle()));

    // Initial states. If not specified, assumed to be 0.
    // shape [num_directions, batch_size, hidden_size]
    auto h_t = reinterpret_cast<T *>(GetBlobHandlePtr(outputs[1]->GetHandle()));
    auto c_t = reinterpret_cast<T *>(GetBlobHandlePtr(outputs[2]->GetHandle()));
    if (inputs.size() >= 6) {
        auto h_0 = reinterpret_cast<T *>(GetBlobHandlePtr(inputs[4]->GetHandle()));
        memcpy((void *)h_t, h_0, num_directions * batch * hidden_size * sizeof(T));
        auto c_0 = reinterpret_cast<T *>(GetBlobHandlePtr(inputs[5]->GetHandle()));
        memcpy((void *)c_t, c_0, num_directions * batch * hidden_size * sizeof(T));
    } else {
        memset((void *)h_t, 0, num_directions * batch * hidden_size * sizeof(T));
        memset((void *)c_t, 0, num_directions * batch * hidden_size * sizeof(T));
    }

    T *w = buffer_weight_input_.force_to<T *>();
    T *r = buffer_weight_recurrent_.force_to<T *>();
    T *b = buffer_bias_.force_to<T *>();

    if (direction == 0 || direction == 1) {
        return LstmSingleDirection(x, y, w, r, b, h_t, c_t, batch, direction);
    } else if (direction == 2) {
        // Y shape [num_directions sequence batch_size hidden_size]
        RawBuffer y_temp = RawBuffer(num_directions * seq_len * batch * hidden_size * sizeof(T));
        auto y0          = y_temp.force_to<T *>();
        auto y1          = y0 + seq_len * batch * hidden_size;
        LstmSingleDirection(x, y0, w, r, b, h_t, c_t, batch, 0);

        auto w1   = w + ROUND_UP(4 * hidden_size, 8) * input_size;
        auto r1   = r + ROUND_UP(4 * hidden_size, 8) * hidden_size;
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

            memcpy(y_data, y0_data, hidden_size * sizeof(T));
            memcpy(y_data + hidden_size, y1_data, hidden_size * sizeof(T));
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "LSTMONNX has invalid direction param");
    }

    return TNN_OK;
}

Status ArmLSTMONNXLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_data_type = inputs[0]->GetBlobDesc().data_type;
    if (input_data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    }
#if TNN_ARM82
    else if (input_data_type == DATA_TYPE_HALF) {
        return ExecFp16(inputs, outputs);
    }
#endif  // TNN_ARM82
    else {
        LOGE("ARM LSTM not support data type: %d\n", input_data_type);
        return Status(TNNERR_LAYER_ERR, "ARM LSTM not support data type");
    }
    return TNN_OK;
}

REGISTER_ARM_ACC(LSTMONNX, LAYER_LSTMONNX);
REGISTER_ARM_PRECISION_FP16(LAYER_LSTMONNX)
REGISTER_ARM_LAYOUT(LAYER_LSTMONNX, DATA_FORMAT_NCHW)

}  // namespace TNN_NS
