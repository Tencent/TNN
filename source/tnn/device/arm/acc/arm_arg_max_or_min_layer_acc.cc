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

#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

Status ArmArgMaxOrMinLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto in_data_type   = inputs[0]->GetBlobDesc().data_type;
    if (in_data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    } else if (in_data_type == DATA_TYPE_BFP16) {
        return Exec<bfp16_t>(inputs, outputs);
    } else {
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }
}

static Status ExecDimC() {
    return TNN_OK;
}

#define ARGMAXMINCAL(input_ptr_base, output_ptr_base, mode)                                 \
    Float4 guard_index(0);                                                                  \
    Float4 guard_value = Float4::load(input_ptr_base);                                      \
    for (int r = 1; r < reduce_dim; ++r) {                                                  \
        auto *input_ptr_r = input_ptr_base + r * outer_dim;                                 \
        Float4 cur_index(r);                                                                \
        Float4 cur_value = Float4::load(input_ptr_r);                                       \
        if (mode == 0) {                                                                    \
            guard_index = Float4::bsl_clt(cur_value, guard_value, cur_index, guard_index);  \
            guard_value = Float4::min(cur_value, guard_value);                              \
        } else {                                                                            \
            guard_index = Float4::bsl_cgt(cur_value, guard_value, cur_index, guard_index);  \
            guard_value = Float4::max(cur_value, guard_value);                              \
        }                                                                                   \
    }                                                                                       \
    Float4::save(output_ptr_base, guard_index);

template<typename T, int mode>
static Status ExecDimN(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_dims  = inputs[0]->GetBlobDesc().dims;

    int inner_dim    = 1;
    int reduce_dim   = input_dims[0];
    int outer_dim    = UP_DIV(input_dims[1], 4) * input_dims[2] * input_dims[3] * 4;

    auto *input_ptr  = static_cast<T *>(inputs[0]->GetHandle().base);
    auto *output_ptr = static_cast<T *>(outputs[0]->GetHandle().base);
    OMP_PARALLEL_FOR_
    for (int o = 0; o < outer_dim; o += 4) {
        auto *input_ptr_o  = input_ptr + o;
        auto *output_ptr_o = output_ptr + o;
        ARGMAXMINCAL(input_ptr_o, output_ptr_o, mode);
    }

    return TNN_OK;
}

template<typename T, int mode>
static Status ExecDimH(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_dims  = inputs[0]->GetBlobDesc().dims;

    int inner_dim    = input_dims[0] * UP_DIV(input_dims[1], 4);
    int reduce_dim   = input_dims[2];
    int outer_dim    = input_dims[3] * 4;

    auto *input_ptr  = static_cast<T *>(inputs[0]->GetHandle().base);
    auto *output_ptr = static_cast<T *>(outputs[0]->GetHandle().base);
    OMP_PARALLEL_FOR_
    for (int i = 0; i < inner_dim; ++i) {
        auto *input_ptr_i  = input_ptr + i * reduce_dim * outer_dim;
        auto *output_ptr_i = output_ptr + i * outer_dim;
        for (int o = 0; o < outer_dim; o += 4) {
            auto *input_ptr_o  = input_ptr_i + o;
            auto *output_ptr_o = output_ptr_i + o;
            ARGMAXMINCAL(input_ptr_o, output_ptr_o, mode);
        }
    }

    return TNN_OK;
}

template<typename T, int mode>
static Status ExecDimW(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input_dims  = inputs[0]->GetBlobDesc().dims;

    int inner_dim    = input_dims[0] * UP_DIV(input_dims[1], 4) * input_dims[2];
    int reduce_dim   = input_dims[3];
    int outer_dim    = 4;

    auto *input_ptr  = static_cast<T *>(inputs[0]->GetHandle().base);
    auto *output_ptr = static_cast<T *>(outputs[0]->GetHandle().base);
    OMP_PARALLEL_FOR_
    for (int i = 0; i < inner_dim; ++i) {
        auto *input_ptr_i  = input_ptr + i * reduce_dim * outer_dim;
        auto *output_ptr_i = output_ptr + i * outer_dim;
        ARGMAXMINCAL(input_ptr_i, output_ptr_i, mode);
    }

    return TNN_OK;
}

template <typename T>
Status ArmArgMaxOrMinLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ArgMaxOrMinLayerParam *>(param_);
    int axis   = param->axis;

    if (axis == 0) {
        if (param->mode == 0) {
            return ExecDimN<T, 0>(inputs, outputs);
        } else {
            return ExecDimN<T, 1>(inputs, outputs);
        }
    } else if (axis == 1) {
        return ExecDimC();
    } else if (axis == 2) {
        if (param->mode == 0) {
            return ExecDimH<T, 0>(inputs, outputs);
        } else {
            return ExecDimH<T, 1>(inputs, outputs);
        }
    } else if (axis == 3) {
        if (param->mode == 0) {
            return ExecDimW<T, 0>(inputs, outputs);
        } else {
            return ExecDimW<T, 1>(inputs, outputs);
        }
    } else {
        return Status(TNNERR_PARAM_ERR, "argmax or argmin axis not support");
    }
}

REGISTER_ARM_ACC(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

}  // namespace TNN_NS
