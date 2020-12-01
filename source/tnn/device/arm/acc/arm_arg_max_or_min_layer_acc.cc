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
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

DECLARE_ARM_ACC(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

Status ArmArgMaxOrMinLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    // Currently, only float data type is supported as output.
    auto out_data_type = outputs[0]->GetBlobDesc().data_type;
    if (out_data_type != DATA_TYPE_FLOAT) {
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }

    auto in_data_type = inputs[0]->GetBlobDesc().data_type;
    if (in_data_type == DATA_TYPE_FLOAT) {
        return Exec<float>(inputs, outputs);
    } else if (in_data_type == DATA_TYPE_BFP16) {
        return Exec<bfp16_t>(inputs, outputs);
    } else {
        return Status(TNNERR_MODEL_ERR, "Error: layer acc dont support datatype");
    }
}

#define ExecWithoutWorkspace                                                                    \
    if (param->mode == 0) {                                                                     \
        return ExecImpl<T, 0>(inputs, outputs, inner_dim, reduce_dim, outer_dim);               \
    } else {                                                                                    \
        return ExecImpl<T, 1>(inputs, outputs, inner_dim, reduce_dim, outer_dim);               \
    }                                                                                           \

#define ExecWithWorkspace                                                                       \
    size_in_bytes = outer_dim * input_byte_size;                                                \
    workspace     = context_->GetSharedWorkSpace(size_in_bytes);                                \
    if (param->mode == 0) {                                                                     \
        return ExecImpl<T, 0>(inputs, outputs, workspace, inner_dim, reduce_dim, outer_dim);    \
    } else {                                                                                    \
        return ExecImpl<T, 1>(inputs, outputs, workspace, inner_dim, reduce_dim, outer_dim);    \
    }                                                                                           \

#define ExecDimCWithoutWorkspace                                                                \
    if (param->mode == 0) {                                                                     \
        return ExecImplC<T, 0>(inputs, outputs, inner_dim, ic, outer_dim);                      \
    } else {                                                                                    \
        return ExecImplC<T, 1>(inputs, outputs, inner_dim, ic, outer_dim);                      \
    }                                                                                           \

#define ExecDimCWithWorkspace                                                                   \
    size_in_bytes = outer_dim * input_byte_size;                                                \
    workspace     = context_->GetSharedWorkSpace(size_in_bytes);                                \
    if (param->mode == 0) {                                                                     \
        return ExecImplC<T, 0>(inputs, outputs, workspace, inner_dim, ic, outer_dim);           \
    } else {                                                                                    \
        return ExecImplC<T, 1>(inputs, outputs, workspace, inner_dim, ic, outer_dim);           \
    }                                                                                           \

template<typename T, int mode>
static void UpdateOnePlane(T *input_ptr_base, float *output_ptr_base, T *workspace_ptr, int reduce_dim, int outer_dim) {
    memcpy(workspace_ptr, input_ptr_base, outer_dim * sizeof(T));
    memset(output_ptr_base, 0, outer_dim * sizeof(float));
    for (int r = 1; r < reduce_dim; ++r) {
        auto *input_ptr_r = input_ptr_base + r * outer_dim;
        Float4 cur_index(r);
        for (int o = 0; o < outer_dim; o += 4) {
            Float4 guard_index = Float4::load(output_ptr_base + o);
            Float4 guard_value = Float4::load(workspace_ptr + o);
            Float4 cur_value   = Float4::load(input_ptr_r + o);
            if (mode == 0) {
                guard_index = Float4::bsl_clt(cur_value, guard_value, cur_index, guard_index);
                guard_value = Float4::min(cur_value, guard_value);
            } else {
                guard_index = Float4::bsl_cgt(cur_value, guard_value, cur_index, guard_index);
                guard_value = Float4::max(cur_value, guard_value);
            }
            Float4::save(output_ptr_base + o, guard_index);
            Float4::save(workspace_ptr + o, guard_value);
        }
    }
}

// loop order: inner_dim -> reduce_dim -> outer_dim
// cache outer result with workspace
template <typename T, int mode>
static Status ExecImpl(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                       void *workspace, int inner_dim, int reduce_dim, int outer_dim) {
    auto *input_ptr     = static_cast<T *>(inputs[0]->GetHandle().base);
    auto *output_ptr    = static_cast<float *>(outputs[0]->GetHandle().base);
    auto *workspace_ptr = static_cast<T *>(workspace);

    for (int i = 0; i < inner_dim; ++i) {
        auto *input_ptr_i  = input_ptr + i * reduce_dim * outer_dim;
        auto *output_ptr_i = output_ptr + i * outer_dim;
        UpdateOnePlane<T, mode>(input_ptr_i, output_ptr_i, workspace_ptr, reduce_dim, outer_dim);
    }

    return TNN_OK;
}

template <typename T, int mode>
static Float4 GetOneValue(T *input_ptr_base, int reduce_dim, int outer_dim, Float4 &guard_index) {
    Float4 guard_value = Float4::load(input_ptr_base);
    for (int r = 1; r < reduce_dim; ++r) {
        auto *input_ptr_r = input_ptr_base + r * outer_dim;
        Float4 cur_index(r);
        Float4 cur_value = Float4::load(input_ptr_r);
        if (mode == 0) {
            guard_index = Float4::bsl_clt(cur_value, guard_value, cur_index, guard_index);
            guard_value = Float4::min(cur_value, guard_value);
        } else {
            guard_index = Float4::bsl_cgt(cur_value, guard_value, cur_index, guard_index);
            guard_value = Float4::max(cur_value, guard_value);
        }
    }
    return guard_value;
}

// loop order: inner_dim -> outer_dim -> reduce_dim
// no need for workspace
template <typename T, int mode>
static Status ExecImpl(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                       int inner_dim, int reduce_dim, int outer_dim) {
    auto *input_ptr  = static_cast<T *>(inputs[0]->GetHandle().base);
    auto *output_ptr = static_cast<float *>(outputs[0]->GetHandle().base);

    OMP_PARALLEL_FOR_
    for (int i = 0; i < inner_dim; ++i) {
        auto *input_ptr_i  = input_ptr + i * reduce_dim * outer_dim;
        auto *output_ptr_i = output_ptr + i * outer_dim;
        for (int o = 0; o < outer_dim; o += 4) {
            auto *input_ptr_o  = input_ptr_i + o;
            auto *output_ptr_o = output_ptr_i + o;
            Float4 guard_index(0);
            Float4 guard_value = GetOneValue<T, mode>(input_ptr_o, reduce_dim, outer_dim, guard_index);
            Float4::save(output_ptr_o, guard_index);
        }
    }

    return TNN_OK;
}

template <int mode>
static void CompareC4Impl(const Float4 &guard_value, const Float4 &guard_index, int start, int end,
                          float &final_value, float &final_index) {
    for (int c = start; c < end; ++c) {
        float cur_value = guard_value[c];
        float cur_index = guard_index[c] * 4 + c;
        if ((mode == 0 && cur_value < final_value) ||
            (mode == 1 && cur_value > final_value)) {
            final_value = cur_value;
            final_index = cur_index;
        }
        if (cur_value == final_value &&
            cur_index < final_index) {
            final_index = cur_index;
        }
    }
}

template <typename T, int mode>
static Float4 CompareC4(const Float4 &guard_value, const Float4 &guard_index,
                        int reduce_dim_c4, int reduce_dim_r4, T *input_ptr_r) {
    float final_value = guard_value[0];
    float final_index = guard_index[0] * 4;
    // compare 4 channels
    if (reduce_dim_c4 != 0) {
        CompareC4Impl<mode>(guard_value, guard_index, 1, 4, final_value, final_index);
    }
    // compare remain channels
    if (reduce_dim_r4 != 0) {
        Float4 cur_index(reduce_dim_c4);
        Float4 cur_value = Float4::load(input_ptr_r);
        CompareC4Impl<mode>(cur_value, cur_index, 0, reduce_dim_r4, final_value, final_index);
    }
    Float4 result(0);
    result.set_lane(final_index, 0);
    return result;
}

template <typename T, int mode>
static Status ExecImplC(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                        int inner_dim, int ic, int outer_dim) {
    auto *input_ptr   = static_cast<T *>(inputs[0]->GetHandle().base);
    auto *output_ptr  = static_cast<float *>(outputs[0]->GetHandle().base);

    int reduce_dim    = UP_DIV(ic, 4);
    int reduce_dim_r4 = ic % 4;
    int reduce_dim_c4 = (reduce_dim_r4 == 0) ? reduce_dim : reduce_dim - 1;

    for (int i = 0; i < inner_dim; ++i) {
        auto *input_ptr_i  = input_ptr + i * reduce_dim * outer_dim;
        auto *output_ptr_i = output_ptr + i * outer_dim;

        OMP_PARALLEL_FOR_
        for (int o = 0; o < outer_dim; o += 4) {
            auto *input_ptr_o  = input_ptr_i + o;
            auto *output_ptr_o = output_ptr_i + o;

            Float4 guard_index(0);
            Float4 guard_value = GetOneValue<T, mode>(input_ptr_o, reduce_dim_c4, outer_dim, guard_index);

            auto *input_ptr_r  = input_ptr_o + reduce_dim_c4 * outer_dim;
            Float4 result      = CompareC4<T, mode>(guard_value, guard_index, reduce_dim_c4,
                                                    reduce_dim_r4, input_ptr_r);

            Float4::save(output_ptr_o, result);
        }
    }

    return TNN_OK;
}

template <typename T, int mode>
static Status ExecImplC(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs,
                        void *workspace, int inner_dim, int ic, int outer_dim) {
    auto *input_ptr     = static_cast<T *>(inputs[0]->GetHandle().base);
    auto *output_ptr    = static_cast<float *>(outputs[0]->GetHandle().base);
    auto *workspace_ptr = static_cast<T *>(workspace);

    int reduce_dim    = UP_DIV(ic, 4);
    int reduce_dim_r4 = ic % 4;
    int reduce_dim_c4 = (reduce_dim_r4 == 0) ? reduce_dim : reduce_dim - 1;

    for (int i = 0; i < inner_dim; ++i) {
        auto *input_ptr_i  = input_ptr + i * reduce_dim * outer_dim;
        auto *output_ptr_i = output_ptr + i * outer_dim;
        UpdateOnePlane<T, mode>(input_ptr_i, output_ptr_i, workspace_ptr, reduce_dim_c4, outer_dim);

        for (int o = 0; o < outer_dim; o += 4) {
            auto *input_ptr_o     = input_ptr_i + o;
            auto *output_ptr_o    = output_ptr_i + o;
            auto *workspace_ptr_o = workspace_ptr + o;

            Float4 guard_value = Float4::load(workspace_ptr_o);
            Float4 guard_index = Float4::load(output_ptr_o);

            auto *input_ptr_r  = input_ptr_o + reduce_dim_c4 * outer_dim;
            Float4 result      = CompareC4<T, mode>(guard_value, guard_index, reduce_dim_c4,
                                                    reduce_dim_r4, input_ptr_r);

            Float4::save(output_ptr_o, result);
        }
    }

    return TNN_OK;
}

template <typename T>
Status ArmArgMaxOrMinLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto dims_input = inputs[0]->GetBlobDesc().dims;
    auto in         = dims_input[0];
    auto ic         = dims_input[1];
    auto ic_r4      = ROUND_UP(dims_input[1], 4);
    auto ih         = dims_input[2];
    auto iw         = dims_input[3];

    int input_byte_size = DataTypeUtils::GetBytesSize(inputs[0]->GetBlobDesc().data_type);
    int size_in_bytes   = 0;
    void *workspace     = nullptr;

    auto param = dynamic_cast<ArgMaxOrMinLayerParam *>(param_);
    CHECK_PARAM_NULL(param);
    int axis   = param->axis;

    if (axis == 0) {
        int inner_dim  = 1;
        int reduce_dim = in;
        int outer_dim  = ic_r4 * ih * iw;
        ExecWithWorkspace;
        // ExecWithoutWorkspace;
    } else if (axis == 1) {
        int inner_dim     = in;
        int outer_dim     = ih * iw * 4;
        ExecDimCWithWorkspace;
        // ExecDimCWithoutWorkspace;
    } else if (axis == 2) {
        int inner_dim  = in * ic_r4 / 4;
        int reduce_dim = ih;
        int outer_dim  = iw * 4;
        ExecWithWorkspace;
        // ExecWithoutWorkspace;
    } else if (axis == 3) {
        int inner_dim  = in * ic_r4 / 4 * ih;
        int reduce_dim = iw;
        int outer_dim  = 4;
        // ExecWithWorkspace;
        ExecWithoutWorkspace;
    } else {
        return Status(TNNERR_PARAM_ERR, "argmax or argmin axis not support");
    }
}

REGISTER_ARM_ACC(ArgMaxOrMin, LAYER_ARG_MAX_OR_MIN);

}  // namespace TNN_NS
