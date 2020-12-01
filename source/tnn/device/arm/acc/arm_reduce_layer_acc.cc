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

#include "tnn/device/arm/acc/arm_reduce_layer_acc.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

ArmReduceLayerAcc::~ArmReduceLayerAcc() {}

Status ArmReduceLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto ret = ArmLayerAcc::Init(context, param, resource, inputs, outputs);
    return ret;
}

template <bool post_cal>
void ArmReduceLayerAcc::ReduceChannel(
        float* input_data, float* output_data, DimsVector& dims_in,
        const int c4n, const int c4r, const Float4 axis_n, const int hw_r, const int hw_c, const int hw) {
    float reduce_c = dims_in[1];
    for (int n = 0; n < dims_in[0]; n++) {
        for (int c = 0; c < c4n; c++) {
            OMP_PARALLEL_FOR_
            for (int i = 0; i < hw_c; i++) {
                int p      = i * 16;
                Float4x4 v = Float4x4::ld4(input_data + p);
                Float4 r, t;
                r.set_lane(*(output_data + p), 0);
                r.set_lane(*(output_data + p + 4), 1);
                r.set_lane(*(output_data + p + 8), 2);
                r.set_lane(*(output_data + p + 12), 3);
                int e      = 4;
                if ((c == c4n - 1) && (c4r != 0)) {
                    e = c4r;
                }
                for (int j = 0; j < e; j++) {
                    // t.value = v.value.val[j];
                    v.get_lane(t, j);
                    r       = op_->Calculate(r, t);
                }
                if (c == c4n - 1) {
                    if (post_cal)
                        r = op_->PostCalculate(r, axis_n);
                }
                *(output_data + p)      = r.value[0];
                *(output_data + p + 4)  = r.value[1];
                *(output_data + p + 8)  = r.value[2];
                *(output_data + p + 12) = r.value[3];
            }

            for (int i = 0; i < hw_r; i++) {
                int p = hw_c * 16 + i * 4;
                int e = 4;
                if ((c == c4n - 1) && (c4r != 0)) {
                    e = c4r;
                }
                for (int j = 0; j < e; j++) {
                    *(output_data + p) = op_->Calculate(*(output_data + p), *(input_data + p + j));
                }
                if (c == c4n - 1) {
                    if (post_cal)
                        *(output_data + p) = op_->PostCalculate(*(output_data + p), reduce_c);
                }
            }

            input_data += hw << 2;
        }
        output_data += hw << 2;
    }
}

Status ArmReduceLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ReduceLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    auto input    = inputs[0];
    auto output   = outputs[0];
    auto dims_in  = input->GetBlobDesc().dims;

    int data_byte_size = DataTypeUtils::GetBytesSize(input->GetBlobDesc().data_type);

    if (input->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        auto input_data  = reinterpret_cast<float *>(input->GetHandle().base);
        auto output_data = reinterpret_cast<float *>(output->GetHandle().base);

        if (op_->NeedPreCalculate()) {
            auto in_count = dims_in[0] * ROUND_UP(dims_in[1], 4) * dims_in[2] * dims_in[3];
            OMP_PARALLEL_FOR_
            for (int i = 0; i < in_count; i += 4) {
                Float4 v = Float4::load(input_data + i);
                Float4 r = op_->PreCalculate(v);
                Float4::save(input_data + i, r);
            }
        }

        auto input_data_a  = input_data;
        auto output_data_a = output_data;
        RawBuffer tmp_out[2];
        for (int i = 0; i < param->axis.size(); ++i) {
            int axis = param->axis[i];
            axis     = axis >= 0 ? axis : axis + (int)dims_in.size();

            auto dims_out = dims_in;
            dims_out[axis] = 1;
            int out_count = dims_out[0] * ROUND_UP(dims_out[1], 4) * dims_out[2] * dims_out[3];

            if (i == 0) {
                input_data_a = input_data;
            } else {
                input_data_a = output_data_a;
            }
            if (i == param->axis.size() - 1) {
                output_data_a = output_data;
            } else {
                tmp_out[0] = RawBuffer(out_count * data_byte_size);
                output_data_a = tmp_out[0].force_to<float*>();
            }

            bool post_cal = op_->PosCalculateOnce() ? (i == param->axis.size() - 1) : true;
            if (post_cal) {
                ReduceOneAxis<true>(input_data_a, output_data_a, dims_in, out_count, axis);
            } else {
                ReduceOneAxis<false>(input_data_a, output_data_a, dims_in, out_count, axis);
            }

            tmp_out[1] = tmp_out[0];
        }
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output->GetBlobDesc().data_type);
        return TNNERR_LAYER_ERR;
    }

    return TNN_OK;
}

template<bool post_cal>
void ArmReduceLayerAcc::ReduceOneAxis(float* input_data, float* output_data, DimsVector& dims_in,
                                      int out_count, int axis) {
    int channels  = dims_in[axis];
    int outer_dim = DimsVectorUtils::Count(dims_in, 0, axis);
    int inner_dim = DimsVectorUtils::Count(dims_in, axis + 1);

    int c4u  = ROUND_UP(dims_in[1], 4);
    int c4n  = UP_DIV(dims_in[1], 4);
    int c4r  = dims_in[1] % 4;
    int hw   = dims_in[2] * dims_in[3];
    int hw_c = hw / 4;
    int hw_r = hw % 4;
    int w4   = dims_in[3] * 4;
    int h4   = dims_in[2] * 4;
    Float4 axis_n(dims_in[axis]);

    op_->DataInit(output_data, out_count);

    if (axis == 1) {
        ReduceChannel<post_cal>(input_data, output_data, dims_in, c4n, c4r, axis_n, hw_r, hw_c, hw);
    } else if (axis == 0) {
        int outer_dim = dims_in[0];
        int inner_dim = c4u * hw;
        int count     = outer_dim * inner_dim;

        OMP_PARALLEL_FOR_
        for (int i = 0; i < inner_dim; i += 4) {
            Float4 r = op_->DataInit();
            for (int j = 0; j < count; j += inner_dim) {
                Float4 v = Float4::load(input_data + j + i);
                r        = op_->Calculate(r, v);
            }
            if (post_cal)
                r = op_->PostCalculate(r, axis_n);
            Float4::save(output_data + i, r);
        }
    } else if (axis == 2) {
        for (int n = 0; n < dims_in[0]; n++) {
            for (int c = 0; c < c4n; c++) {
                OMP_PARALLEL_FOR_
                for (int w = 0; w < w4; w += 4) {
                    Float4 r = op_->DataInit();
                    for (int h = 0; h < h4; h += 4) {
                        Float4 v = Float4::load(input_data + w + h * dims_in[3]);
                        r = op_->Calculate(r, v);
                    }
                    if (post_cal)
                        r = op_->PostCalculate(r, axis_n);
                    Float4::save(output_data + w, r);
                }
                input_data += hw << 2;
                output_data += dims_in[3] << 2;
            }
        }
    } else {
        for (int n = 0; n < dims_in[0]; n++) {
            for (int c = 0; c < c4n; c++) {
                OMP_PARALLEL_FOR_
                for (int h = 0; h < h4; h += 4) {
                    Float4 r = op_->DataInit();
                    for (int w = 0; w < w4; w += 4) {
                        Float4 v = Float4::load(input_data + w + h * dims_in[3]);
                        r        = op_->Calculate(r, v);
                    }
                    if (post_cal)
                        r = op_->PostCalculate(r, axis_n);
                    Float4::save(output_data + h, r);
                }
                input_data += hw << 2;
                output_data += dims_in[2] << 2;
            }
        }
    }

    dims_in[axis] = 1;
}

}  // namespace TNN_NS
