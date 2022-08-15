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
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/omp_utils.h"

namespace TNN_NS {

ArmReduceLayerAcc::~ArmReduceLayerAcc() {}

Status ArmReduceLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto ret = ArmLayerAcc::Init(context, param, resource, inputs, outputs);
    return ret;
}

template <bool post_cal>
void ArmReduceLayerAcc::ReduceChannel(float *input_data, float *output_data, DimsVector &dims_in, const int c4n,
                                      const int c4r, const Float4 axis_n, const int hw_r, const int hw_c,
                                      const int hw) {
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
                int e = 4;
                if ((c == c4n - 1) && (c4r != 0)) {
                    e = c4r;
                }
                for (int j = 0; j < e; j++) {
                    // t.value = v.value.val[j];
                    v.get_lane(t, j);
                    r = op_->Calculate(r, t);
                }
                if (c == c4n - 1) {
                    if (post_cal)
                        r = op_->PostCalculate(r, axis_n);
                }
                *(output_data + p)      = r[0];
                *(output_data + p + 4)  = r[1];
                *(output_data + p + 8)  = r[2];
                *(output_data + p + 12) = r[3];
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

static bool NeedRepack(const DimsVector &src_dims, const DimsVector &dst_dims) {
    if (dst_dims.size() < 2) {
        // skip repack for one-dimensional tensor
        return false;
    }
    return ((src_dims.size() != dst_dims.size()) && (src_dims[1] != dst_dims[1]));
}

Status ArmReduceLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto param = dynamic_cast<ReduceLayerParam *>(param_);
    CHECK_PARAM_NULL(param);

    auto input   = inputs[0];
    auto output  = outputs[0];
    auto dims_in = input->GetBlobDesc().dims;
    if (dims_in.size() == 1) {
        // treat 1D blob with nc4hw4 format as 2D blob
        dims_in.push_back(1);
    }

    int data_byte_size = DataTypeUtils::GetBytesSize(input->GetBlobDesc().data_type);

    if (input->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        auto input_data  = reinterpret_cast<float *>(GetBlobHandlePtr(input->GetHandle()));
        auto output_data = reinterpret_cast<float *>(GetBlobHandlePtr(output->GetHandle()));

        if (op_->NeedPreCalculate()) {
            auto in_count = dims_in[0] * ROUND_UP(dims_in[1], 4) * DimsVectorUtils::Count(dims_in, 2);
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
            axis     = axis >= 0 ? axis : axis + (int)input->GetBlobDesc().dims.size();

            auto dims_out  = dims_in;
            dims_out[axis] = 1;
            int out_count  = dims_out[0] * ROUND_UP(dims_out[1], 4) * DimsVectorUtils::Count(dims_out, 2);

            if (i == 0) {
                input_data_a = input_data;
            } else {
                input_data_a = output_data_a;
            }
            if (i == param->axis.size() - 1 && !NeedRepack(dims_out, output->GetBlobDesc().dims)) {
                output_data_a = output_data;
            } else {
                tmp_out[0]    = RawBuffer(out_count * data_byte_size);
                output_data_a = tmp_out[0].force_to<float *>();
            }

            bool post_cal = op_->PosCalculateOnce() ? (i == param->axis.size() - 1) : true;
            if (post_cal) {
                ReduceOneAxis<true>(input_data_a, output_data_a, dims_in, out_count, axis);
            } else {
                ReduceOneAxis<false>(input_data_a, output_data_a, dims_in, out_count, axis);
            }

            tmp_out[1] = tmp_out[0];
        }

        if (NeedRepack(dims_in, output->GetBlobDesc().dims)) {
            tmp_out[0]    = RawBuffer(ROUND_UP(DimsVectorUtils::Count(dims_in), 4) * data_byte_size);
            auto tmp_data = tmp_out[0].force_to<float *>();
            auto c_src    = DimsFunctionUtils::GetDim(dims_in, 1);
            auto hw_src   = DimsVectorUtils::Count(dims_in, 2);
            auto c_dst    = DimsFunctionUtils::GetDim(output->GetBlobDesc().dims, 1);
            auto hw_dst   = DimsVectorUtils::Count(output->GetBlobDesc().dims, 2);
            for (int b = 0; b < dims_in[0]; ++b) {
                UnpackC4(tmp_data, output_data_a + b * ROUND_UP(c_src, 4) * hw_src, hw_src, c_src);
                PackC4(output_data + b * ROUND_UP(c_dst, 4) * hw_dst, tmp_data, hw_dst, c_dst);
            }
        }
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output->GetBlobDesc().data_type);
        return TNNERR_LAYER_ERR;
    }

    return TNN_OK;
}

template <bool post_cal>
void ArmReduceLayerAcc::ReduceOneAxis(float *input_data, float *output_data, DimsVector &dims_in, int out_count,
                                      int axis) {
    int c4u  = ROUND_UP(dims_in[1], 4);
    int c4n  = UP_DIV(dims_in[1], 4);
    int c4r  = dims_in[1] % 4;
    int hw   = DimsVectorUtils::Count(dims_in, 2);
    int hw_c = hw / 4;
    int hw_r = hw % 4;
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
    } else {
        int outer_dim  = dims_in[0] * c4n * DimsVectorUtils::Count(dims_in, 2, axis);
        int reduce_dim = dims_in[axis];
        int inner_dim  = DimsVectorUtils::Count(dims_in, axis + 1);
        OMP_PARALLEL_FOR_
        for (int o = 0; o < outer_dim; ++o) {
            auto input_data_o  = input_data + o * reduce_dim * inner_dim * 4;
            auto output_data_o = output_data + o * inner_dim * 4;
            for (int i = 0; i < inner_dim; ++i) {
                auto input_data_i  = input_data_o + i * 4;
                auto output_data_i = output_data_o + i * 4;
                Float4 res         = op_->DataInit();
                for (int r = 0; r < reduce_dim; ++r) {
                    Float4 val = Float4::load(input_data_i + r * inner_dim * 4);
                    res        = op_->Calculate(res, val);
                }
                if (post_cal)
                    res = op_->PostCalculate(res, axis_n);
                Float4::save(output_data_i, res);
            }
        }
    }

    dims_in[axis] = 1;
}

}  // namespace TNN_NS
