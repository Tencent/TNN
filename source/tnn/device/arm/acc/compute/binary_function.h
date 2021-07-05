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

#ifndef TNN_ARM_COMPUTE_BINARY_FUNCTION_H_
#define TNN_ARM_COMPUTE_BINARY_FUNCTION_H_

#include "tnn/device/arm/acc/arm_binary_layer_acc.h"

namespace TNN_NS {

// alpha and beta used for hardswish
template<ArmBinaryOpType type, typename dtype>
dtype binary_op(const dtype &a, const dtype &b, float alpha = 0, float beta = 0) {
    return a;
}

void PadShape(const int pad_size, const int dim_size, DimsVector &pad_shape, DimsVector in_shape);

void BroadCastTypeFilter(const DimsVector &dims_output, const DimsVector &dims_input, BroadcastType &type);

void BroadCastInit(const DimsVector &dims, const DimsVector &dims0, const DimsVector &dims1,
                   BroadcastType &type, DimsVector &dims_broadcast, bool &swap_flag);
                   
void BinaryComputeOffset(DimsVector &offset, const DimsVector dims_in, const DimsVector dims_out);

template <typename T>
void BinaryComputeFirst(const DimsVector input_offset, const DimsVector output_offset,
                        const DimsVector output_shape, T* input_ptr, T* output_ptr) {
    DimsVector out_shape;
    DimsVector in_offset;
    DimsVector ou_offset;
    // support maximum 6 dimension, may be extended in furture
    out_shape.resize(6);
    in_offset.resize(6);
    ou_offset.resize(6);
    // if dim < 6, pad to 6
    int pad_size = 6 - output_shape.size();
    for (int i = 0; i < pad_size; i++) {
        out_shape[i] = 1;
        in_offset[i] = 0;
        ou_offset[i] = 0;
    }
    for (int i = pad_size; i < 6; i++) {
        out_shape[i] = output_shape[i - pad_size];
        in_offset[i] = input_offset[i - pad_size];
        ou_offset[i] = output_offset[i - pad_size];
    }

    for (int i0 = 0; i0 < out_shape[0]; i0++) {
        auto in_i0 = input_ptr + i0 * in_offset[0];
        auto ou_i0 = output_ptr + i0 * ou_offset[0];
        for (int i1 = 0; i1 < out_shape[1]; i1++) {
            auto in_i1 = in_i0 + i1 * in_offset[1];
            auto ou_i1 = ou_i0 + i1 * ou_offset[1];
            for (int i2 = 0; i2 < out_shape[2]; i2++) {
                auto in_i2 = in_i1 + i2 * in_offset[2];
                auto ou_i2 = ou_i1 + i2 * ou_offset[2];
                for (int i3 = 0; i3 < out_shape[3]; i3++) {
                    auto in_i3 = in_i2 + i3 * in_offset[3];
                    auto ou_i3 = ou_i2 + i3 * ou_offset[3];
                    for (int i4 = 0; i4 < out_shape[4]; i4++) {
                        auto in_i4 = in_i3 + i4 * in_offset[4];
                        auto ou_i4 = ou_i3 + i4 * ou_offset[4];
                        for (int i5 = 0; i5 < out_shape[5]; i5++) {
                            auto in_i5 = in_i4 + i5 * in_offset[5];
                            auto ou_i5 = ou_i4 + i5 * ou_offset[5];
                            ou_i5[0] = in_i5[0];
                        }
                    }
                }
            }
        }
    }
}

template <typename T, ArmBinaryOpType op_type>
void BinaryCompute(const DimsVector input_offset, const DimsVector output_offset,
                   const DimsVector output_shape, T* input_ptr, T* output_ptr,
                   float alpha = 0.f, float beta = 0.f) {
    DimsVector out_shape;
    DimsVector in_offset;
    DimsVector ou_offset;
    // support maximum 6 dimension, may be extended in furture
    out_shape.resize(6);
    in_offset.resize(6);
    ou_offset.resize(6);
    // if dim < 6, pad to 6
    int pad_size = 6 - output_shape.size();
    for (int i = 0; i < pad_size; i++) {
        out_shape[i] = 1;
        in_offset[i] = 0;
        ou_offset[i] = 0;
    }
    for (int i = pad_size; i < 6; i++) {
        out_shape[i] = output_shape[i - pad_size];
        in_offset[i] = input_offset[i - pad_size];
        ou_offset[i] = output_offset[i - pad_size];
    }

    for (int i0 = 0; i0 < out_shape[0]; i0++) {
        auto in_i0 = input_ptr + i0 * in_offset[0];
        auto ou_i0 = output_ptr + i0 * ou_offset[0];
        for (int i1 = 0; i1 < out_shape[1]; i1++) {
            auto in_i1 = in_i0 + i1 * in_offset[1];
            auto ou_i1 = ou_i0 + i1 * ou_offset[1];
            for (int i2 = 0; i2 < out_shape[2]; i2++) {
                auto in_i2 = in_i1 + i2 * in_offset[2];
                auto ou_i2 = ou_i1 + i2 * ou_offset[2];
                for (int i3 = 0; i3 < out_shape[3]; i3++) {
                    auto in_i3 = in_i2 + i3 * in_offset[3];
                    auto ou_i3 = ou_i2 + i3 * ou_offset[3];
                    for (int i4 = 0; i4 < out_shape[4]; i4++) {
                        auto in_i4 = in_i3 + i4 * in_offset[4];
                        auto ou_i4 = ou_i3 + i4 * ou_offset[4];
                        for (int i5 = 0; i5 < out_shape[5]; i5++) {
                            auto in_i5 = in_i4 + i5 * in_offset[5];
                            auto ou_i5 = ou_i4 + i5 * ou_offset[5];
                            ou_i5[0] = binary_op<op_type, T>(ou_i5[0], in_i5[0], alpha, beta);
                        }
                    }
                }
            }
        }
    }
}

template <typename T, ArmBinaryOpType op_type>
Status BinaryGeneralFunc(void *output_ptr, std::vector<void*> &input_ptrs, DimsVector output_shape,
                         std::vector<DimsVector> &input_shapes, void *workspace,
                         float alpha = 0.f, float beta = 0.f) {
    size_t output_size = DimsVectorUtils::Count(output_shape);
    T *output_nchw = reinterpret_cast<T *>(workspace);
    T *input_nchw = output_nchw + output_size;
    T *ou_ptr = reinterpret_cast<T *>(output_ptr);

    DimsVector output_offset;
    BinaryComputeOffset(output_offset, output_shape, output_shape);
    for (int i = 0; i < input_shapes.size(); i++) {
        auto input_shape = input_shapes[i];
        T *input_data = reinterpret_cast<T *>(input_ptrs[i]);

        DimsVector input_shape_pad;
        input_shape_pad.resize(output_shape.size());
        PadShape(output_shape.size() - input_shape.size(), output_shape.size(), input_shape_pad, input_shape);

        int input_batch = input_shape_pad[0];
        int input_channel = input_shape_pad[1];
        int input_hw = DimsVectorUtils::Count(input_shape_pad, 2);
        // nc4hw4 to nchw
        UnpackFloatBlob(input_nchw, input_data, input_batch, input_channel, input_hw);

        DimsVector input_offset;
        BinaryComputeOffset(input_offset, input_shape, output_shape);
        if (i == 0) {
            BinaryComputeFirst<T>(input_offset, output_offset, output_shape, input_nchw, output_nchw);
        } else {
            BinaryCompute<T, op_type>(input_offset, output_offset, output_shape, input_nchw, output_nchw, alpha, beta);
        }
    }

    int output_batch = output_shape[0];
    int output_channel = output_shape[1];
    int output_hw = DimsVectorUtils::Count(output_shape, 2);
    PackFloatBlob(ou_ptr, output_nchw, output_batch, output_channel, output_hw);

    return TNN_OK;
}

/*
Binary func with different opreator,
set dims0 full shape, dims1 broadcast shape, so we need to swap input ptrs
*/
template <typename T, ArmBinaryOpType op_type, typename VEC = Float4, int pack = 4>
Status BinaryFunc(void *out_ptr, void *input0_ptr, void *input1_ptr, DimsVector &dims0,
                  DimsVector &dims1, float alpha = 0.f, float beta = 0.f) {
    DimsVector dims = DimsVectorUtils::Max(dims0, dims1);
    DimsVector dims_broadcast;
    BroadcastType type = BroadcastTypeUnknown;
    auto _input0       = reinterpret_cast<T *>(input0_ptr);
    auto _input1       = reinterpret_cast<T *>(input1_ptr);
    auto output_ptr    = reinterpret_cast<T *>(out_ptr);
    bool swap_flag     = false;

    BroadCastInit(dims, dims0, dims1, type, dims_broadcast, swap_flag);

    if (swap_flag) {
        std::swap(_input0, _input1);
    }

    if (dims_broadcast.size()) {
        type = (dims_broadcast[1] == 1) ? BroadcastTypeSingle : BroadcastTypeChannel;
    }

    int count = DimsVectorUtils::Count(dims);
    if (dims.size() >= 2) {
        count = count / dims[1];
        count = count * ROUND_UP(dims[1], pack);
    }
    int count_quad = UP_DIV(count, pack);

    int hw_stride = 1;
    if (dims.size() > 2) {
        hw_stride = DimsVectorUtils::Count(dims, 2);
    }
    int w_stride = 1;
    if (dims.size() > 3) {
        w_stride = DimsVectorUtils::Count(dims, 3);
    }

    if (type == BroadcastTypeNormal) {
        for (int n = 0; n < count_quad; n++) {
            auto v1 = VEC::load(_input0 + n * pack);
            auto v2 = VEC::load(_input1 + n * pack);
            VEC::save(output_ptr + n * pack, binary_op<op_type, VEC>(v1, v2, alpha, beta));
        }

        return TNN_OK;
    }

    if (swap_flag) {
        if (type == BroadcastTypeSingle) {
            // broadcast single
            for (int n = 0; n < count_quad; n++) {
                auto v1 = VEC::load(_input0 + n * pack);
                auto v2 = VEC(_input1[0]);
                VEC::save(output_ptr + n * pack, binary_op<op_type, VEC>(v2, v1, alpha, beta));
            }
        } else if (type == BroadcastTypeChannel) {
            // broadcast channel
            for (int n = 0; n < count_quad; n++) {
                int b               = n / (hw_stride * UP_DIV(dims[1], pack));
                int channel_4_index = n / (hw_stride) - b * UP_DIV(dims[1], pack);
                auto v1             = VEC::load(_input0 + n * pack);
                auto v2             = VEC::load(_input1 + channel_4_index * pack);
                VEC::save(output_ptr + n * pack, binary_op<op_type, VEC>(v2, v1, alpha, beta));
            }
        } else if (type == BroadcastTypeElement) {
            // broadcast chw
            for (int n = 0; n < count_quad; n++) {
                int channel_4_index = n % (hw_stride * UP_DIV(dims[1], pack));
                auto v1             = VEC::load(_input0 + n * pack);
                auto v2             = VEC::load(_input1 + channel_4_index * pack);
                VEC::save(output_ptr + n * pack, binary_op<op_type, VEC>(v2, v1, alpha, beta));
            }
        } else if (type == BroadcastTypeHeightWidth) {
            // broadcast hw
            for (int n = 0; n < count_quad; n++) {
                int hw_index = n % (hw_stride);
                auto v1      = VEC::load(_input0 + n * pack);
                auto v2      = VEC(_input1[hw_index * pack]);
                VEC::save(output_ptr + n * pack, binary_op<op_type, VEC>(v2, v1, alpha, beta));
            }
        } else if (type == BroadcastTypeWidth) {
            // broadcast w
            for (int n = 0; n < count_quad; n++) {
                int w_index = n % (w_stride);
                auto v1      = VEC::load(_input0 + n * pack);
                auto v2      = VEC(_input1[w_index * pack]);
                VEC::save(output_ptr + n * pack, binary_op<op_type, VEC>(v2, v1, alpha, beta));
            }
        } else {
            LOGE("Error: invalid add type\n");
            return Status(TNNERR_LAYER_ERR, "Error: Binary layer's unsupported broadcast type");
        }
    } else {
        if (type == BroadcastTypeSingle) {
            // broadcast single
            for (int n = 0; n < count_quad; n++) {
                auto v1 = VEC::load(_input0 + n * pack);
                auto v2 = VEC(_input1[0]);
                VEC::save(output_ptr + n * pack, binary_op<op_type, VEC>(v1, v2, alpha, beta));
            }
        } else if (type == BroadcastTypeChannel) {
            // broadcast channel
            for (int n = 0; n < count_quad; n++) {
                int b               = n / (hw_stride * UP_DIV(dims[1], pack));
                int channel_4_index = n / (hw_stride) - b * UP_DIV(dims[1], pack);
                auto v1             = VEC::load(_input0 + n * pack);
                auto v2             = VEC::load(_input1 + channel_4_index * pack);
                VEC::save(output_ptr + n * pack, binary_op<op_type, VEC>(v1, v2, alpha, beta));
            }
        } else if (type == BroadcastTypeElement) {
            // broadcast chw
            for (int n = 0; n < count_quad; n++) {
                int channel_4_index = n % (hw_stride * UP_DIV(dims[1], pack));
                auto v1             = VEC::load(_input0 + n * pack);
                auto v2             = VEC::load(_input1 + channel_4_index * pack);
                VEC::save(output_ptr + n * pack, binary_op<op_type, VEC>(v1, v2, alpha, beta));
            }
        } else if (type == BroadcastTypeHeightWidth) {
            // broadcast hw
            for (int n = 0; n < count_quad; n++) {
                int hw_index = n % (hw_stride);
                auto v1      = VEC::load(_input0 + n * pack);
                auto v2      = VEC(_input1[hw_index * pack]);
                VEC::save(output_ptr + n * pack, binary_op<op_type, VEC>(v1, v2, alpha, beta));
            }
        } else if (type == BroadcastTypeWidth) {
            // broadcast w
            for (int n = 0; n < count_quad; n++) {
                int w_index = n % (w_stride);
                auto v1      = VEC::load(_input0 + n * pack);
                auto v2      = VEC(_input1[w_index * pack]);
                VEC::save(output_ptr + n * pack, binary_op<op_type, VEC>(v1, v2, alpha, beta));
            }
        } else {
            LOGE("Error: invalid add type\n");
            return Status(TNNERR_LAYER_ERR, "Error: Binary layer's unsupported broadcast type");
        }
    }

    return TNN_OK;
}

}  // namespace TNN_NS

#endif
