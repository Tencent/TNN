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
#if TNN_ARM82
#include "tnn/device/arm/acc/arm_binary_layer_acc.h"
#include "tnn/device/arm/acc/compute/binary_function.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/device/arm/acc/Half8.h"

namespace TNN_NS {

template<> fp16_t binary_op<ArmBinaryOpType::kADD, fp16_t>(const fp16_t &a, const fp16_t &b, float alpha, float beta) {
    return a + b;
}
template<> fp16_t binary_op<ArmBinaryOpType::kSUB, fp16_t>(const fp16_t &a, const fp16_t &b, float alpha, float beta) {
    return a - b;
}
template<> fp16_t binary_op<ArmBinaryOpType::kMUL, fp16_t>(const fp16_t &a, const fp16_t &b, float alpha, float beta) {
    return a * b;
}
template<> fp16_t binary_op<ArmBinaryOpType::kDIV, fp16_t>(const fp16_t &a, const fp16_t &b, float alpha, float beta) {
    return a / b;
}
template<> fp16_t binary_op<ArmBinaryOpType::kMAX, fp16_t>(const fp16_t &a, const fp16_t &b, float alpha, float beta) {
    return a > b ? a : b;
}
template<> fp16_t binary_op<ArmBinaryOpType::kMIN, fp16_t>(const fp16_t &a, const fp16_t &b, float alpha, float beta) {
    return a < b ? a : b;
}

template<> Half8 binary_op<ArmBinaryOpType::kADD, Half8>(const Half8 &a, const Half8 &b, float alpha, float beta) {
    return a + b;
}
template<> Half8 binary_op<ArmBinaryOpType::kSUB, Half8>(const Half8 &a, const Half8 &b, float alpha, float beta) {
    return a - b;
}
template<> Half8 binary_op<ArmBinaryOpType::kMUL, Half8>(const Half8 &a, const Half8 &b, float alpha, float beta) {
    return a * b;
}
template<> Half8 binary_op<ArmBinaryOpType::kDIV, Half8>(const Half8 &a, const Half8 &b, float alpha, float beta) {
    return Half8::div(a, b);
}
template<> Half8 binary_op<ArmBinaryOpType::kMAX, Half8>(const Half8 &a, const Half8 &b, float alpha, float beta) {
    return Half8::max(a, b);
}
template<> Half8 binary_op<ArmBinaryOpType::kMIN, Half8>(const Half8 &a, const Half8 &b, float alpha, float beta) {
    return Half8::min(a, b);
}

template <ArmBinaryOpType op_type>
Status BinaryGeneralFp16Func(void *output_ptr, std::vector<void *> &input_ptrs, DimsVector output_shape,
                             std::vector<DimsVector> &input_shapes, void *workspace) {
    size_t output_size = DimsVectorUtils::Count(output_shape);
    fp16_t *output_nchw = reinterpret_cast<fp16_t *>(workspace);
    fp16_t *input_nchw = output_nchw + output_size;
    fp16_t *out_ptr = reinterpret_cast<fp16_t *>(output_ptr);

    DimsVector output_offset;
    BinaryComputeOffset(output_offset, output_shape, output_shape);
    for (int i = 0; i < input_shapes.size(); i++) {
        auto input_shape = input_shapes[i];
        fp16_t *input_data = reinterpret_cast<fp16_t *>(input_ptrs[i]);

        DimsVector input_shape_pad;
        input_shape_pad.resize(output_shape.size());
        PadShape(output_shape.size() - input_shape.size(), output_shape.size(), input_shape_pad, input_shape);

        int input_batch = input_shape_pad[0];
        int input_channel = input_shape_pad[1];
        int input_hw = DimsVectorUtils::Count(input_shape_pad, 2);
        // nc8hw8 to nchw
        UnpackHalfBlob(input_nchw, input_data, input_batch, input_channel, input_hw);

        DimsVector input_offset;
        BinaryComputeOffset(input_offset, input_shape, output_shape);
        if (i == 0) {
            BinaryComputeFirst<fp16_t>(input_offset, output_offset, output_shape, input_nchw, output_nchw);
        } else {
            BinaryCompute<fp16_t, op_type>(input_offset, output_offset, output_shape, input_nchw, output_nchw);
        }
    }

    int output_batch = output_shape[0];
    int output_channel = output_shape[1];
    int output_hw = DimsVectorUtils::Count(output_shape, 2);
    PackHalfBlob(out_ptr, output_nchw, output_batch, output_channel, output_hw);

    return TNN_OK;
}

Status ArmBinaryLayerAcc::allocateBufferParamHalf(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    auto layer_res = dynamic_cast<EltwiseLayerResource *>(resource_);

    if (layer_res && broadcast_.GetBytesSize() == 0) {
        RawBuffer element_handle = layer_res->element_handle;
        auto dims                = layer_res->element_shape;
        auto output_dims         = outputs[0]->GetBlobDesc().dims;
        DimsVector dims_pad;
        dims_pad.resize(output_dims.size());
        PadShape(output_dims.size() - dims.size(), output_dims.size(), dims_pad, dims);

        auto layer_res_size = element_handle.GetDataCount();
        auto data_byte_size = DataTypeUtils::GetBytesSize(DATA_TYPE_HALF);
        auto layer_data     = element_handle.force_to<void *>();
        if (element_handle.GetDataType() == DATA_TYPE_FLOAT ||
            element_handle.GetDataType() == DATA_TYPE_HALF) {

            fp16_t *filter_half_ptr = nullptr;
            RawBuffer filter_half(layer_res_size * data_byte_size);
            if (element_handle.GetDataType() == DATA_TYPE_HALF) {
                filter_half_ptr = reinterpret_cast<fp16_t *>(layer_data);
            } else {
                Float2Half(filter_half.force_to<fp16_t *>(), reinterpret_cast<const float *>(layer_data), layer_res_size);
                filter_half_ptr = filter_half.force_to<fp16_t *>();
            }

            if (layer_res_size == 1) {
                // broadcast single, just memcpy
                RawBuffer temp(8 * layer_res_size * data_byte_size);
                memcpy(temp.force_to<void *>(), filter_half_ptr, layer_res_size * data_byte_size);
                broadcast_ = temp;
            } else {
                // pack bias from nchw to nc8hw8
                int count = DimsVectorUtils::Count(dims_pad);
                if (dims_pad.size() >= 2) {
                    count = count / dims_pad[1];
                    count = count * ROUND_UP(dims_pad[1], 8);
                }
                int channel = 1;
                if (dims_pad.size() > 1) {
                    channel = dims_pad[1];
                }
                int hw_stride = 1;
                if (dims_pad.size() > 2) {
                    hw_stride = DimsVectorUtils::Count(dims_pad, 2);
                }

                RawBuffer temp(count * data_byte_size);
                for (int b = 0; b < dims_pad[0]; b++) {
                    fp16_t *src = filter_half_ptr + b * DimsVectorUtils::Count(dims_pad, 1);
                    fp16_t *dst = temp.force_to<fp16_t *>() + b * channel * hw_stride;
                    PackC8(dst, src, hw_stride, channel);
                }
                broadcast_ = temp;
            }
        }
    }

    return TNN_OK;
}

template <ArmBinaryOpType op_type>
Status ArmBinaryLayerAcc::ExecFp16(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto output = outputs[0];
    auto output_dims = output->GetBlobDesc().dims;

    if (btype_ == BroadcastTypeUnknown) {
        LOGE("Error: unknown broadcast type\n");
        return Status(TNNERR_LAYER_ERR, "Error: Binary layer's unknown broadcast type");
    } else if (btype_ == BroadcastTypeGeneral) {
        auto output_ptr = GetBlobHandlePtr(output->GetHandle());
        size_t output_size = DimsVectorUtils::Count(output_dims);
        void *workspace = context_->GetSharedWorkSpace(output_size * 2 * sizeof(fp16_t));
        BinaryGeneralFp16Func<op_type>(output_ptr, input_ptrs_, output_dims, input_shapes_, workspace);
    } else {
        auto output_ptr = GetBlobHandlePtr(output->GetHandle());
        auto input0_ptr = input_ptrs_[0];
        auto input1_ptr = input_ptrs_[1];

        DimsVector input0_pad_shape, input1_pad_shape;
        input0_pad_shape.resize(output_dims.size());
        input1_pad_shape.resize(output_dims.size());
        PadShape(output_dims.size() - input_shapes_[0].size(), output_dims.size(), input0_pad_shape, input_shapes_[0]);
        PadShape(output_dims.size() - input_shapes_[1].size(), output_dims.size(), input1_pad_shape, input_shapes_[1]);

        BinaryFunc<fp16_t, op_type, Half8, 8>(output_ptr, input0_ptr, input1_ptr, input0_pad_shape, input1_pad_shape);

        for (int i = 2; i < input_ptrs_.size(); i++) {
            auto input_ptr = input_ptrs_[i];
            PadShape(output_dims.size() - input_shapes_[i].size(), output_dims.size(), input0_pad_shape, input_shapes_[i]);
            BinaryFunc<fp16_t, op_type, Half8, 8>(output_ptr, output_ptr, input_ptr, output_dims, input0_pad_shape);
        }
    }

    return TNN_OK;
}
template Status ArmBinaryLayerAcc::ExecFp16<ArmBinaryOpType::kADD>(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
template Status ArmBinaryLayerAcc::ExecFp16<ArmBinaryOpType::kSUB>(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
template Status ArmBinaryLayerAcc::ExecFp16<ArmBinaryOpType::kMUL>(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
template Status ArmBinaryLayerAcc::ExecFp16<ArmBinaryOpType::kMAX>(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
template Status ArmBinaryLayerAcc::ExecFp16<ArmBinaryOpType::kMIN>(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);
template Status ArmBinaryLayerAcc::ExecFp16<ArmBinaryOpType::kDIV>(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs);

}  // namespace TNN_NS
#endif
