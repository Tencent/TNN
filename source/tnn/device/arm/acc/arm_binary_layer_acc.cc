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

#include "tnn/device/arm/acc/arm_binary_layer_acc.h"

#include "tnn/device/arm/acc/compute/binary_function.h"
#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

template <>
float binary_op<ArmBinaryOpType::kADD, float>(const float &a, const float &b, float alpha, float beta) {
    return a + b;
}
template <>
bfp16_t binary_op<ArmBinaryOpType::kADD, bfp16_t>(const bfp16_t &a, const bfp16_t &b, float alpha, float beta) {
    return static_cast<bfp16_t>(static_cast<float>(a) + static_cast<float>(b));
}
template <>
float binary_op<ArmBinaryOpType::kSUB, float>(const float &a, const float &b, float alpha, float beta) {
    return a - b;
}
template <>
bfp16_t binary_op<ArmBinaryOpType::kSUB, bfp16_t>(const bfp16_t &a, const bfp16_t &b, float alpha, float beta) {
    return static_cast<bfp16_t>(static_cast<float>(a) - static_cast<float>(b));
}
template <>
float binary_op<ArmBinaryOpType::kMUL, float>(const float &a, const float &b, float alpha, float beta) {
    return a * b;
}
template <>
bfp16_t binary_op<ArmBinaryOpType::kMUL, bfp16_t>(const bfp16_t &a, const bfp16_t &b, float alpha, float beta) {
    return static_cast<bfp16_t>(static_cast<float>(a) * static_cast<float>(b));
}
template <>
float binary_op<ArmBinaryOpType::kDIV, float>(const float &a, const float &b, float alpha, float beta) {
    return a / b;
}
template <>
bfp16_t binary_op<ArmBinaryOpType::kDIV, bfp16_t>(const bfp16_t &a, const bfp16_t &b, float alpha, float beta) {
    return static_cast<bfp16_t>(static_cast<float>(a) / static_cast<float>(b));
}
template <>
float binary_op<ArmBinaryOpType::kMAX, float>(const float &a, const float &b, float alpha, float beta) {
    return a > b ? a : b;
}
template <>
bfp16_t binary_op<ArmBinaryOpType::kMAX, bfp16_t>(const bfp16_t &a, const bfp16_t &b, float alpha, float beta) {
    return static_cast<float>(a) > static_cast<float>(b) ? a : b;
}
template <>
float binary_op<ArmBinaryOpType::kMIN, float>(const float &a, const float &b, float alpha, float beta) {
    return a < b ? a : b;
}
template <>
bfp16_t binary_op<ArmBinaryOpType::kMIN, bfp16_t>(const bfp16_t &a, const bfp16_t &b, float alpha, float beta) {
    return static_cast<float>(a) < static_cast<float>(b) ? a : b;
}
template <>
float binary_op<ArmBinaryOpType::kHARDSWISH, float>(const float &a, const float &b, float alpha, float beta) {
    return a * MAX(MIN(b * alpha + beta, 1.0f), 0.f);
}
template <>
bfp16_t binary_op<ArmBinaryOpType::kHARDSWISH, bfp16_t>(const bfp16_t &a, const bfp16_t &b, float alpha, float beta) {
    return static_cast<bfp16_t>(static_cast<float>(a) * MAX(MIN(static_cast<float>(b) * alpha + beta, 1.0f), 0.f));
}
template <>
Float4 binary_op<ArmBinaryOpType::kADD, Float4>(const Float4 &a, const Float4 &b, float alpha, float beta) {
    return a + b;
}
template <>
Float4 binary_op<ArmBinaryOpType::kSUB, Float4>(const Float4 &a, const Float4 &b, float alpha, float beta) {
    return a - b;
}
template <>
Float4 binary_op<ArmBinaryOpType::kMUL, Float4>(const Float4 &a, const Float4 &b, float alpha, float beta) {
    return a * b;
}
template <>
Float4 binary_op<ArmBinaryOpType::kDIV, Float4>(const Float4 &a, const Float4 &b, float alpha, float beta) {
    return Float4::div(a, b);
}
template <>
Float4 binary_op<ArmBinaryOpType::kMAX, Float4>(const Float4 &a, const Float4 &b, float alpha, float beta) {
    return Float4::max(a, b);
}
template <>
Float4 binary_op<ArmBinaryOpType::kMIN, Float4>(const Float4 &a, const Float4 &b, float alpha, float beta) {
    return Float4::min(a, b);
}
template <>
Float4 binary_op<ArmBinaryOpType::kHARDSWISH, Float4>(const Float4 &a, const Float4 &b, float alpha, float beta) {
    return a * Float4::max(Float4::min(b * alpha + beta, 1.0f), 0.f);
}

Status ArmBinaryLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    desc_for_config_const_blob_ = outputs[0]->GetBlobDesc();
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        RETURN_ON_NEQ(allocateBufferParam(inputs, outputs), TNN_OK);
    }
#if TNN_ARM82
    else if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        RETURN_ON_NEQ(allocateBufferParamHalf(inputs, outputs), TNN_OK);
    }
#endif

    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(resource_);

    // prepare input shapes
    input_shapes_.clear();
    input_shapes_.reserve(4);
    auto output      = outputs[0];
    auto output_dims = output->GetBlobDesc().dims;

    if (broadcast_.GetBytesSize() > 0) {
        DimsVector input_shape0 = inputs[0]->GetBlobDesc().dims;
        if (layer_param->weight_input_index == 0) {
            // bias as another input
            input_shapes_.push_back(layer_res->element_shape);
            input_shapes_.push_back(input_shape0);
        } else {
            input_shapes_.push_back(input_shape0);
            input_shapes_.push_back(layer_res->element_shape);
        }
    } else {
        if (inputs.size() == 1) {
            input_shapes_.push_back(inputs[0]->GetBlobDesc().dims);
            input_shapes_.push_back(inputs[0]->GetBlobDesc().dims);
        } else {
            for (size_t inid = 0; inid < inputs.size(); inid++) {
                input_shapes_.push_back(inputs[inid]->GetBlobDesc().dims);
            }
        }
    }

    btype_ = BroadcastTypeUnknown;
    // check broadcast type is general or other optimized ncxhwx types
    // if type is general, go to nchw general impl
    DimsVector input_pad_shape;
    input_pad_shape.resize(output_dims.size());
    for (int i = 0; i < input_shapes_.size(); i++) {
        int pad_size = output_dims.size() - input_shapes_[i].size();
        PadShape(pad_size, output_dims.size(), input_pad_shape, input_shapes_[i]);
        BroadCastTypeFilter(output_dims, input_pad_shape, btype_);
        if (btype_ == BroadcastTypeGeneral) {
            break;
        }
    }

    return TNN_OK;
}

// if reshape, reset input_shapes and broadcast type
Status ArmBinaryLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(resource_);

    // prepare input shapes
    input_shapes_.clear();
    input_shapes_.reserve(4);
    auto output      = outputs[0];
    auto output_dims = output->GetBlobDesc().dims;

    if (broadcast_.GetBytesSize() > 0) {
        DimsVector input_shape0 = inputs[0]->GetBlobDesc().dims;
        if (layer_param->weight_input_index == 0) {
            // bias as another input
            input_shapes_.push_back(layer_res->element_shape);
            input_shapes_.push_back(input_shape0);
        } else {
            input_shapes_.push_back(input_shape0);
            input_shapes_.push_back(layer_res->element_shape);
        }
    } else {
        if (inputs.size() == 1) {
            input_shapes_.push_back(inputs[0]->GetBlobDesc().dims);
            input_shapes_.push_back(inputs[0]->GetBlobDesc().dims);
        } else {
            for (size_t inid = 0; inid < inputs.size(); inid++) {
                input_shapes_.push_back(inputs[inid]->GetBlobDesc().dims);
            }
        }
    }

    btype_ = BroadcastTypeUnknown;
    // check broadcast type is general or other optimized ncxhwx types
    // if type is general, go to nchw general impl
    DimsVector input_pad_shape;
    input_pad_shape.resize(output_dims.size());
    for (int i = 0; i < input_shapes_.size(); i++) {
        int pad_size = output_dims.size() - input_shapes_[i].size();
        PadShape(pad_size, output_dims.size(), input_pad_shape, input_shapes_[i]);
        BroadCastTypeFilter(output_dims, input_pad_shape, btype_);
        if (btype_ == BroadcastTypeGeneral) {
            break;
        }
    }

    return TNN_OK;
}

// SUPPORTED DATATYPES
bool ArmBinaryLayerAcc::DataTypeSupported(DataType data_type) {
    if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_HALF || data_type == DATA_TYPE_BFP16)
        return true;
    else
        return false;
}

Status ArmBinaryLayerAcc::ConfigBuffer2ArmBlobDesc(BlobDesc &desc) {
    DimsVector config_dims   = desc_for_config_const_blob_.dims;
    DimsVector original_dims = desc.dims;
    DimsVector pad_dims;
    if (config_dims.size() > 0) {
        pad_dims.resize(config_dims.size());
        int pad_size = config_dims.size() - original_dims.size();
        PadShape(pad_size, config_dims.size(), pad_dims, original_dims);
    } else {
        pad_dims = original_dims;
    }

    desc.dims        = pad_dims;
    desc.device_type = desc_for_config_const_blob_.device_type;
    desc.data_type   = desc_for_config_const_blob_.data_type;
    desc.data_format = desc_for_config_const_blob_.data_format;
    return TNN_OK;
}

ArmBinaryLayerAcc::~ArmBinaryLayerAcc() {}

Status ArmBinaryLayerAcc::allocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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

        if (element_handle.GetDataType() == DATA_TYPE_HALF)
            element_handle = ConvertHalfHandle(element_handle);

        auto layer_res_size = element_handle.GetDataCount();
        auto data_byte_size = DataTypeUtils::GetBytesSize(element_handle.GetDataType());
        auto layer_data     = element_handle.force_to<void *>();
        if (element_handle.GetDataType() == DATA_TYPE_FLOAT) {
            if (layer_res_size == 1) {
                // broadcast single, just memcpy
                RawBuffer temp(layer_res_size * data_byte_size);
                memcpy(temp.force_to<void *>(), layer_data, layer_res_size * data_byte_size);
                broadcast_ = temp;
            } else {
                // pack bias from nchw to nc4hw4
                int count = DimsVectorUtils::Count(dims_pad);
                if (dims_pad.size() >= 2) {
                    count = count / dims_pad[1];
                    count = count * ROUND_UP(dims_pad[1], 4);
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
                DataFormatConverter::ConvertFromNCHWToNCHW4Float(
                    static_cast<float *>(layer_data), temp.force_to<float *>(), dims_pad[0], channel, hw_stride, 1);
                broadcast_ = temp;
            }

            if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
                RawBuffer bfp16_temp(broadcast_.GetBytesSize() / 2);
                bfp16_temp.SetDataType(DATA_TYPE_BFP16);
                auto src = broadcast_.force_to<float *>();
                auto dst = bfp16_temp.force_to<bfp16_t *>();
                if (broadcast_.GetDataCount() == 1) {
                    dst[0] = src[0];
                } else {
                    FloatConvert(src, dst, broadcast_.GetDataCount() / 4);
                }
            }
        } else {
            // Todo
        }
    }

    return TNN_OK;
}

template <typename T, ArmBinaryOpType op_type>
Status ArmBinaryLayerAcc::Exec(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto output      = outputs[0];
    auto output_dims = output->GetBlobDesc().dims;

    if (btype_ == BroadcastTypeUnknown) {
        LOGE("Error: unknown broadcast type\n");
        return Status(TNNERR_LAYER_ERR, "Error: Binary layer's unknown broadcast type");
    } else if (btype_ == BroadcastTypeGeneral) {
        auto output_ptr    = GetBlobHandlePtr(output->GetHandle());
        size_t output_size = DimsVectorUtils::Count(output_dims);
        void *workspace    = context_->GetSharedWorkSpace(output_size * 2 * sizeof(T));
        BinaryGeneralFunc<T, op_type>(output_ptr, input_ptrs_, output_dims, input_shapes_, workspace, alpha_, beta_);
    } else {
        auto output_ptr = GetBlobHandlePtr(output->GetHandle());
        auto input0_ptr = input_ptrs_[0];
        auto input1_ptr = input_ptrs_[1];

        // input0_shape != output_shape && input1_shape != output_shape -> general impl
        if (!DimsVectorUtils::Equal(output_dims, input_shapes_[0]) &&
            !DimsVectorUtils::Equal(output_dims, input_shapes_[1])) {
            std::vector<DimsVector> shapes_tmp = {input_shapes_[0], input_shapes_[1]};
            std::vector<void *> ptrs_tmp       = {input0_ptr, input1_ptr};
            size_t output_size                 = DimsVectorUtils::Count(output_dims);
            void *workspace                    = context_->GetSharedWorkSpace(output_size * 2 * sizeof(T));
            BinaryGeneralFunc<T, op_type>(output_ptr, ptrs_tmp, output_dims, shapes_tmp, workspace, alpha_, beta_);
        } else {
            DimsVector input0_pad_shape, input1_pad_shape;
            input0_pad_shape.resize(output_dims.size());
            input1_pad_shape.resize(output_dims.size());
            PadShape(output_dims.size() - input_shapes_[0].size(), output_dims.size(), input0_pad_shape,
                     input_shapes_[0]);
            PadShape(output_dims.size() - input_shapes_[1].size(), output_dims.size(), input1_pad_shape,
                     input_shapes_[1]);

            BinaryFunc<T, op_type>(output_ptr, input0_ptr, input1_ptr, input0_pad_shape, input1_pad_shape, alpha_,
                                   beta_);
        }

        for (int i = 2; i < input_ptrs_.size(); i++) {
            auto input_ptr = input_ptrs_[i];
            DimsVector input0_pad_shape;
            PadShape(output_dims.size() - input_shapes_[i].size(), output_dims.size(), input0_pad_shape,
                     input_shapes_[i]);
            BinaryFunc<T, op_type>(output_ptr, output_ptr, input_ptr, output_dims, input0_pad_shape, alpha_, beta_);
        }
    }

    return TNN_OK;
}

Status ArmBinaryLayerAcc::ExecInt8(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status ArmBinaryLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(param_);
    // prepare input ptrs, since blob memory is allocted after init
    input_ptrs_.clear();
    input_ptrs_.reserve(4);
    if (broadcast_.GetBytesSize() > 0) {
        if (layer_param->weight_input_index == 0) {
            // bias as another input
            input_ptrs_.push_back(broadcast_.force_to<void *>());
            input_ptrs_.push_back(GetBlobHandlePtr(inputs[0]->GetHandle()));
        } else {
            input_ptrs_.push_back(GetBlobHandlePtr(inputs[0]->GetHandle()));
            input_ptrs_.push_back(broadcast_.force_to<void *>());
        }
    } else {
        if (inputs.size() == 1) {
            input_ptrs_.push_back(GetBlobHandlePtr(inputs[0]->GetHandle()));
            input_ptrs_.push_back(GetBlobHandlePtr(inputs[0]->GetHandle()));
        } else {
            for (size_t inid = 0; inid < inputs.size(); inid++) {
                input_ptrs_.push_back(GetBlobHandlePtr(inputs[inid]->GetHandle()));
            }
        }
    }

    auto data_type = outputs[0]->GetBlobDesc().data_type;
    if (data_type == DATA_TYPE_FLOAT) {
        // return Exec<float>(inputs, outputs);
        switch (op_type_) {
            case ArmBinaryOpType::kADD:
                return Exec<float, ArmBinaryOpType::kADD>(inputs, outputs);
            case ArmBinaryOpType::kSUB:
                return Exec<float, ArmBinaryOpType::kSUB>(inputs, outputs);
            case ArmBinaryOpType::kMUL:
                return Exec<float, ArmBinaryOpType::kMUL>(inputs, outputs);
            case ArmBinaryOpType::kDIV:
                return Exec<float, ArmBinaryOpType::kDIV>(inputs, outputs);
            case ArmBinaryOpType::kMAX:
                return Exec<float, ArmBinaryOpType::kMAX>(inputs, outputs);
            case ArmBinaryOpType::kMIN:
                return Exec<float, ArmBinaryOpType::kMIN>(inputs, outputs);
            case ArmBinaryOpType::kHARDSWISH:
                return Exec<float, ArmBinaryOpType::kHARDSWISH>(inputs, outputs);

            default:
                LOGE("Error, unknown binary op_type\n");
                return TNNERR_LAYER_ERR;
        }
    } else if (data_type == DATA_TYPE_BFP16) {
        switch (op_type_) {
            case ArmBinaryOpType::kADD:
                return Exec<bfp16_t, ArmBinaryOpType::kADD>(inputs, outputs);
            case ArmBinaryOpType::kSUB:
                return Exec<bfp16_t, ArmBinaryOpType::kSUB>(inputs, outputs);
            case ArmBinaryOpType::kMUL:
                return Exec<bfp16_t, ArmBinaryOpType::kMUL>(inputs, outputs);
            case ArmBinaryOpType::kDIV:
                return Exec<bfp16_t, ArmBinaryOpType::kDIV>(inputs, outputs);
            case ArmBinaryOpType::kMAX:
                return Exec<bfp16_t, ArmBinaryOpType::kMAX>(inputs, outputs);
            case ArmBinaryOpType::kMIN:
                return Exec<bfp16_t, ArmBinaryOpType::kMIN>(inputs, outputs);
            case ArmBinaryOpType::kHARDSWISH:
                return Exec<bfp16_t, ArmBinaryOpType::kHARDSWISH>(inputs, outputs);

            default:
                LOGE("Error, unknown binary op_type\n");
                return TNNERR_LAYER_ERR;
        }
    } else if (data_type == DATA_TYPE_INT8) {
        if (op_type_ == ArmBinaryOpType::kADD) {
            return ExecInt8(inputs, outputs);
        } else {
            LOGE("Error, int8 binary op only support add\n");
            return TNNERR_LAYER_ERR;
        }
    }
#if TNN_ARM82
    else if (data_type == DATA_TYPE_HALF) {
        switch (op_type_) {
            case ArmBinaryOpType::kADD:
                return ExecFp16<ArmBinaryOpType::kADD>(inputs, outputs);
            case ArmBinaryOpType::kSUB:
                return ExecFp16<ArmBinaryOpType::kSUB>(inputs, outputs);
            case ArmBinaryOpType::kMUL:
                return ExecFp16<ArmBinaryOpType::kMUL>(inputs, outputs);
            case ArmBinaryOpType::kDIV:
                return ExecFp16<ArmBinaryOpType::kDIV>(inputs, outputs);
            case ArmBinaryOpType::kMAX:
                return ExecFp16<ArmBinaryOpType::kMAX>(inputs, outputs);
            case ArmBinaryOpType::kMIN:
                return ExecFp16<ArmBinaryOpType::kMIN>(inputs, outputs);
                break;

            default:
                LOGE("Error, unknown binary op_type\n");
                return TNNERR_LAYER_ERR;
        }
    }
#endif
    else {
        return TNNERR_LAYER_ERR;
    }
}

}  // namespace TNN_NS
