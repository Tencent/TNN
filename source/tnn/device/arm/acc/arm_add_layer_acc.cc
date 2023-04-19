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

#include "tnn/device/arm/acc/arm_add_layer_acc.h"
#include "tnn/utils/dims_function_utils.h"
#include "tnn/utils/omp_utils.h"

#ifdef TNN_USE_NEON
#include <arm_neon.h>
#endif

namespace TNN_NS {

ArmAddLayerAcc::~ArmAddLayerAcc() {}

Status ArmAddLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    op_type_ = ArmBinaryOpType::kADD;
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
        return allocateBufferParamInt8(inputs, outputs);
    } else {
        Status status = ArmBinaryLayerAcc::Init(context, param, resource, inputs, outputs);
        if (status != TNN_OK) {
            return status;
        }
        return TNN_OK;
    }
}

// SUPPORTED DATATYPES
bool ArmAddLayerAcc::DataTypeSupported(DataType data_type) {
    if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_HALF || data_type == DATA_TYPE_BFP16 ||
        data_type == DATA_TYPE_INT8 || data_type == DATA_TYPE_INT32)
        return true;
    else
        return false;
}

Status ArmAddLayerAcc::allocateBufferParamInt8(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    // alloc scale buffer, two input scales and output scale
    if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8 && !input0_int_scale_.GetBytesSize()) {
        auto dims_output    = outputs[0]->GetBlobDesc().dims;
        int total_byte_size = ROUND_UP(dims_output[1], 4) * sizeof(float);

        const float *i0_scale =
            reinterpret_cast<BlobInt8 *>(inputs[0])->GetIntResource()->scale_handle.force_to<float *>();

        const float *i1_scale =
            reinterpret_cast<BlobInt8 *>(inputs[1])->GetIntResource()->scale_handle.force_to<float *>();

        const float *o_scale =
            reinterpret_cast<BlobInt8 *>(outputs[0])->GetIntResource()->scale_handle.force_to<float *>();
        int scale_cnt = reinterpret_cast<BlobInt8 *>(inputs[0])->GetIntResource()->scale_handle.GetDataCount();
        RawBuffer temp_buffer0(total_byte_size);
        RawBuffer temp_buffer1(total_byte_size);
        RawBuffer temp_buffer2(total_byte_size);
        float *temp_ptr0 = temp_buffer0.force_to<float *>();
        float *temp_ptr1 = temp_buffer1.force_to<float *>();
        float *temp_ptr2 = temp_buffer2.force_to<float *>();
        for (int i = 0; i < dims_output[1]; i++) {
            int scale_idx = scale_cnt == 1 ? 0 : i;
            temp_ptr0[i]  = i0_scale[scale_idx];
            temp_ptr1[i]  = i1_scale[scale_idx];
            temp_ptr2[i]  = 1.0 / o_scale[scale_idx];
        }
        input0_int_scale_ = temp_buffer0;
        input1_int_scale_ = temp_buffer1;
        output_int_scale_ = temp_buffer2;
    }
    return TNN_OK;
}

Status ArmAddLayerAcc::ExecInt8(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto output = outputs[0];
    auto dims   = output->GetBlobDesc().dims;

    if (output->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        // only support inputs.size() == 2
        if (inputs.size() > 2) {
            return Status(TNNERR_UNSUPPORT_NET, "INPUT > 2 NOT IMPLEMENT FOR INT8");
        }
        auto output_ptr   = reinterpret_cast<int8_t *>(GetBlobHandlePtr(output->GetHandle()));
        auto input0_ptr   = reinterpret_cast<int8_t *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
        auto input1_ptr   = reinterpret_cast<int8_t *>(GetBlobHandlePtr(inputs[1]->GetHandle()));
        auto output_scale = output_int_scale_.force_to<float *>();
        auto input0_scale = input0_int_scale_.force_to<float *>();
        auto input1_scale = input1_int_scale_.force_to<float *>();
        MatrixAddInt8(output_ptr, input0_ptr, input1_ptr, output_scale, input0_scale, input1_scale,
                      ROUND_UP(dims[1], 4), DimsVectorUtils::Count(dims, 2));
    } else {
        LOGE("Error: layer acc dont support datatype: %d\n", output->GetBlobDesc().data_type);
        return TNNERR_LAYER_ERR;
    }

    return TNN_OK;
}

Status ArmAddLayerAcc::ExecInt32(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto output = outputs[0];

    if (output->GetBlobDesc().data_type != DATA_TYPE_INT32) {
        LOGE("Error: layer acc dont support datatype: %d\n", output->GetBlobDesc().data_type);
        return TNNERR_LAYER_ERR;
    }

    auto dims = output->GetBlobDesc().dims;
    int count = DimsFunctionUtils::GetDim(dims, 0) * ROUND_UP(DimsFunctionUtils::GetDim(dims, 1), 4) * DimsVectorUtils::Count(dims, 2);

    if (broadcast_.GetDataCount() != 4) {
        LOGE("Error: unsupported broadcast shape\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported int32 add layer's broadcast shape");
    }

    // only support inputs.size() == 1
    if (inputs.size() != 1) {
        return Status(TNNERR_UNSUPPORT_NET, "INPUT > 1 NOT IMPLEMENT FOR INT32");
    }

    if (!(DimsVectorUtils::Equal(inputs[0]->GetBlobDesc().dims, dims))) {
        LOGE("Error: mismatch input and output shape\n");
        return Status(TNNERR_LAYER_ERR, "Unsupported int32 mul layer's inputs shape");
    }

    auto output_ptr = reinterpret_cast<int32_t *>(GetBlobHandlePtr(output->GetHandle()));
    auto input0_ptr = reinterpret_cast<int32_t *>(GetBlobHandlePtr(inputs[0]->GetHandle()));
    auto input1_ptr = broadcast_.force_to<int32_t *>();

#ifdef TNN_USE_NEON
    int32x4_t b = vdupq_n_s32(input1_ptr[0]);
#else
    int32_t b = input1_ptr[0];
#endif

    OMP_PARALLEL_FOR_
    for (int i = 0; i < count; i += 4) {
#ifdef TNN_USE_NEON
        int32x4_t res = vaddq_s32(vld1q_s32(input0_ptr + i), b);
        vst1q_s32(output_ptr + i, res);
#else
        for (int j = 0; j < 4; ++j) {
            output_ptr[i+j] = input0_ptr[i+j] * b;
        }
#endif
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Add, LAYER_ADD)
REGISTER_ARM_PRECISION_FP16(LAYER_ADD)
REGISTER_ARM_LAYOUT(LAYER_ADD, DATA_FORMAT_NC4HW4)

}  // namespace TNN_NS
