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

#include "tnn/device/arm/arm_common.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/interpreter/raw_buffer.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

template <typename T>
static void _operator_add(T *output_ptr, T *input0_ptr, T *input1_ptr, DimsVector &dims0, DimsVector &dims1) {
    DimsVector dims = DimsVectorUtils::Max(dims0, dims1);
    AddOpType type = ADD_ELEMENT;
    auto _input0   = input0_ptr;
    auto _input1   = input1_ptr;
    OperatorAddPreparation();

    int count      = ROUND_UP(dims[1], 4) * dims[2] * dims[3];
    int count_quad = UP_DIV(count, 4);

    if (type == ADD_SINGLE) {
        // broadcast single
        count_quad *= dims[0];
        for (int n = 0; n < count_quad; n++) {
            Float4::save(output_ptr + n * 4, Float4::load(_input0 + n * 4) + Float4(_input1[0]));
        }
    } else if (type == ADD_ELEMENT) {
        // no broadcast
        if (dims0[0] == dims1[0] && dims0[1] == dims1[1]) {
            count_quad *= dims[0];
            for (int n = 0; n < count_quad; n++) {
                Float4::save(output_ptr + n * 4, Float4::load(_input0 + n * 4) + Float4::load(_input1 + n * 4));
            }
        } else if (dims0[1] == dims1[1]) {
            // broadcast chw
            for (int batch = 0; batch < dims[0]; batch++) {
                auto input0_batch_ = _input0 + count * batch;
                auto output_batch_ = output_ptr + count * batch;
                for (int n = 0; n < count_quad; n++) {
                    Float4::save(output_batch_ + n * 4,
                                 Float4::load(input0_batch_ + n * 4) + Float4::load(_input1 + n * 4));
                }
            }
        } else {
            // broadcast hw
            for (int batch = 0; batch < dims[0]; batch++) {
                auto input0_batch_ = _input0 + count * batch;
                auto output_batch_ = output_ptr + count * batch;
                for (int n = 0; n < count_quad; n++) {
                    auto hw_index = n % (dims[2] * dims[3]);
                    Float4::save(output_batch_ + n * 4,
                                 Float4::load(input0_batch_ + n * 4) + Float4(_input1[hw_index * 4]));
                }
            }
        }
    } else if (type == ADD_CHANNEL) {
        // broadcast channel
        count_quad *= dims[0];
        for (int n = 0; n < count_quad; n++) {
            int b               = n / (dims[2] * dims[3] * UP_DIV(dims[1], 4));
            int channel_4_index = n / (dims[2] * dims[3]) - b * UP_DIV(dims[1], 4);
            Float4::save(output_ptr + n * 4,
                         Float4::load(_input0 + n * 4) + Float4::load(_input1 + channel_4_index * 4));
        }
    } else {
        LOGE("Error: invalid add type\n");
    }
}

#if TNN_ARM82
extern void _operator_add_fp16(fp16_t *output_ptr, fp16_t *input0_ptr, fp16_t *input1_ptr, DimsVector &dims0,
                               DimsVector &dims1);
#endif

ArmAddLayerAcc::~ArmAddLayerAcc() {}

Status ArmAddLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(ArmLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    return allocateBufferParam(inputs, outputs);
}

Status ArmAddLayerAcc::allocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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

    if (!output_bias_.GetBytesSize()) {
        auto layer_res = dynamic_cast<EltwiseLayerResource *>(resource_);
        if (layer_res) {
            bias_shape_         = {1, 1, 1, 1};
            auto dims           = outputs[0]->GetBlobDesc().dims;
            auto layer_res_size = layer_res->element_handle.GetDataCount();
            if (layer_res_size == 1) {
                // broadcast sigle
                bias_shape_[1] = 1;
            } else if (layer_res_size == dims[1]) {
                // broadcast channel
                bias_shape_[1] = dims[1];
            } else if (layer_res_size == dims[2] * dims[3]) {
                // broad cast hw
                bias_shape_[2] = dims[2];
                bias_shape_[3] = dims[3];
            } else if (layer_res_size == dims[1] * dims[2] * dims[3]) {
                // broad cast chw
                bias_shape_[1] = dims[1];
                bias_shape_[2] = dims[2];
                bias_shape_[3] = dims[3];
            } else {
                return Status(TNNERR_MODEL_ERR, "Error: unsupported broadcast type");
            }
#if TNN_ARM82
            if (inputs[0]->GetBlobDesc().data_type == DATA_TYPE_HALF) {
                auto buffer_size = ROUND_UP(bias_shape_[1], 8) * bias_shape_[2] * bias_shape_[3] * sizeof(fp16_t);
                RawBuffer temp(buffer_size);
                // pack bias from nchw to nc8hw8
                fp16_t *b_dst            = temp.force_to<fp16_t *>();
                RawBuffer element_handle = layer_res->element_handle;
                if (element_handle.GetDataType() == DATA_TYPE_HALF)
                    element_handle = ConvertHalfHandle(element_handle);
                float *b_src = element_handle.force_to<float *>();
                auto hw     = bias_shape_[2] * bias_shape_[3];
                memset(b_dst, 0, buffer_size);
                for (int c = 0; c < bias_shape_[1]; c++) {
                    int ci = c % 8;
                    int co = c / 8;
                    for (int cur_hw = 0; cur_hw < hw; cur_hw++) {
                        b_dst[co * 8 * hw + cur_hw * 8 + ci] = b_src[c * hw + cur_hw];
                    }
                }
                output_bias_ = temp;
                return TNN_OK;
            }
#endif
            auto buffer_size = ROUND_UP(bias_shape_[1], 4) * bias_shape_[2] * bias_shape_[3] * sizeof(float);
            RawBuffer temp(buffer_size);
            // pack bias from nchw to nc4hw4
            auto *b_dst              = temp.force_to<float *>();
            RawBuffer element_handle = layer_res->element_handle;
            if (element_handle.GetDataType() == DATA_TYPE_HALF)
                element_handle = ConvertHalfHandle(element_handle);
            auto *b_src = element_handle.force_to<float *>();
            auto hw     = bias_shape_[2] * bias_shape_[3];
            memset(b_dst, 0, buffer_size);
            for (int c = 0; c < bias_shape_[1]; c++) {
                int ci = c % 4;
                int co = c / 4;
                for (int cur_hw = 0; cur_hw < hw; cur_hw++) {
                    b_dst[co * 4 * hw + cur_hw * 4 + ci] = b_src[c * hw + cur_hw];
                }
            }
            output_bias_ = temp;
        }
    }
    return TNN_OK;
}

Status ArmAddLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(resource_);
    if (!((inputs.size() == 1 && layer_res) || inputs.size() >= 2)) {
        LOGE("Error: ArmAddLayerAcc invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "ArmAddLayerAcc invalid inputs count");
    }

    std::vector<void *> input_ptrs;
    std::vector<DimsVector> input_shapes;
    input_ptrs.reserve(4);
    input_shapes.reserve(4);

    AddOpType type;
    auto output = outputs[0];
    auto dims   = output->GetBlobDesc().dims;

    if (inputs.size() == 1) {
        input_ptrs.push_back(GetBlobHandlePtr(inputs[0]->GetHandle()));
        input_shapes.push_back(dims);
        // bias is another input
        input_ptrs.push_back(output_bias_.force_to<void *>());
        input_shapes.push_back(bias_shape_);
    } else {
        // type = ADD_ELEMENT;
        for (size_t inid = 0; inid < inputs.size(); inid++) {
            input_ptrs.push_back(GetBlobHandlePtr(inputs[inid]->GetHandle()));
            input_shapes.push_back(inputs[inid]->GetBlobDesc().dims);
        }
    }

    if (input_ptrs.size() < 2) {
        LOGE("Error: invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "Add layer's inputs size must >= 2");
    }

    if (output->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        // int count       = dims[0] * ROUND_UP(dims[1], 4) * dims[2] * dims[3];
        // int count_quad  = UP_DIV(count, 4);
        auto output_ptr = reinterpret_cast<float *>(GetBlobHandlePtr(output->GetHandle()));
        auto input0_ptr = reinterpret_cast<float *>(input_ptrs[0]);
        auto input1_ptr = reinterpret_cast<float *>(input_ptrs[1]);

        _operator_add<float>(output_ptr, input0_ptr, input1_ptr, input_shapes[0], input_shapes[1]);

        for (int i = 2; i < input_ptrs.size(); i++) {
            auto input_ptr = reinterpret_cast<float *>(input_ptrs[i]);
            _operator_add(output_ptr, output_ptr, input_ptr, dims, input_shapes[i]);
        }
    } else if (output->GetBlobDesc().data_type == DATA_TYPE_INT8) {
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
                      ROUND_UP(dims[1], 4), dims[2], dims[3]);
    } else if (output->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
        auto output_ptr = reinterpret_cast<bfp16_t *>(GetBlobHandlePtr(output->GetHandle()));
        auto input0_ptr = reinterpret_cast<bfp16_t *>(input_ptrs[0]);
        auto input1_ptr = reinterpret_cast<bfp16_t *>(input_ptrs[1]);

        _operator_add<bfp16_t>(output_ptr, input0_ptr, input1_ptr, input_shapes[0], input_shapes[1]);

        for (int i = 2; i < input_ptrs.size(); i++) {
            auto input_ptr = reinterpret_cast<bfp16_t *>(input_ptrs[i]);
            _operator_add(output_ptr, output_ptr, input_ptr, dims, input_shapes[i]);
        }
    }
#if TNN_ARM82
    else if (output->GetBlobDesc().data_type == DATA_TYPE_HALF) {
        auto output_ptr = reinterpret_cast<fp16_t *>(GetBlobHandlePtr(output->GetHandle()));
        auto input0_ptr = reinterpret_cast<fp16_t *>(input_ptrs[0]);
        auto input1_ptr = reinterpret_cast<fp16_t *>(input_ptrs[1]);

        _operator_add_fp16(output_ptr, input0_ptr, input1_ptr, input_shapes[0], input_shapes[1]);

        for (int i = 2; i < input_ptrs.size(); i++) {
            auto input_ptr = reinterpret_cast<fp16_t *>(input_ptrs[i]);
            _operator_add_fp16(output_ptr, output_ptr, input_ptr, dims, input_shapes[i]);
        }
    }
#endif
    else {
        LOGE("Error: layer acc dont support datatype: %d\n", output->GetBlobDesc().data_type);
        return TNNERR_LAYER_ERR;
    }

    return TNN_OK;
}

REGISTER_ARM_ACC(Add, LAYER_ADD)
REGISTER_ARM_PRECISION_FP16(LAYER_ADD)

}  // namespace TNN_NS
