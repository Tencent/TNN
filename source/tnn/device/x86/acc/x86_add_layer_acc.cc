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

#include "tnn/device/x86/x86_common.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/device/x86/acc/x86_add_layer_acc.h"
#include "tnn/device/x86/acc/compute/x86_compute_int8.h"

namespace TNN_NS {

X86AddLayerAcc::~X86AddLayerAcc() {}

Status X86AddLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                            const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(X86BinaryOpLayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    op_type_ = X86BinaryOpType::kADD;
    return allocateBufferParam(inputs, outputs);
}

Status X86AddLayerAcc::allocateBufferParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
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

Status X86AddLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(resource_);
    if (!((inputs.size() == 1 && layer_res) || inputs.size() >= 2)) {
        LOGE("Error: X86AddLayerAcc invalid inputs count\n");
        return Status(TNNERR_LAYER_ERR, "X86AddLayerAcc invalid inputs count");
    }

    if (outputs[0]->GetBlobDesc().data_type == DATA_TYPE_INT8) {
        auto dims = outputs[0]->GetBlobDesc().dims;
        // only support inputs.size() == 2
        if (inputs.size() > 2) {
            return Status(TNNERR_UNSUPPORT_NET, "INPUT > 2 NOT IMPLEMENT FOR INT8");
        }
        auto output_ptr   = handle_ptr<int8_t *>(outputs[0]->GetHandle());
        auto input0_ptr   = handle_ptr<int8_t *>(inputs[0]->GetHandle());
        auto input1_ptr   = handle_ptr<int8_t *>(inputs[1]->GetHandle());
        auto output_scale = output_int_scale_.force_to<float *>();
        auto input0_scale = input0_int_scale_.force_to<float *>();
        auto input1_scale = input1_int_scale_.force_to<float *>();
        X86MatrixAddInt8(output_ptr, input0_ptr, input1_ptr, output_scale, input0_scale, input1_scale,
                         ROUND_UP(dims[1], 4), DimsVectorUtils::Count(dims, 2));
    } else {
        return X86BinaryOpLayerAcc::DoForward(inputs, outputs);
    }

    return TNN_OK;
}

REGISTER_X86_ACC(Add, LAYER_ADD);

}   // namespace TNN_NS