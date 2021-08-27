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

// author: sanerzheng@tencent.com

#include "tnn/device/arm/arm_util.h"
#include "tnn/train/grad/layer_grad.h"
#include "tnn/train/grad/utils.h"
#include "tnn/train/operations/op_builder.h"

namespace TNN_NS {
namespace train {
DECLARE_LAYER_GRAD(SoftMax, LAYER_SOFTMAX);
Status SoftMaxLayerGrad::OnGrad(const BaseLayer *layer, TrainContext &context) {
    auto inputs  = layer->input_blobs_;
    auto outputs = layer->output_blobs_;
    if (inputs.size() != 1 || outputs.size() != 1) {
        return Status(TNN_TRAIN_ERROR, "input size or output size not match in SoftMaxLayerGrad");
    }
    auto input0_desc      = inputs[0]->GetBlobDesc();
    auto output_desc      = outputs[0]->GetBlobDesc();
    auto input0_data_type = input0_desc.data_type;
    auto output_data_type = output_desc.data_type;
    auto input0_dims      = input0_desc.dims;
    auto output_dims      = output_desc.dims;
    if ((output_data_type != DATA_TYPE_BFP16 && output_data_type != DATA_TYPE_FLOAT) ||
        input0_data_type != output_data_type) {
        return Status(TNN_TRAIN_ERROR, "output datatype not match in SoftMaxLayerGrad");
    }
    if (!DimsVectorUtils::Equal(input0_dims, output_dims)) {
        return Status(TNN_TRAIN_ERROR, "output datatype not match in SoftMaxLayerGrad");
    }
    auto layer_param = dynamic_cast<SoftmaxLayerParam *>(layer->param_);
    if (layer_param == nullptr)
        return Status(TNN_TRAIN_ERROR, "SoftMax layer param axis error");
    auto iter_output_grad = context.backward_grads_blob.find(outputs[0]);
    if (iter_output_grad == context.backward_grads_blob.end()) {
        return Status(TNN_TRAIN_ERROR, "SoftMax layer output grad not found");
    }

    int axis    = layer_param->axis;
    axis        = static_cast<int>((axis + input0_dims.size()) % input0_dims.size());
    int batch   = DimsVectorUtils::Count(input0_dims, 0, axis);
    int channel = input0_dims[axis];
    int count   = DimsVectorUtils::Count(input0_dims, axis + 1);

    int total_count      = DimsVectorUtils::Count(input0_dims);
    int batch_for_pack   = input0_dims[0];
    int channel_for_pack = input0_dims[1];
    int hw_for_pack      = DimsVectorUtils::Count(input0_dims, 2);

    void *input_ptr =
        static_cast<void *>(static_cast<char *>(inputs[0]->GetHandle().base) + inputs[0]->GetHandle().bytes_offset);
    void *output_ptr =
        static_cast<void *>(static_cast<char *>(outputs[0]->GetHandle().base) + outputs[0]->GetHandle().bytes_offset);
    void *output_grad_ptr = iter_output_grad->second->force_to<void *>();

    RawBuffer input_buffer;
    ConvertToNCHW(input_ptr, input_buffer, input0_desc);

    RawBuffer output_buffer;
    ConvertToNCHW(output_ptr, output_buffer, output_desc);

    RawBuffer output_grad_buffer;
    ConvertToNCHW(output_grad_ptr, output_grad_buffer, iter_output_grad->second.get());

    // already init with 0.0
    // input_grad[i] = E(0<=j<N)(grad[j]*output[i](1-ouput[j])) when i = j
    // input_grad[i] = E(0<=j<N)(grad[j]*output[i]*-ouput[j]) when i != j
    std::shared_ptr<RawBuffer> input_grad =
        std::make_shared<RawBuffer>(total_count * DataTypeUtils::GetBytesSize(input0_data_type), input0_dims);
    if (input0_data_type == DATA_TYPE_FLOAT) {
        for (int n = 0; n < batch; n++) {
            float *output_batch      = static_cast<float *>(output_ptr) + n * channel * count;
            float *output_grad_batch = static_cast<float *>(output_grad_ptr) + n * channel * count;
            float *input_grad_batch  = input_grad->force_to<float *>() + n * channel * count;
            for (int i = 0; i < channel; ++i) {
                for (int k = 0; k < count; ++k) {
                    for (int j = 0; j < channel; ++j) {
                        if (i == j)
                            input_grad_batch[i * count + k] +=
                                output_grad_batch[j * count + k] * (1.0 - output_batch[j * count + k]);
                        else
                            input_grad_batch[i * count + k] -=
                                output_grad_batch[j * count + k] * output_batch[j * count + k];
                    }
                    input_grad_batch[i * count + k] *= output_batch[i * count + k];
                }
            }
        }
    } else /* TODO bfp16*/ {
        return Status(TNN_TRAIN_ERROR, "SoftMaxLayerGrad don't support bft16 for now");
    }
    ConvertToNC4HW4(input_grad, input0_desc);
    input_grad->SetDataType(input0_data_type);
    input_grad->SetDataFormat(input0_desc.data_format);
    UpdateGradValue(inputs[0], input_grad, context);
    return Status(TNN_OK);
}
REGISTER_LAYER_GRAD(SoftMax, LAYER_SOFTMAX);

} // namespace train
} // namespace TNN_NS