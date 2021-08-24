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

#include "tnn/train/grad/layer_grad.h"
#include "tnn/train/operations/op_builder.h"
#include "tnn/train/grad/utils.h"
#include "tnn/device/arm/arm_util.h"

namespace TNN_NS {
namespace train {
DECLARE_LAYER_GRAD(SoftMax, LAYER_SOFTMAX);
Status SoftMaxLayerGrad::OnGrad(const BaseLayer* layer, TrainContext& context){
    auto inputs = layer->input_blobs_;
    auto outputs = layer->output_blobs_;
    if(inputs.size() != 1 || outputs.size() != 1) {
        return Status(TNN_TRAIN_ERROR, "input size or output size not match in SoftMaxLayerGrad");
    }
    auto input0_desc = inputs[0]->GetBlobDesc();
    auto output_desc = outputs[0]->GetBlobDesc();
    auto input0_data_type = input0_desc.data_type;
    auto output_data_type = output_desc.data_type;
    auto input0_dims = input0_desc.dims;
    auto output_dims = output_desc.dims;
    if((output_data_type != DATA_TYPE_BFP16 && output_data_type != DATA_TYPE_FLOAT) || input0_data_type != output_data_type) {
       return Status(TNN_TRAIN_ERROR, "output datatype not match in SoftMaxLayerGrad"); 
    }
    if(!DimsVectorUtils::Equal(input0_dims, output_dims)) {
        return Status(TNN_TRAIN_ERROR, "output datatype not match in SoftMaxLayerGrad"); 
    }
    auto layer_param = dynamic_cast<SoftmaxLayerParam *>(layer->param_);
    if(layer_param == nullptr)
        return Status(TNN_TRAIN_ERROR, "SoftMax layer param axis error");
    auto iter_output_grad = context.backward_grads_blob.find(outputs[0]);
    if(iter_output_grad == context.backward_grads_blob.end()) {
        return Status(TNN_TRAIN_ERROR, "SoftMax layer output grad not found");
    }
        
    int axis           = layer_param->axis;
    axis               = static_cast<int>((axis + input0_dims.size()) % input0_dims.size());
    int batch          = DimsVectorUtils::Count(input0_dims, 0, axis);
    int channel        = input0_dims[axis];
    int count          = DimsVectorUtils::Count(input0_dims, axis + 1);

    int total_count = DimsVectorUtils::Count(input0_dims);
    int batch_for_pack = input0_dims[0];
    int channel_for_pack = input0_dims[1];
    int hw_for_pack = DimsVectorUtils::Count(input0_dims, 2);

    void* input_ptr = static_cast<void*>(static_cast<char*>(inputs[0]->GetHandle().base) + inputs[0]->GetHandle().bytes_offset);
    void* output_ptr = static_cast<void*>(static_cast<char*>(outputs[0]->GetHandle().base) + outputs[0]->GetHandle().bytes_offset);
    void* output_grad_ptr = iter_output_grad->second->force_to<void*>();

    // TODO: abstract three code blocks below to an util function
    RawBuffer input_buffer; 
    if(input0_desc.data_format == DATA_FORMAT_NC4HW4) {
        input_buffer = RawBuffer(total_count* DataTypeUtils::GetBytesSize(input0_data_type));
        if(input0_data_type== DATA_TYPE_BFP16) {
            UnpackFloatBlob(input_buffer.force_to<bfp16_t*>(), static_cast<bfp16_t*>(input_ptr), batch_for_pack, channel_for_pack, hw_for_pack);
            input_ptr = input_buffer.force_to<void*>();
        }
        else if(input0_data_type == DATA_TYPE_FLOAT) {
            UnpackFloatBlob(input_buffer.force_to<float*>(), static_cast<float*>(input_ptr), batch_for_pack, channel_for_pack, hw_for_pack);    
            input_ptr = input_buffer.force_to<void*>();
        }
    }

    RawBuffer output_buffer;
    if(output_desc.data_format == DATA_FORMAT_NC4HW4) {
        output_buffer = RawBuffer(total_count* DataTypeUtils::GetBytesSize(output_data_type));
        if(output_data_type == DATA_TYPE_BFP16) {
            UnpackFloatBlob(output_buffer.force_to<bfp16_t*>(), static_cast<bfp16_t*>(output_ptr), batch_for_pack, channel_for_pack, hw_for_pack);
            output_ptr = output_buffer.force_to<void*>();
        }
        else if(output_data_type == DATA_TYPE_FLOAT) {
            UnpackFloatBlob(output_buffer.force_to<float*>(), static_cast<float*>(output_ptr), batch_for_pack, channel_for_pack, hw_for_pack);    
            output_ptr = output_buffer.force_to<void*>();
        }
    }

    RawBuffer output_grad_buffer;
    if(iter_output_grad->second->GetDataFormat() == DATA_FORMAT_NC4HW4) {
        output_grad_buffer = RawBuffer(total_count* DataTypeUtils::GetBytesSize(output_data_type));
        if(output_data_type == DATA_TYPE_BFP16) {
            UnpackFloatBlob(output_grad_buffer.force_to<bfp16_t*>(), static_cast<bfp16_t*>(output_grad_ptr), batch_for_pack, channel_for_pack, hw_for_pack);
            output_grad_ptr = output_grad_buffer.force_to<void*>();
        }
        else if(output_data_type == DATA_TYPE_FLOAT) {
            UnpackFloatBlob(output_grad_buffer.force_to<float*>(), static_cast<float*>(output_grad_ptr), batch_for_pack, channel_for_pack, hw_for_pack);    
            output_grad_ptr = output_grad_buffer.force_to<void*>();
        }
    }
    //already init with 0.0
    //input_grad[i] = E(0<=j<N)(grad[j]*output[i](1-ouput[j])) when i = j
    //input_grad[i] = E(0<=j<N)(grad[j]*output[i]*-ouput[j]) when i != j
    std::shared_ptr<RawBuffer> input_grad = std::make_shared<RawBuffer>(total_count * DataTypeUtils::GetBytesSize(input0_data_type), input0_dims);
    if(input0_data_type == DATA_TYPE_FLOAT) {
        for (int n = 0; n < batch; n++) {
            float* output_batch = static_cast<float*>(output_ptr) + n * channel * count;
            float* output_grad_batch = static_cast<float*>(output_grad_ptr) + n * channel * count;
            float* input_grad_batch = input_grad->force_to<float*>() + n * channel * count;
            for (int i = 0; i < channel; ++i) {
                for(int k = 0; k < count; ++k) {
                    for(int j=0; j<channel; ++j) {
                        if(i == j)
                            input_grad_batch[i*count + k] += output_grad_batch[j*count + k] * (1.0 - output_batch[j*count + k]);
                        else
                            input_grad_batch[i*count + k] -= output_grad_batch[j*count + k] * output_batch[j*count + k];
                    }
                    input_grad_batch[i*count + k] *= output_batch[i*count + k];
                }
            }
        }
    } else/* TODO bfp16*/ {

    }
    if( input0_desc.data_format == DATA_FORMAT_NC4HW4) {
        std::shared_ptr<RawBuffer> tmpbuffer = std::make_shared<RawBuffer>(CalculateElementCount(input0_desc) * DataTypeUtils::GetBytesSize(output_data_type));
        if(output_data_type == DATA_TYPE_BFP16) {
            PackFloatBlob(tmpbuffer->force_to<bfp16_t*>(), input_grad->force_to<bfp16_t*>(), batch_for_pack, channel_for_pack, hw_for_pack);
        }
        else if(output_data_type == DATA_TYPE_FLOAT) {
            PackFloatBlob(tmpbuffer->force_to<float*>(), input_grad->force_to<float*>(), batch_for_pack, channel_for_pack, hw_for_pack);    
        }
        input_grad = tmpbuffer;
    }
    input_grad->SetDataType(input0_data_type);
    input_grad->SetDataFormat(input0_desc.data_format);
    UpdateGradValue(inputs[0], input_grad, context);
    return Status(TNN_OK); 
}
REGISTER_LAYER_GRAD(SoftMax, LAYER_SOFTMAX);

} //namespace train         
} //namspace TNN_NS