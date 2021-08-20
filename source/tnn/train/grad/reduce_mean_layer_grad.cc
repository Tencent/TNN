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
DECLARE_LAYER_GRAD(ReduceMean, LAYER_REDUCE_MEAN);
void CalculateReduceDims(Blob *input_blob, ReduceLayerParam *layer_param,
                           std::vector<std::tuple<int, int, int>> &reduce_dims) {
    auto input_dims = input_blob->GetBlobDesc().dims;
    auto axes       = layer_param->axis;
    std::sort(axes.begin(), axes.end());
    reduce_dims.clear();
    for (const auto &axis : axes) {
        int outer_count   = DimsVectorUtils::Count(input_dims, 0, axis);
        int reducer_count = input_dims[axis];
        int inner_count   = DimsVectorUtils::Count(input_dims, axis + 1);
        inner_count       = inner_count == 0 ? 1 : inner_count;
        reduce_dims.emplace_back(std::make_tuple(outer_count, reducer_count, inner_count));
        input_dims[axis] = 1;
    }
}
Status ReduceMeanLayerGrad::OnGrad(const BaseLayer* layer, TrainContext& context){
    auto inputs = layer->input_blobs_;
    auto outputs = layer->output_blobs_;
    if(inputs.size() != 1 || outputs.size() != 1) {
        return Status(TNN_TRAIN_ERROR, "input size or output size not match in ReduceMeanLayerGrad");
    }
    auto input0_desc = inputs[0]->GetBlobDesc();
    auto output_desc = outputs[0]->GetBlobDesc();
    auto input0_data_type = input0_desc.data_type;
    auto output_data_type = output_desc.data_type;
    auto input0_dims = input0_desc.dims;
    auto output_dims = output_desc.dims;
    if((output_data_type != DATA_TYPE_BFP16 && output_data_type != DATA_TYPE_FLOAT) || input0_data_type != output_data_type) {
       return Status(TNN_TRAIN_ERROR, "output datatype not match in ReduceMeanLayerGrad"); 
    }
    auto layer_param = dynamic_cast<ReduceLayerParam *>(layer->param_);
    if(layer_param == nullptr || layer_param->axis.size() <= 0 )
        return Status(TNN_TRAIN_ERROR, "reduce layer param axis error");
    auto iter_output_grad = context.backward_grads_blob.find(outputs[0]);
    if(iter_output_grad == context.backward_grads_blob.end()) {
        return Status(TNN_TRAIN_ERROR, "reduce layer output grad not found");
    }
    // TODO:Abstract all the reduce method to an op 
    int batch = input0_dims[0];
    int channel = input0_dims[1];
    int hw = DimsVectorUtils::Count(input0_dims, 2);
    int input_count = batch * channel * hw;
    int output_count = DimsVectorUtils::Count(output_dims);
    void* input_ptr = static_cast<void*>(static_cast<char*>(inputs[0]->GetHandle().base) + inputs[0]->GetHandle().bytes_offset);
    //void* output_ptr = outputs[0]->GetHandle().base + outputs[0]->GetHandle().bytes_offset;
    RawBuffer input_buffer(input_count* DataTypeUtils::GetBytesSize(input0_data_type));
    if(input0_desc.data_format == DATA_FORMAT_NC4HW4) {
        if(input0_data_type== DATA_TYPE_BFP16) {
            UnpackFloatBlob(input_buffer.force_to<bfp16_t*>(), static_cast<bfp16_t*>(input_ptr), batch, channel, hw);
            input_ptr = input_buffer.force_to<void*>();
        }
        else if(input0_data_type == DATA_TYPE_FLOAT)
            UnpackFloatBlob(input_buffer.force_to<float*>(), static_cast<float*>(input_ptr), batch, channel, hw);    
            input_ptr = input_buffer.force_to<void*>();
    }
    // batch = output_dims[0];
    // channel = output_dims[1];
    // hw = DimsVectorUtils::Count(output_dims, 2);
    // RawBuffer output_buffer(output_count* DataTypeUtils::GetBytesSize(output_data_type));
    // if(output_desc.data_format == DATA_FORMAT_NC4HW4) {
    //     if(output_data_type == DATA_TYPE_BFP16) {
    //         UnpackFloatBlob(output_buffer.force_to<bfp16_t*>(), static_cast<bfp16_t*>(output_ptr), batch, channel, hw);
    //         output_ptr = output_buffer.force_to<void*>();
    //     }
    //     else if(output_data_type == DATA_TYPE_FLOAT) {
    //         UnpackFloatBlob(output_buffer.force_to<float*>(), static_cast<float*>(output_ptr), batch, channel, hw);    
    //         output_ptr = output_buffer.force_to<void*>();
    //     }
    // }

    // input_grad = 1.0/reducer_count * outout_grad ; then broadcast it from output shape to input shape;
    // <outer count, reduce count, inner count>
    std::vector<std::tuple<int, int, int>> reduce_dims;
    CalculateReduceDims(inputs[0], layer_param, reduce_dims);
    std::shared_ptr<RawBuffer> buffer1 = std::make_shared<RawBuffer>(DimsVectorUtils::Count(input0_dims) * DataTypeUtils::GetBytesSize(input0_data_type));
    std::shared_ptr<RawBuffer> buffer2 = std::make_shared<RawBuffer>(DimsVectorUtils::Count(input0_dims) * DataTypeUtils::GetBytesSize(input0_data_type));
    std::shared_ptr<RawBuffer> last_grad_res;
    memcpy(buffer1->force_to<void*>(), iter_output_grad->second->force_to<void*>(), output_count* DataTypeUtils::GetBytesSize(output_data_type));
    if(input0_data_type == DATA_TYPE_FLOAT) {
        float* input_buffer_ptr = buffer1->force_to<float*>();
        float* output_buffer_ptr = buffer2->force_to<float*>();
        for(int i=0; i<output_count; i++) {
            input_buffer_ptr[i] *= (float) output_count / (float)input_count;
        }
        int pos_output;
        int pos_input;
        for (int i = reduce_dims.size() - 1; i >=0 ; --i) {
            auto reduce_dim   = reduce_dims[i];
            auto outer_count  = std::get<0>(reduce_dim);
            auto reduce_count = std::get<1>(reduce_dim);
            auto inner_count  = std::get<2>(reduce_dim);
            for(auto i1=0; i1<outer_count; ++i1) {
                for(auto i2=0;i2<reduce_count; ++i2)
                    for(auto i3=0; i3<inner_count; ++i3) {
                        pos_input = i1*inner_count + i3;
                        pos_output = i1*reduce_count*inner_count + i2*inner_count + i3;
                        output_buffer_ptr[pos_output] = input_buffer_ptr[pos_input];
                    }
            }
            std::swap(input_buffer_ptr, output_buffer_ptr);
        }
        if(reduce_dims.size() % 2 == 0) {
            last_grad_res = buffer1;
        } else {
            last_grad_res = buffer2;
        }
    } else/* TODO bfp16*/ {

    }
    if( input0_desc.data_format == DATA_FORMAT_NC4HW4) {
        std::shared_ptr<RawBuffer> tmpbuffer = std::make_shared<RawBuffer>(Calculate1DMemorySize(input0_desc).dims[0]* DataTypeUtils::GetBytesSize(output_data_type));
        if(output_data_type == DATA_TYPE_BFP16) {
            PackFloatBlob(tmpbuffer->force_to<bfp16_t*>(), last_grad_res->force_to<bfp16_t*>(), batch, channel, hw);
        }
        else if(output_data_type == DATA_TYPE_FLOAT) {
            PackFloatBlob(tmpbuffer->force_to<float*>(), last_grad_res->force_to<float*>(), batch, channel, hw);    
        }
        last_grad_res = tmpbuffer;
    }
    last_grad_res->SetDataType(input0_data_type);
    last_grad_res->SetBufferDims(input0_dims);
    last_grad_res->SetDataFormat(input0_desc.data_format);
    UpdateGradValue(inputs[0], last_grad_res, context);
    return Status(TNN_OK); 
}
REGISTER_LAYER_GRAD(ReduceMean, LAYER_REDUCE_MEAN);

} //namespace train         
} //namspace TNN_NS