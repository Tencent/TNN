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

#include "tnn/train/grad/binary_layer_grad.h"
#include "tnn/train/operations/op_builder.h"
#include "tnn/utils/dims_offset_utils.h"

namespace TNN_NS {
namespace train {
DECLARE_BINARY_LAYER_GRAD(Add, LAYER_ADD);
DECLARE_BINARY_LAYER_GRAD(Sub, LAYER_SUB);
DECLARE_BINARY_LAYER_GRAD(Mul, LAYER_MUL);
//TODO: DIV MAX MIN
typedef std::function<float(float, float)> ELEWISE_OP;

/*
 * input or grads must be nchw format
 * need to improve performance
 */
Status cal_binary_grad(std::vector<float *> &input_grad_ptrs, const std::vector<float *> &input_ptrs, const std::vector<DimsVector> &input_shapes, const float *output_grad_ptr,
                 DimsVector shape_output, LayerType layer_type) {
    const int count        = DimsVectorUtils::Count(shape_output);
    auto input0_shape  = input_shapes[0];
    auto input1_shape  = input_shapes[1];
    DimsVector input0_index;
    DimsVector input1_index;
    input0_index.reserve(input0_shape.size());
    input1_index.reserve(input1_shape.size());
    int input0_offset, input1_offset;
    float* input0_ptr = input_ptrs[0];
    float* input1_ptr = input_ptrs[1];
    float* input0_grad_ptr = input_grad_ptrs[0];
    float* input1_grad_ptr = input_grad_ptrs[1];  
    for(int offset = 0; offset< count; ++offset) {
        DimsVector output_index = DimsOffsetUtils::ConvertOffsetToIndex(shape_output, offset);
        input0_index.clear();
        input1_index.clear();
        int diff = shape_output.size() - input0_shape.size();
        for(int i = 0; i < input0_shape.size(); ++i) {
            input0_index.push_back(std::min(output_index[i + diff], input0_shape[i] - 1));
        }
        diff = shape_output.size() - input1_shape.size();
        for(int i = 0; i < input1_shape.size(); ++i) {
            input1_index.push_back(std::min(output_index[i + diff], input1_shape[i] - 1));
        } 
        input0_offset = DimsOffsetUtils::ConvertIndexToOffset(input0_shape, input0_index);
        input1_offset = DimsOffsetUtils::ConvertIndexToOffset(input1_shape, input1_index);
        switch (layer_type)
        {
            case LAYER_ADD:
                input0_grad_ptr[input0_offset] += output_grad_ptr[offset];
                input1_grad_ptr[input1_offset] += output_grad_ptr[offset];
                break;
            case LAYER_SUB:
                input0_grad_ptr[input0_offset] += output_grad_ptr[offset];
                input1_grad_ptr[input1_offset] -= output_grad_ptr[offset];
                break;
            case LAYER_MUL:
                input0_grad_ptr[input0_offset] += output_grad_ptr[offset] * input1_ptr[input1_offset];
                input1_grad_ptr[input1_offset] += output_grad_ptr[offset] * input0_ptr[input0_offset];
                break;
            case LAYER_DIV:
                // y = x1 / x2
                // dy/dx1 = 1.0/x2
                // dy/dx2 = x1 / x2^2
                input0_grad_ptr[input0_offset] += output_grad_ptr[offset] / input1_ptr[input1_offset];
                input1_grad_ptr[input1_offset] += output_grad_ptr[offset] * input0_ptr[input0_offset] / (input1_ptr[offset] * input1_ptr[offset]);
                break;         
            default:
                return Status(TNN_TRAIN_ERROR, "BinaryLayerGrad not support layer type");
        }        

    }
    return TNN_OK;
}

Status BinaryLayerGrad::OnGrad(const BaseLayer *layer, TrainContext &context) {
    auto inputs  = layer->input_blobs_;
    auto outputs = layer->output_blobs_;
    if (inputs.size() < 1 || inputs.size() > 2 || outputs.size() != 1) {
        return Status(TNN_TRAIN_ERROR, "input size or output size not match in BinaryLayerGrad");
    }
    auto output_dims = outputs[0]->GetBlobDesc().dims;
    auto output_data_type = outputs[0]->GetBlobDesc().data_type;
    bool need_broadcast = false;
    for(int i=1; i<inputs.size(); ++i) {
        auto input_data_type = inputs[i]->GetBlobDesc().data_type; 
        if (input_data_type != output_data_type) {
            return Status(TNN_TRAIN_ERROR, "input datatype and output datatype not match in BinaryLayerGrad");
        }
    }
    if (output_data_type != DATA_TYPE_FLOAT) {
        return Status(TNN_TRAIN_ERROR, "output datatype not match in BinaryLayerGrad");
    }
    auto iter_output         = context.backward_grads_blob.find(outputs[0]);
    if (iter_output == context.backward_grads_blob.end()) {
        return Status(TNN_TRAIN_ERROR, "BinaryLayerGrad output grad not find");
    }

    auto layer_param = dynamic_cast<MultidirBroadcastLayerParam *>(layer->param_);
    if (!layer_param) {
        LOGE("Error: BinaryLayerGrad layer param is nil\n");
        return Status(TNN_TRAIN_ERROR, "Error: BinaryLayerGrad layer param is nil");
    }
    auto layer_res = dynamic_cast<EltwiseLayerResource *>(layer->resource_);
    if (!((inputs.size() == 1 && layer_res) || inputs.size() >= 2)) {
        LOGE("Error: BinaryLayerGrad invalid inputs count\n");
        return Status(TNN_TRAIN_ERROR, "BinaryLayerGrad invalid inputs count");
    }

    LayerType layer_type = layer->type_;
    std::vector<std::shared_ptr<RawBuffer>> input_grads;
    std::vector<RawBuffer> input_tmp_buffers; //just for cache; may be host transfered data;
    std::vector<DimsVector> input_shapes;
    std::vector<float *> input_ptrs;
    std::vector<float *> input_grad_ptrs;
    ParamWrappers grad_keys;

    //so ugly code blob!!!
    if (inputs.size() >= 2) {
        input_ptrs.resize(inputs.size());
        for (size_t inid = 0; inid < inputs.size(); inid++) {
            auto dims = inputs[inid]->GetBlobDesc().dims;
            void *data = GetBlobHandle(inputs[inid]);
            input_shapes.push_back(dims);
            input_tmp_buffers.push_back(RawBuffer());
            ConvertToNCHW(data, input_tmp_buffers.back(), inputs[inid]->GetBlobDesc());
            input_ptrs.push_back(static_cast<float *>(data));

            auto input_grad = std::make_shared<RawBuffer>(DimsVectorUtils::Count(dims) * DataTypeUtils::GetBytesSize(output_data_type), dims);
            input_grad->SetDataType(output_data_type);
            input_grad->SetDataFormat(DATA_FORMAT_NCHW);
            input_grad_ptrs.push_back(input_grad->force_to<float *>());
            input_grads.push_back(input_grad);
            grad_keys.push_back(ParamWrapper(inputs[inid]));
        }
    } else if(layer_res->element_handle.GetBytesSize() > 0) {
        DimsVector input_shape0 = inputs[0]->GetBlobDesc().dims;
        void *data = GetBlobHandle(inputs[0]);
        input_tmp_buffers.push_back(RawBuffer());
        ConvertToNCHW(data, input_tmp_buffers.back(), inputs[0]->GetBlobDesc());
        if (layer_param->weight_input_index == 0) {
            //layer resource weight must be nchw, so push it directly
            input_ptrs.push_back(layer_res->element_handle.force_to<float *>()); 
            input_shapes.push_back(layer_res->element_shape);
            
            auto weight_grad = std::make_shared<RawBuffer>(DimsVectorUtils::Count(layer_res->element_shape) * DataTypeUtils::GetBytesSize(output_data_type), layer_res->element_shape);
            weight_grad->SetDataType(output_data_type);
            weight_grad->SetDataFormat(DATA_FORMAT_NCHW);
            input_grad_ptrs.push_back(weight_grad->force_to<float *>());
            input_grads.push_back(weight_grad);
            grad_keys.push_back(ParamWrapper(&(layer_res->element_handle)));

            input_ptrs.push_back(static_cast<float *>(data));
            input_shapes.push_back(input_shape0);
            auto input_grad = std::make_shared<RawBuffer>(DimsVectorUtils::Count(input_shape0) * DataTypeUtils::GetBytesSize(output_data_type), input_shape0);
            input_grad->SetDataType(output_data_type);
            input_grad->SetDataFormat(DATA_FORMAT_NCHW);
            input_grad_ptrs.push_back(input_grad->force_to<float *>());
            input_grads.push_back(input_grad);
            grad_keys.push_back(ParamWrapper(inputs[0]));
        } else {
            input_ptrs.push_back(static_cast<float *>(data));
            input_shapes.push_back(input_shape0);
            auto input_grad = std::make_shared<RawBuffer>(DimsVectorUtils::Count(input_shape0) * DataTypeUtils::GetBytesSize(output_data_type), input_shape0);
            input_grad->SetDataType(output_data_type);
            input_grad->SetDataFormat(DATA_FORMAT_NCHW);
            input_grad_ptrs.push_back(input_grad->force_to<float *>());
            input_grads.push_back(input_grad);
            grad_keys.push_back(ParamWrapper(inputs[0]));

            input_ptrs.push_back(layer_res->element_handle.force_to<float *>()); 
            input_shapes.push_back(layer_res->element_shape);
            
            auto weight_grad = std::make_shared<RawBuffer>(DimsVectorUtils::Count(layer_res->element_shape) * DataTypeUtils::GetBytesSize(output_data_type), layer_res->element_shape);
            weight_grad->SetDataType(output_data_type);
            weight_grad->SetDataFormat(DATA_FORMAT_NCHW);
            input_grad_ptrs.push_back(weight_grad->force_to<float *>());
            input_grads.push_back(weight_grad);
            grad_keys.push_back(ParamWrapper(&(layer_res->element_handle)));
        }
    } else {
        return Status(TNNERR_LAYER_ERR, "BinaryLayerGrad input size error");
    }
    cal_binary_grad(input_grad_ptrs, input_ptrs, input_shapes, iter_output->second->force_to<float*>(), iter_output->second->GetBufferDims(), layer_type);
    for(int i=0; i<grad_keys.size(); ++i) {
        if(!grad_keys[i].IsBlobPointer() && !grad_keys[i].IsRawbufferSharedPtr()) 
            return Status(TNNERR_LAYER_ERR, "BinaryLayerGrad calcute error");
        auto data_format = grad_keys[i].GetBlobOrRawbufferDataformat();
        auto cur_grad = input_grads[i];
        if(data_format == DATA_FORMAT_NC4HW4) {
            BlobDesc desc;
            desc.dims = grad_keys[i].GetBlobOrRawbufferDims();;
            desc.data_format = data_format;
            desc.data_type = output_data_type;
        
            ConvertToNC4HW4(cur_grad, desc);
        }
        if(grad_keys[i].IsBlobPointer())
            UpdateGradValue(grad_keys[i].GetBlobPointer(), cur_grad, context);
        else 
            UpdateGradValue(grad_keys[i].GetBlobPointer(), cur_grad, context);
    }
    return Status(TNN_OK);
}
REGISTER_BINARY_LAYER_GRAD(Add, LAYER_ADD);
REGISTER_BINARY_LAYER_GRAD(Sub, LAYER_SUB);
REGISTER_BINARY_LAYER_GRAD(Mul, LAYER_MUL);

} // namespace train
} // namespace TNN_NS