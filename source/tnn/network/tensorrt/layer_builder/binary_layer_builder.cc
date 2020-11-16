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

#include "tnn/network/tensorrt/layer_builder/binary_layer_builder.h"

namespace TNN_NS {

BinaryTRTLayerBuilder::BinaryTRTLayerBuilder(LayerType ignore) : TensorRTLayerBuilder(ignore) {
}

ILayer* BinaryTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto input_foreign_tensor1 = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    bool int8 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor1)->GetInt8Mode();
    if (input_blobs_.size() == 2) {
        auto input_foreign_tensor1 = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
        auto input_foreign_tensor2 = dynamic_cast<ForeignBlob*>(input_blobs_[1])->GetForeignTensor();
        auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
        auto input_tensor1 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor1)->GetTensor();
        auto input_tensor2 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor2)->GetTensor();

        ILayer* last_layer;
        IElementWiseLayer* layer = network->addElementWise(*input_tensor1, *input_tensor2, m_op);
        if (layer != nullptr) {
            layer->setName(layer_name_.c_str());
            last_layer = layer;
        }

        if (int8) {
            float output_scale_value = std::dynamic_pointer_cast<TensorRTTensor>(output_foreign_tensor)->GetIntResource()->scale_handle.force_to<float*>()[0];
            Weights output_quant_shift;
            output_quant_shift.type = nvinfer1::DataType::kFLOAT;
            output_quant_shift.values = nullptr;
            output_quant_shift.count = 0;

            Weights output_quant_scale;
            output_quant_scale.type = nvinfer1::DataType::kFLOAT;
            float* output_quant_scale_data = (float*)malloc(sizeof(float));
            int8_weight_data.push_back(output_quant_scale_data);
            *output_quant_scale_data = output_scale_value;
            output_quant_scale.values = (void*)output_quant_scale_data;
            output_quant_scale.count = 1;

            Weights output_quant_power;
            output_quant_power.type = nvinfer1::DataType::kFLOAT;
            output_quant_power.values = nullptr;
            output_quant_power.count = 0;

            auto output_quant_layer = network->addScale(*(layer->getOutput(0)), ScaleMode::kUNIFORM,
                output_quant_shift, output_quant_scale, output_quant_power);
            std::string output_quant_layer_name = layer_name_ + "_output_quant_";
            output_quant_layer->setOutputType(0, nvinfer1::DataType::kINT8);
            output_quant_layer->setName(output_quant_layer_name.c_str());

            Weights output_dequant_shift;
            output_dequant_shift.type = nvinfer1::DataType::kFLOAT;
            output_dequant_shift.values = nullptr;
            output_dequant_shift.count = 0;

            Weights output_dequant_scale;
            output_dequant_scale.type = nvinfer1::DataType::kFLOAT;
            float* output_dequant_scale_data = (float*)malloc(sizeof(float));
            int8_weight_data.push_back(output_dequant_scale_data);
            *output_dequant_scale_data = 1 / output_scale_value;
            output_dequant_scale.values = (void*)output_dequant_scale_data;
            output_dequant_scale.count = 1;

            Weights output_dequant_power;
            output_dequant_power.type = nvinfer1::DataType::kFLOAT;
            output_dequant_power.values = nullptr;
            output_dequant_power.count = 0;

            auto output_dequant_layer = network->addScale(*(output_quant_layer->getOutput(0)), ScaleMode::kUNIFORM,
                output_dequant_shift, output_dequant_scale, output_dequant_power);
            std::string output_dequant_layer_name = layer_name_ + "_output_dequant_";
            output_dequant_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
            output_dequant_layer->setName(output_dequant_layer_name.c_str());
            last_layer = output_dequant_layer;
        }
        return last_layer;
    } else {
        auto paramlist = dynamic_cast<MultidirBroadcastLayerParam*>(param_);
        Weights weight;
        auto resource = dynamic_cast<EltwiseLayerResource*>(resource_);
        weight.type = nvinfer1::DataType::kFLOAT;
        weight.values = resource->element_handle.force_to<void*>();
        weight.count = resource->element_handle.GetDataCount();
        Dims4 dims(1, weight.count, 1, 1);
        int n = input_blobs_[0]->GetBlobDesc().dims[0];
        int c = input_blobs_[0]->GetBlobDesc().dims[1];
        int h = input_blobs_[0]->GetBlobDesc().dims[2];
        int w = input_blobs_[0]->GetBlobDesc().dims[3];
        if (weight.count == 1) {
            // broadcast on chw
            dims = Dims4(1, 1, 1, 1);
        } else if (weight.count == h * w) {
            // broadcast on channel  
            dims = Dims4(1, 1, h, w);
        } else if (weight.count == c) {
            // broadcast on hw
            dims = Dims4(1, c, 1, 1);
        } else if (weight.count == c * h * w) {
            // no broadcast
            dims = Dims4(1, c, h, w);
        } else if (weight.count == c * w) {
            // broadcast on h
            dims = Dims4(1, c, 1, w);
        } else if (weight.count == c * h) {
            // broadcast on w
            dims = Dims4(1, c, h, 1);
        }  else if (w!= 1 && h==1 && weight.count % w == 0) {
            // for weights shape: {1, 1, h, w}
            //     input   shape: {1, c, 1, w}
            h = weight.count / w;
            dims = Dims4(1, 1, h, w);
        } else if (h!= 1 && c==1 && weight.count % h == 0) {
            // for weights shape: {1, c, h, 1}
            //     input   shape: {1, 1, h, w}
            c = weight.count / h;
            dims = Dims4(1, c, h, 1);
        } else if (c!= 1 && h==1 && weight.count % c == 0) {
            // for weights shape: {1, c, h, 1}
            //     input   shape: {1, c, 1, w}
            h = weight.count / c;
            dims = Dims4(1, c, h, 1);
        }
        ILayer* const_layer = network->addConstant(dims, weight);
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
        auto src_a = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
        auto src_b = const_layer->getOutput(0);
        if (paramlist->weight_input_index == 0) {
            std::swap(src_a, src_b);
        }
        IElementWiseLayer* layer = network->addElementWise(*src_a, *src_b, m_op);
        if (layer != nullptr) {
            layer->setName(layer_name_.c_str());
        }
        return layer;
    }
}

}  //  namespace TNN_NS