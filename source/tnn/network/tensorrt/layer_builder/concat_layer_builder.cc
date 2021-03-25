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

#include "tnn/network/tensorrt/tensorrt_network.h"
#include "tnn/network/tensorrt/layer_builder/tensorrt_layer_builder.h"

namespace TNN_NS {

DECLARE_TENSORRT_LAYER_BUILDER(Concat, LAYER_CONCAT);

ILayer* ConcatTRTLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto paramlist = dynamic_cast<ConcatLayerParam*>(param_);
    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    bool int8 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetInt8Mode();
    size_t nbInputs = input_blobs_.size();
    ITensor ** input_tensors = new ITensor*[nbInputs];
    for (int i = 0; i < nbInputs; i++) {
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[i])->GetForeignTensor();
        auto tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
        input_tensors[i] = tensor;
    }

    m_network->m_concat_blob_names.insert(output_blobs_[0]->GetBlobDesc().name);

    ILayer* last_layer;
    IConcatenationLayer* layer = network->addConcatenation(input_tensors, nbInputs);
    if (layer != nullptr) {
        layer->setName(layer_name_.c_str());
        layer->setAxis(paramlist->axis);
        last_layer = layer;
    }
    delete [] input_tensors;

    if (int8) {
        float output_scale_value = std::dynamic_pointer_cast<TensorRTTensor>(output_foreign_tensor)->GetIntResource()->scale_handle.force_to<float*>()[0];
        Weights output_quant_shift;
        output_quant_shift.type = nvinfer1::DataType::kFLOAT;
        output_quant_shift.values = nullptr;
        output_quant_shift.count = 0;

        Weights output_quant_scale;
        float* output_quant_scale_data = (float*)malloc(sizeof(float));
        int8_weight_data.push_back(output_quant_scale_data);
        *output_quant_scale_data = output_scale_value;
        output_quant_scale.type = nvinfer1::DataType::kFLOAT;
        output_quant_scale.values = (void*)output_quant_scale_data;
        output_quant_scale.count = 1;

        Weights output_quant_power;
        output_quant_power.type = nvinfer1::DataType::kFLOAT;
        output_quant_power.values = nullptr;
        output_quant_power.count = 0;

        auto output_quant_layer = network->addScale(*(last_layer->getOutput(0)), ScaleMode::kUNIFORM,
            output_quant_shift, output_quant_scale, output_quant_power);
        std::string output_quant_name = layer_name_ + "_output_quant_";
        output_quant_layer->setOutputType(0, nvinfer1::DataType::kINT8);
        output_quant_layer->setName(output_quant_name.c_str());

        Weights output_dequant_shift;
        output_dequant_shift.type = nvinfer1::DataType::kFLOAT;
        output_dequant_shift.values = nullptr;
        output_dequant_shift.count = 0;

        Weights output_dequant_scale;
        float* output_dequant_scale_data = (float*)malloc(sizeof(float));
        int8_weight_data.push_back(output_dequant_scale_data);
        *output_dequant_scale_data = 1 / output_scale_value;
        output_dequant_scale.type = nvinfer1::DataType::kFLOAT;
        output_dequant_scale.values = (void*)output_dequant_scale_data;
        output_dequant_scale.count = 1;

        Weights output_dequant_power;
        output_dequant_power.type = nvinfer1::DataType::kFLOAT;
        output_dequant_power.values = nullptr;
        output_dequant_power.count = 0;

        auto output_dequant_layer = network->addScale(*(output_quant_layer->getOutput(0)),
            ScaleMode::kUNIFORM, output_dequant_shift, output_dequant_scale, output_dequant_power);
        std::string output_dequant_name = layer_name_ + "_output_dequant_";
        output_dequant_layer->setOutputType(0, nvinfer1::DataType::kFLOAT);
        output_dequant_layer->setName(output_dequant_name.c_str());
        last_layer = output_dequant_layer;
        std::dynamic_pointer_cast<TensorRTTensor>(output_foreign_tensor)->SetQuantized();
    }

    return last_layer;
}

REGISTER_TENSORRT_LAYER_BUILDER(Concat, LAYER_CONCAT);

}  //  namespace TNN_NS

