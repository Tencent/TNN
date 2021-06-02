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

#include "tnn/network/tensorrt/layer_builder/tensorrt_plugin_layer_builder.h"
#include "tnn/network/tensorrt/utils.h"

namespace TNN_NS {

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Gather, LAYER_GATHER);

bool GatherTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    auto layer_param = dynamic_cast<GatherLayerParam*>(param_);

    auto support_fp32_i32 = (inOut[pos].type == nvinfer1::DataType::kFLOAT ||
                             inOut[pos].type == nvinfer1::DataType::kINT32) &&
                             inOut[pos].format == nvinfer1::TensorFormat::kNCHW;
    auto support_i32 = inOut[pos].type == nvinfer1::DataType::kINT32 &&
                       inOut[pos].format == nvinfer1::TensorFormat::kNCHW;
    auto support_f32_f16_i32 = (inOut[pos].type == nvinfer1::DataType::kFLOAT ||
                                inOut[pos].type == nvinfer1::DataType::kINT32 ||
                                inOut[pos].type == nvinfer1::DataType::kHALF) &&
                                inOut[pos].format == nvinfer1::TensorFormat::kNCHW;

    if (layer_param->data_in_resource) {
        // if data in resource, output dtype only support fp32 & int32
        if (layer_param->indices_in_resource) {
            // resource -> input_data, output[0] -> output_data
            // resource -> input_indices
            if (nbInputs != 0 && nbOutputs != 1) {
                return false;
            }
            return support_fp32_i32;
        } else {
            // resource -> input_data, output[0] -> output_data
            // input[0] -> input_indices
            if (nbInputs != 1 && nbOutputs != 1) {
                return false;
            }
            if (pos == 0) {
                return support_i32;
            } else {
                return support_fp32_i32;
            }
        }
    } else {
        // if data not in resouce, output dtype = input dtype, support fp32 & fp16 & int32
        if (layer_param->indices_in_resource) {
            // input[0] -> input_data, output[0] -> output_data
            // resource -> input_indices
            if (nbInputs != 1 && nbOutputs != 1) {
                return false;
            }
            return support_f32_f16_i32 && inOut[pos].type == inOut[0].type;
        } else {
            // input[0] -> input_data, output[0] -> output_data
            // input[1] -> input_indices
            if (nbInputs != 2 && nbOutputs != 1) {
                return false;
            }
            if (pos == 1) {
                return support_i32;
            } else {
                return support_f32_f16_i32 && inOut[pos].type == inOut[0].type;
            }
        }
    }
}

Status GatherTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* GatherTRTPluginLayerBuilder::getPluginType() const {
    return "Gather";
}

nvinfer1::DataType GatherTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    auto layer_param = dynamic_cast<GatherLayerParam*>(param_);
    auto layer_resource = dynamic_cast<GatherLayerResource*>(resource_);

    // if data_in_reousce, output dtype == resource dtype
    // else output dtype == input dtype
    if (layer_param->data_in_resource) {
        return ConvertToTRTDataType(layer_resource->data.GetDataType());
    } else {
        return inputTypes[0];
    }
}

// trt gather performs slow in some cases, and may not support indices_data < 0
// if shape tensor, we keep trt IGatherLayer
// if not shape tensor, we use customized gather implementation
ILayer* GatherTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    // if shape tensor
    if (GetInputITensors()[0]->getDimensions().nbDims == 0 ||
        GetInputITensors()[0]->getDimensions().nbDims == 1) {

        auto layer_param = dynamic_cast<GatherLayerParam*>(param_);
        if (layer_param == nullptr) {
            LOGE("GatherTRTLayerBuilder layer_param is null");
            return nullptr;
        }
        int axis = layer_param->axis;

        auto layer_resource = dynamic_cast<GatherLayerResource*>(resource_);
        if ((layer_param->data_in_resource || layer_param->indices_in_resource) && !layer_resource) {
            LOGE("Gather resource is invalid");
            return nullptr;
        }

        nvinfer1::ITensor * data = nullptr;
        nvinfer1::ITensor * indices = nullptr;
        if (layer_param->data_in_resource) {
            auto const_layer = ConvertWeightToConstLayer(network, &(layer_resource->data));
            if (const_layer != nullptr) {
                data = const_layer->getOutput(0);
            }
        } else {
            auto foreign_tensor = dynamic_cast<ForeignBlob*>(*input_blobs_.begin())->GetForeignTensor();
            data = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
        }

        if (layer_param->indices_in_resource) {
            auto const_layer = ConvertWeightToConstLayer(network, &(layer_resource->indices));
            if (const_layer != nullptr) {
                indices = const_layer->getOutput(0);
            }
        } else {
            auto foreign_tensor = dynamic_cast<ForeignBlob*>(*input_blobs_.rbegin())->GetForeignTensor();
            indices = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();
        }

        if (data == nullptr || indices == nullptr) {
            LOGE("GatherTRTLayerBuilder can not find data or indices\n");
            return nullptr;
        }

        return network->addGather(*data, *indices, axis);
    }
    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

DimsExprs GatherTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    auto layer_param = dynamic_cast<GatherLayerParam*>(param_);
    auto layer_resource = dynamic_cast<GatherLayerResource*>(resource_);

    DimsVector data_dims, indices_dims;
    DimsExprs inputs_data, inputs_indices;
    int data_size, indices_size;
    int input_count = 0;
    if (layer_param->data_in_resource) {
        data_dims = layer_resource->data.GetBufferDims();
        data_size = data_dims.size();
    } else {
        inputs_data = inputs[input_count];
        data_size = inputs_data.nbDims;
        input_count++;
    }

    if (layer_param->indices_in_resource) {
        indices_dims = layer_resource->indices.GetBufferDims();
        indices_size = indices_dims.size();
    } else {
        inputs_indices = inputs[input_count];
        indices_size = inputs_indices.nbDims;
        input_count++;
    }

    int axis = layer_param->axis;
    DimsExprs output;
    int idx = 0;
    if (axis > 0 && axis < data_size) {
        for (int i = 0; i < axis; i++) {
            if (layer_param->data_in_resource) {
                output.d[idx] = exprBuilder.constant(data_dims[i]);
            } else {
                output.d[idx] = inputs_data.d[i];
            }
            idx++;
        }
    }
    for (int i = 0; i < indices_size; i++) {
        if (layer_param->indices_in_resource) {
            output.d[idx] = exprBuilder.constant(indices_dims[i]);
        } else {
            output.d[idx] = inputs_indices.d[i];
        }
        idx++;
    }
    if (axis < data_size - 1) {
        for (int i = axis + 1; i < data_size; i++) {
            if (layer_param->data_in_resource) {
                output.d[idx] = exprBuilder.constant(data_dims[i]);
            } else {
                output.d[idx] = inputs_data.d[i];
            }
            idx++;
        }
    }
    output.nbDims = idx;

    return output;
}

const char* GatherPluginCreator::getPluginName() const {
    return "Gather";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Gather, LAYER_GATHER);

}  //  namespace TNN_NS
