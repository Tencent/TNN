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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(RoiAlign, LAYER_ROIALIGN);

bool RoiAlignTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    if (!(nbInputs == 3 && nbOutputs == 1 && pos < nbInputs + nbOutputs)) return false;
    switch (pos) {
        case 0: return inOut[pos].type == nvinfer1::DataType::kFLOAT;
        case 1: return inOut[pos].type == nvinfer1::DataType::kFLOAT;
        case 2: return inOut[pos].type == nvinfer1::DataType::kINT32;
        case 3: return inOut[pos].type == nvinfer1::DataType::kFLOAT;
        default: return false;
    }
}

Status RoiAlignTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* RoiAlignTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "RoiAlign";
}

nvinfer1::DataType RoiAlignTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept {
    return inputTypes[0];
}

ILayer* RoiAlignTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    auto layer_param = dynamic_cast<RoiAlignLayerParam*>(param_);
    if (!layer_param) {
        LOGE("RoiAlignTRTPluginLayerBuilder: Unable to get layer param.");
        return nullptr;
    }

    int coordinate_transformation_mode = layer_param->aligned;
    int mode = layer_param->mode;
    int output_height = layer_param->output_height;
    int output_width = layer_param->output_width;
    int sampling_ratio = layer_param->sampling_ratio;
    float spatial_scale = layer_param->spatial_scale;

    auto creator = getPluginRegistry()->getPluginCreator("ROIAlign_TRT", "1", "");
    if (!creator) {
        LOGE("ROIAlignTRTLayerBuilder: Unable to find creator for TRT ROIAlign_TRT V1 Plugin Layer.");
        return nullptr;
    }

    std::vector<ITensor*> input_tensors;
    for (int i = 0; i < input_blobs_.size(); i++) {
        auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[i])->GetForeignTensor();
        input_tensors.push_back(std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor());
    }

    std::vector<nvinfer1::PluginField> roi_align_v1_field;
    roi_align_v1_field.emplace_back("coordinate_transformation_mode", &coordinate_transformation_mode, nvinfer1::PluginFieldType::kINT32, 1);
    roi_align_v1_field.emplace_back("mode", &mode, nvinfer1::PluginFieldType::kINT32, 1);
    roi_align_v1_field.emplace_back("output_height", &output_height, nvinfer1::PluginFieldType::kINT32, 1);
    roi_align_v1_field.emplace_back("output_width", &output_width, nvinfer1::PluginFieldType::kINT32, 1);
    roi_align_v1_field.emplace_back("sampling_ratio", &sampling_ratio, nvinfer1::PluginFieldType::kINT32, 1);
    roi_align_v1_field.emplace_back("spatial_scale", &spatial_scale, nvinfer1::PluginFieldType::kFLOAT32, 1);

    PluginFieldCollection roi_align_v1_fc {6, roi_align_v1_field.data()};
    IPluginV2* plugin_obj = creator->createPlugin(layer_name_.c_str(), &roi_align_v1_fc);
    auto layer = network->addPluginV2(input_tensors.data(), input_blobs_.size(), *plugin_obj);
    if (layer != nullptr) {
        layer->setName((layer_name_).c_str());
    }

    return layer;
}

DimsExprs RoiAlignTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    RoiAlignLayerParam* param = dynamic_cast<RoiAlignLayerParam*>(param_);

    DimsExprs output;
    output.nbDims = 4;
    output.d[0] = inputs[1].d[0];
    output.d[1] = inputs[0].d[1];
    output.d[2] = exprBuilder.constant(param->output_height);
    output.d[3] = exprBuilder.constant(param->output_width);
    return output;
}

const char* RoiAlignPluginCreator::getPluginName() const noexcept {
    return "RoiAlign";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(RoiAlign, LAYER_ROIALIGN);


}  //  namespace TNN_NS
