// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(Fused, LAYER_FUSED);

bool FusedTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    auto layer_param = dynamic_cast<FusedLayerParam*>(param_);
    if (!layer_param) {
        LOGE("FusedTRTLayerBuilder: Unable to get layer param.");
        return false;
    }

    if (layer_param->type == FusionType_AddBiasResidualLayerNorm ||
        layer_param->type == FusionType_FFN) {
        return (inOut[pos].type == nvinfer1::DataType::kHALF
                && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR
                && inOut[pos].type == inOut[0].type);
    } else if (layer_param->type == FusionType_Attention) {
        if (pos == 0 || pos >= nbInputs) {
            // TODO: ADD FLOAT
            //return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF)
            return (inOut[pos].type == nvinfer1::DataType::kHALF
                    && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR
                    && inOut[pos].type == inOut[0].type);
        } else {
            if (pos == 1 && layer_param->has_attention_mask) {
                // attention_mask
                //return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF)
                return (inOut[pos].type == nvinfer1::DataType::kHALF
                        && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR
                        && inOut[pos].type == inOut[0].type);
            } else {
                // trt_offsets or other shape-related inputs
                return (inOut[pos].type == nvinfer1::DataType::kINT32
                        && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
            }
        }
    }

    return false;
}

Status FusedTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* FusedTRTPluginLayerBuilder::getPluginType() const noexcept {
    return "Fused";
}

nvinfer1::DataType FusedTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const noexcept {
    return inputTypes[0];
}

ILayer* FusedTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) noexcept {
    auto layer_param = dynamic_cast<FusedLayerParam*>(param_);
    if (!layer_param) {
        LOGE("FusedTRTLayerBuilder: Unable to get layer param.");
        return nullptr;
    }

    if (layer_param->type == FusionType_TRTPlugin_BertQKVtoContextV1) {
        if (layer_param->bert_mha_hidden_size <= 0 || layer_param->bert_mha_num_heads <= 0) {
            LOGE("FusedTRTLayerBuilder: TRT QKVToContext V1 Plugin Layer got Wrong Param: num_heads and hidden_size for Multi-head Attention.");
            return nullptr;
        }

        auto creator = getPluginRegistry()->getPluginCreator("CustomQKVToContextPluginDynamic", "1");
        if (!creator) {
            LOGE("FusedTRTLayerBuilder: Unable to find creator for TRT QKVToContext V1 Plugin Layer.");
            return nullptr;
        }

        std::vector<ITensor*> input_tensors;
        auto in_x_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
        input_tensors.push_back(std::dynamic_pointer_cast<TensorRTTensor>(in_x_foreign_tensor)->GetTensor());
        if (input_blobs_.size() >= 2) {
            // input[1]: ReduceSum-ed Attention Mask of size [Batch]
            auto mask_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[1])->GetForeignTensor();
            input_tensors.push_back(std::dynamic_pointer_cast<TensorRTTensor>(mask_foreign_tensor)->GetTensor());
        }

        int in_data_type = int(input_tensors[0]->getType());   // DataType: kFloat = 0, kHalf = 1;
        int has_in_mask = input_blobs_.size() >= 2 ? true : false;
        float default_dq_probs = 1.0f/127.0f;

        std::vector<nvinfer1::PluginField> mha_v1_field;
        mha_v1_field.emplace_back("type_id", &in_data_type, nvinfer1::PluginFieldType::kINT32, 1);
        mha_v1_field.emplace_back("has_mask", &has_in_mask, nvinfer1::PluginFieldType::kINT32, 1);
        mha_v1_field.emplace_back("hidden_size", &(layer_param->bert_mha_hidden_size), nvinfer1::PluginFieldType::kINT32, 1);
        mha_v1_field.emplace_back("num_heads", &(layer_param->bert_mha_num_heads), nvinfer1::PluginFieldType::kINT32, 1);
        mha_v1_field.emplace_back("dq_probs", &default_dq_probs, nvinfer1::PluginFieldType::kFLOAT32, 1);

        PluginFieldCollection mhaV1FC {5, mha_v1_field.data()};
        IPluginV2* pluginObj = creator->createPlugin(layer_name_.c_str(), &mhaV1FC);
        auto layer = network->addPluginV2(input_tensors.data(), input_blobs_.size(), *pluginObj);
        if (layer != nullptr) {
            layer->setName((layer_name_).c_str());
        }

        return layer;
    } else if (layer_param->type == FusionType_TRTPlugin_BertQKVtoContextV2) {
        if (layer_param->bert_mha_hidden_size <= 0 || layer_param->bert_mha_num_heads <= 0) {
            LOGE("FusedTRTLayerBuilder: TRT QKVToContext V2 Plugin Layer got Wrong Param: num_heads and hidden_size for Multi-head Attention.");
            return nullptr;
        }

        auto creator = getPluginRegistry()->getPluginCreator("CustomQKVToContextPluginDynamic", "2");
        if (!creator) {
            LOGE("FusedTRTLayerBuilder: Unable to find creator for TRT QKVToContext V2 Plugin Layer.");
            return nullptr;
        }

        auto foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
        auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(foreign_tensor)->GetTensor();

        // V2.varseqlen has 4 inputs:
        // in_x:        [Total Seqlen under dense mode, Hidden_Size, ]
        // input_mask:  [Batch] dummy in var_seqlen mode, no size Requirements.
        // cu_seqlen:   [Batch+1], data be like [0, seq0_len, seq0_len+seq1_len, ..., cumulative sum of all all seq lens]
        // dummy:       [MaxSeqLen] dummy in var_seqlen mode, size of dummy required to be [MaxSeqLen]
        std::vector<ITensor*> input_tensors;
        auto in_x_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
        auto dummy_mask_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[1])->GetForeignTensor();
        auto cu_seqlen_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[2])->GetForeignTensor();
        auto dummy_max_seqlen_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[3])->GetForeignTensor();
        input_tensors.push_back(std::dynamic_pointer_cast<TensorRTTensor>(in_x_foreign_tensor)->GetTensor());
        input_tensors.push_back(std::dynamic_pointer_cast<TensorRTTensor>(dummy_mask_foreign_tensor)->GetTensor());
        input_tensors.push_back(std::dynamic_pointer_cast<TensorRTTensor>(cu_seqlen_foreign_tensor)->GetTensor());
        input_tensors.push_back(std::dynamic_pointer_cast<TensorRTTensor>(dummy_max_seqlen_foreign_tensor)->GetTensor());

        int in_data_type = 1;   // DataType: kFloat = 0, kHalf = 1, V2.var_seqlen requires half type to be one.
        int has_in_mask = 1;    // V2.var_seqlen requires input mask to be true (actually mask is dummy)
        float default_dq_probs = 1.0f/127.0f;
        int enable_var_seqlen = 1;

        std::vector<nvinfer1::PluginField> mha_v2_field;
        mha_v2_field.emplace_back("type_id", &in_data_type, nvinfer1::PluginFieldType::kINT32, 1);
        mha_v2_field.emplace_back("has_mask", &has_in_mask, nvinfer1::PluginFieldType::kINT32, 1);
        mha_v2_field.emplace_back("hidden_size", &(layer_param->bert_mha_hidden_size), nvinfer1::PluginFieldType::kINT32, 1);
        mha_v2_field.emplace_back("num_heads", &(layer_param->bert_mha_num_heads), nvinfer1::PluginFieldType::kINT32, 1);
        mha_v2_field.emplace_back("dq_probs", &default_dq_probs, nvinfer1::PluginFieldType::kFLOAT32, 1);
        mha_v2_field.emplace_back("var_seqlen", &enable_var_seqlen, nvinfer1::PluginFieldType::kFLOAT32, 1);

        PluginFieldCollection mhaV2FC {6, mha_v2_field.data()};
        IPluginV2* pluginObj = creator->createPlugin(layer_name_.c_str(), &mhaV2FC);
        auto layer = network->addPluginV2(input_tensors.data(), 4, *pluginObj);
        if (layer != nullptr) {
            layer->setName((layer_name_).c_str());
        }

        return layer;
    } else if (layer_param->type == FusionType_TRTPlugin_BertQKVtoContextV3) {
        if (layer_param->bert_mha_hidden_size <= 0 || layer_param->bert_mha_num_heads <= 0) {
            LOGE("FusedTRTLayerBuilder: TRT QKVToContext V3 Plugin Layer got Wrong Param: num_heads and hidden_size for Multi-head Attention.");
            return nullptr;
        }

        auto creator = getPluginRegistry()->getPluginCreator("CustomQKVToContextPluginDynamic", "3");
        if (!creator) {
            LOGE("FusedTRTLayerBuilder: Unable to find creator for TRT QKVToContext V3 Plugin Layer.");
            return nullptr;
        }

        return nullptr;
    } else if (layer_param->type == FusionType_AddBiasResidualLayerNorm ||
               layer_param->type == FusionType_FFN ||
               layer_param->type == FusionType_Attention) {
        return TensorRTPluginLayerBuilder::AddToNetwork(network);
    } else {
        LOGE("FusedTRTLayerBuilder: Layer fusion Type not supported.");
        return nullptr;
    }

    return nullptr;
}

DimsExprs FusedTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
}

const char* FusedPluginCreator::getPluginName() const noexcept {
    return "Fused";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(Fused, LAYER_FUSED);

}  //  namespace TNN_NS
