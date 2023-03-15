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

#include "coreml_base_layer.h"
#include "tnn/utils/data_type_utils.h"

namespace TNN_NS {

DECLARE_COREML_LAYER_WITH_FUNC_DATA(Instancenorm, LAYER_INST_BATCH_NORM,
                                virtual std::vector<CoreML__Specification__NeuralNetworkLayer*> GetCoreMLLayerPtrs();,
                                    std::shared_ptr<CoreMLBaseLayer> coreml_layer_groupnorm_;
                                    std::shared_ptr<LayerInfo> groupnorm_layer_info_;
                                    std::shared_ptr<RawBuffer> groupnorm_rawbuffer_scale_;
                                    std::shared_ptr<RawBuffer> groupnorm_rawbuffer_bias_;);

std::vector<CoreML__Specification__NeuralNetworkLayer*> CoreMLInstancenormLayer::GetCoreMLLayerPtrs() {
    auto all_ptrs = CoreMLBaseLayer::GetCoreMLLayerPtrs();
    if (coreml_layer_groupnorm_) {
        auto ptrs = coreml_layer_groupnorm_->GetCoreMLLayerPtrs();
        all_ptrs.insert(all_ptrs.end(), ptrs.begin(), ptrs.end());
    }
    return all_ptrs;
}

Status CoreMLInstancenormLayer::BuildLayerType() {
    //nullfy coreml_layer_, ortherwise GetCoreMLLayerPtrs will get wrong result
    coreml_layer_ = nullptr;
    return TNN_OK;
}

Status CoreMLInstancenormLayer::BuildLayerParam() {
    int input_shape_size = 0;
    if (net_resource_ && layer_info_->inputs.size()>0) {
        if (net_resource_->blob_shapes_map.find(layer_info_->inputs[0]) != net_resource_->blob_shapes_map.end()) {
            auto input_shape = net_resource_->blob_shapes_map[layer_info_->inputs[0]];
            input_shape_size = (int)input_shape.size();
        }
    }

    //layer param
    auto param = dynamic_cast<InstanceNormLayerParam *>(layer_info_->param.get());
    CHECK_PARAM_NULL(param);
    const int channels = param->channels;
    
    auto layer_res = dynamic_cast<InstanceNormLayerResource *>(layer_resource_);
    if (!layer_res || layer_res->scale_handle.GetDataCount()<=0 || layer_res->bias_handle.GetDataCount()<=0) {
        LOGE("CoreMLInstancenormLayer has invalid layer resource\n");
        return Status(TNNERR_MODEL_ERR, "CoreMLInstancenormLayer has invalid layer resource");
    }
    const int count_scale = layer_res->scale_handle.GetDataCount();
    const int count_bias = layer_res->bias_handle.GetDataCount();
    if (count_bias != channels || count_bias != channels) {
        LOGE("CoreMLInstancenormLayer has invalid layer resource\n");
        return Status(TNNERR_MODEL_ERR, "CoreMLInstancenormLayer has invalid layer resource");
    }
    
    coreml_layer_groupnorm_ = CreateCoreMLBaseLayer(LAYER_GROUP_NORM);
    if (coreml_layer_groupnorm_ == nullptr) {
        LOGE("Error: CreateCoreMLBaseLayer failed, dont support type:GrouoNorm\n");
        return Status(TNNERR_PARAM_ERR, "CreateCoreMLBaseLayer failed, dont support op LAYER_GROUP_NORM");
    }
    coreml_layer_groupnorm_->SetNetResource(net_resource_);
    
    //build groupnorm
    {
        //Note: groupnorm layer use const_resource_map to save weights inedead of layer resource
        //scale_name
        auto name_scale = layer_info_->name + "-groupnrom-scale";
        //Insure scale and bias have correct dims so the mul and add op in coreml_group_norm_layer can have right broadcast type
        DimsVector dims_scale;
        for (int i=0; i<input_shape_size-1; i++) {
            dims_scale.push_back(1);
        }
        dims_scale[0] = channels;
        if (layer_res->scale_handle.GetBufferDims().size() <= 0) {
            layer_res->scale_handle.SetBufferDims(dims_scale);
        }
        groupnorm_rawbuffer_scale_ = std::shared_ptr<RawBuffer>(new RawBuffer(layer_res->scale_handle));
        net_resource_->constant_map[name_scale] = groupnorm_rawbuffer_scale_;
        //scale_name
        auto name_bias = layer_info_->name + "-groupnrom-bias";
        if (layer_res->bias_handle.GetBufferDims().size() <= 0) {
            layer_res->bias_handle.SetBufferDims(dims_scale);
        }
        groupnorm_rawbuffer_bias_ = std::shared_ptr<RawBuffer>(new RawBuffer(layer_res->bias_handle));
        net_resource_->constant_map[name_bias] = groupnorm_rawbuffer_bias_;
        groupnorm_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
        auto groupnorm_param = std::shared_ptr<GroupNormLayerParam>(new GroupNormLayerParam);
        {
            groupnorm_layer_info_->type = LAYER_GROUP_NORM;
            groupnorm_layer_info_->name = layer_info_->name + "-groupnrom";
            groupnorm_layer_info_->inputs = {layer_info_->inputs[0], name_scale, name_bias};
            groupnorm_layer_info_->outputs = layer_info_->outputs;
            groupnorm_layer_info_->param = groupnorm_param;
            {
                groupnorm_param->type = groupnorm_layer_info_->type;
                groupnorm_param->name = groupnorm_layer_info_->name;
                groupnorm_param->group = param->channels;
                groupnorm_param->eps = param->eps;
            }
        }
        RETURN_ON_NEQ(coreml_layer_groupnorm_->Init(groupnorm_layer_info_.get(), nullptr),  TNN_OK);
    }
    return TNN_OK;
}

Status CoreMLInstancenormLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLInstancenormLayer::BuildLayerInputs() {
    return std::vector<std::string>();
}

std::vector<std::string> CoreMLInstancenormLayer::BuildLayerOutputs() {
    return std::vector<std::string>();
}

REGISTER_COREML_LAYER(Instancenorm, LAYER_INST_BATCH_NORM);

}  // namespace TNN_NS
