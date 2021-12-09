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
#include "coreml_unsqueeze_layer.h"
#include "coreml_squeeze_layer.h"

namespace TNN_NS {

DECLARE_COREML_LAYER_WITH_FUNC_DATA(Reshape, LAYER_RESHAPE,
                                     virtual Status BuildSqueezeLayer();
                                     virtual Status BuildUnsqueezeLayer();,
                                     std::shared_ptr<void> coreml_layer_shape_;
                                     std::shared_ptr<CoreML__Specification__Tensor*> coreml_layer_inputtensor_arr_;
                                     std::vector<std::shared_ptr<CoreML__Specification__Tensor> > coreml_layer_inputtensor_;
                                     std::shared_ptr<CoreML__Specification__Tensor*> coreml_layer_outputtensor_arr_;
                                     std::vector<std::shared_ptr<CoreML__Specification__Tensor> > coreml_layer_outputtensor_;
                                     int input_shape_size = 0;
                                     int output_shape_size = 0;
                                     std::shared_ptr<LayerInfo> unsqueeze_layer_info_;
                                     std::shared_ptr<UnsqueezeLayerParam> unsqueeze_param_;
                                     std::shared_ptr<LayerInfo> squeeze_layer_info_;
                                     std::shared_ptr<SqueezeLayerParam> squeeze_param_;);

Status CoreMLReshapeLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_RANK_PRESERVING_RESHAPE;
    return TNN_OK;
}

Status CoreMLReshapeLayer::BuildLayerParam() {
   
    if (net_resource_ && layer_info_->inputs.size()>0 && layer_info_->outputs.size()>0) {
        if (net_resource_->blob_shapes_map.find(layer_info_->inputs[0]) != net_resource_->blob_shapes_map.end()) {
            auto input_shape = net_resource_->blob_shapes_map[layer_info_->inputs[0]];
            input_shape_size = (int)input_shape.size();
        }
        
        if (net_resource_->blob_shapes_map.find(layer_info_->outputs[0]) != net_resource_->blob_shapes_map.end()) {
            auto output_shape = net_resource_->blob_shapes_map[layer_info_->outputs[0]];
            output_shape_size = (int)output_shape.size();
        }
    }
    
    //layer param
    auto param = layer_info_->param.get();
    auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
    CHECK_PARAM_NULL(reshape_param);
    auto input_size = layer_info_->inputs.size();
    auto output_size = layer_info_->outputs.size();
    auto shape_size = reshape_param->shape.size();
    auto shape = reshape_param->shape;
    if (input_shape_size <= 0 || output_shape_size <= 0 || shape_size != output_shape_size) {
        return Status(TNNERR_MODEL_ERR, "CoreMLReshapeLayer has invalid input shape, output shape, or ReshapeLayerParam");
    }
    
    // add unsqueeze to expenad dims
    int reshape_size = MAX(output_shape_size, input_shape_size);
    if (output_shape_size  > input_shape_size) {
        RETURN_ON_NEQ(BuildUnsqueezeLayer(), TNN_OK);
    }
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__RankPreservingReshapeLayerParams>(new CoreML__Specification__RankPreservingReshapeLayerParams);
    coreml_layer_->rankpreservingreshape = (CoreML__Specification__RankPreservingReshapeLayerParams *)coreml_layer_param_.get();
    core_ml__specification__rank_preserving_reshape_layer_params__init(coreml_layer_->rankpreservingreshape);
    coreml_layer_->rankpreservingreshape->n_targetshape = reshape_size;
    coreml_layer_shape_ = std::shared_ptr<int64_t>(new int64_t [coreml_layer_->rankpreservingreshape->n_targetshape], [](int64_t* p) { delete[] p; });
    coreml_layer_->rankpreservingreshape->targetshape = (int64_t *)coreml_layer_shape_.get();
    if(output_shape_size  < input_shape_size){
        auto reduce_dims = input_shape_size - output_shape_size;
        for(int i=0;i<reduce_dims;i++){
            shape.push_back(1);
        }
    }
    for(int i=0; i<reshape_size; i++){
        coreml_layer_->rankpreservingreshape->targetshape[i] = shape[i];
    }
    
    //input & output rank must be equal!
    //set inputtensor rank
    coreml_layer_->n_inputtensor = input_size;
    coreml_layer_inputtensor_arr_ = std::shared_ptr<CoreML__Specification__Tensor*>(new CoreML__Specification__Tensor* [input_size], [](CoreML__Specification__Tensor** p) { delete[] p; });
    coreml_layer_->inputtensor = coreml_layer_inputtensor_arr_.get();
    for(int i=0; i<input_size; i++){
        coreml_layer_inputtensor_.push_back(std::shared_ptr<CoreML__Specification__Tensor>(new CoreML__Specification__Tensor));
        coreml_layer_->inputtensor[i] = coreml_layer_inputtensor_[i].get();
        core_ml__specification__tensor__init(coreml_layer_->inputtensor[i]);
        coreml_layer_->inputtensor[i]->rank = (uint32_t)reshape_size;
    }
    
    //set outputtensor rank
    coreml_layer_->n_outputtensor = output_size;
    coreml_layer_outputtensor_arr_ = std::shared_ptr<CoreML__Specification__Tensor*>(new CoreML__Specification__Tensor* [output_size], [](CoreML__Specification__Tensor** p) { delete[] p; });
    coreml_layer_->outputtensor = coreml_layer_outputtensor_arr_.get();
    for(int i=0; i<output_size; i++){
        coreml_layer_outputtensor_.push_back(std::shared_ptr<CoreML__Specification__Tensor>(new CoreML__Specification__Tensor));
        coreml_layer_->outputtensor[i] = coreml_layer_outputtensor_[i].get();
        core_ml__specification__tensor__init(coreml_layer_->outputtensor[i]);
        coreml_layer_->outputtensor[i]->rank = (uint32_t)reshape_size;
    }
    
    // add squeeze to reduce dims
    if (output_shape_size  < input_shape_size) {
        RETURN_ON_NEQ(BuildSqueezeLayer(), TNN_OK);
    }
    
    return TNN_OK;
}

Status CoreMLReshapeLayer::BuildSqueezeLayer() {
    auto param = layer_info_->param.get();
    auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
    auto squeeze = std::make_shared<CoreMLSqueezeLayer>(LAYER_SQUEEZE);
    squeeze_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto layer_info = squeeze_layer_info_.get();
    layer_info->type = LAYER_SQUEEZE;
    layer_info->name = reshape_param->name + "-squeeze";
    layer_info->inputs = {reshape_param->name + "-squeeze-in"};
    layer_info->outputs = layer_info_->outputs;
    squeeze_param_ = std::shared_ptr<SqueezeLayerParam>(new SqueezeLayerParam);
    layer_info->param = squeeze_param_;
    auto squeeze_param = squeeze_param_.get();
    std::vector<int> axes = {};
    auto  reduce_dims = input_shape_size - output_shape_size;
    for(int i=0;i<reduce_dims;i++){
        axes.push_back(i-reduce_dims);
    }
    squeeze_param->axes = axes;
    RETURN_ON_NEQ(squeeze->Init(layer_info, nullptr), TNN_OK);
    coreml_layer_squeeze_ = {squeeze};
    
    return TNN_OK;
}

Status CoreMLReshapeLayer::BuildUnsqueezeLayer() {
    auto param = layer_info_->param.get();
    auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
    auto unsqueeze = std::make_shared<CoreMLUnsqueezeLayer>(LAYER_UNSQUEEZE);
    unsqueeze_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
    auto layer_info = unsqueeze_layer_info_.get();
    layer_info->type = LAYER_UNSQUEEZE;
    layer_info->name = reshape_param->name + "-unsqueeze";
    layer_info->inputs = layer_info_->inputs;
    layer_info->outputs =  {reshape_param->name + "-unsqueeze-out"};
    unsqueeze_param_ = std::shared_ptr<UnsqueezeLayerParam>(new UnsqueezeLayerParam);
    layer_info->param = unsqueeze_param_;
    auto unsqueeze_param = unsqueeze_param_.get();
    std::vector<int> axes = {};
    auto expand_dims = output_shape_size - input_shape_size;
    for(int i=0;i<expand_dims;i++){
        axes.push_back(i-expand_dims);
    }
    unsqueeze_param->axes = axes;
    RETURN_ON_NEQ(unsqueeze->Init(layer_info, nullptr), TNN_OK);
    coreml_layer_unsqueeze_ = {unsqueeze};

    return TNN_OK;
}

Status CoreMLReshapeLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLReshapeLayer::BuildLayerInputs() {
    if (!layer_info_) {
        return std::vector<std::string>();
    } else {
        if(output_shape_size  > input_shape_size) {
            auto param = layer_info_->param.get();
            auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
            return {reshape_param->name + "-unsqueeze-out"};
        }
        return layer_info_->inputs;
    }
}

std::vector<std::string> CoreMLReshapeLayer::BuildLayerOutputs() {
    if (!layer_info_) {
        return std::vector<std::string>();
    } else {
        if(output_shape_size  < input_shape_size) {
            auto param = layer_info_->param.get();
            auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
            return {reshape_param->name + "-squeeze-in"};
        }
        return layer_info_->outputs;
    }
}

REGISTER_COREML_LAYER(Reshape, LAYER_RESHAPE);

}  // namespace TNN_NS
