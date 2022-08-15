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

// Use ReshapeStatic

namespace TNN_NS {

DECLARE_COREML_LAYER_WITH_FUNC_DATA(Reshape, LAYER_RESHAPE,
                                     virtual Status BuildPermute0Layer();
                                     virtual Status BuildPermute1Layer();
                                     bool IsDynamic();,
                                     std::shared_ptr<void> coreml_layer_shape_;
                                     std::shared_ptr<CoreML__Specification__Tensor*> coreml_layer_inputtensor_arr_;
                                     std::vector<std::shared_ptr<CoreML__Specification__Tensor> > coreml_layer_inputtensor_;
                                     std::shared_ptr<CoreML__Specification__Tensor*> coreml_layer_outputtensor_arr_;
                                     std::vector<std::shared_ptr<CoreML__Specification__Tensor> > coreml_layer_outputtensor_;
                                     int input_shape_size_ = 0;
                                     int output_shape_size_ = 0;
                                     std::shared_ptr<LayerInfo> permute0_layer_info_;
                                     std::shared_ptr<LayerInfo> permute1_layer_info_;);

bool CoreMLReshapeLayer::IsDynamic() {
    if (layer_info_ && layer_info_->inputs.size() >= 2 && net_resource_) {
        auto shape_name = layer_info_->inputs[1];
        if (net_resource_->constant_blob_flags.find(shape_name) != net_resource_->constant_blob_flags.end()) {
            auto blob_flag = net_resource_->constant_blob_flags[shape_name];
            if (blob_flag == DATA_FLAG_CHANGE_NEVER) {
                return false;
            } else {
                return true;
            }
        } else {
            return true;
        }
    }
    return false;
}

Status CoreMLReshapeLayer::BuildLayerType() {
    //layer type
    if (IsDynamic()) {
        coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_RESHAPE_DYNAMIC;
    } else {
        coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_RESHAPE_STATIC;
    }
    
    return TNN_OK;
}

Status CoreMLReshapeLayer::BuildLayerParam() {
    
    std::vector<int> input_shape,output_shape;
    if (net_resource_ && layer_info_->inputs.size()>0 && layer_info_->outputs.size()>0) {
        if (net_resource_->blob_shapes_map.find(layer_info_->inputs[0]) != net_resource_->blob_shapes_map.end()) {
            input_shape = net_resource_->blob_shapes_map[layer_info_->inputs[0]];
            input_shape_size_ = (int)input_shape.size();
        }
        
        if (net_resource_->blob_shapes_map.find(layer_info_->outputs[0]) != net_resource_->blob_shapes_map.end()) {
            output_shape = net_resource_->blob_shapes_map[layer_info_->outputs[0]];
            output_shape_size_ = (int)output_shape.size();
        }
    }
    
    //reshape mode dynamic
    if (IsDynamic()) {
        coreml_layer_param_ = std::shared_ptr<CoreML__Specification__ReshapeDynamicLayerParams>(new CoreML__Specification__ReshapeDynamicLayerParams);
        coreml_layer_->reshapedynamic = (CoreML__Specification__ReshapeDynamicLayerParams *)coreml_layer_param_.get();
        core_ml__specification__reshape_dynamic_layer_params__init(coreml_layer_->reshapedynamic);
    } else {
        //reshape mode default
        //layer param
        auto param = layer_info_->param.get();
        auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
        CHECK_PARAM_NULL(reshape_param);
        auto shape_size = reshape_param->shape.size();
        auto shape = reshape_param->shape;
        // reshape_type:
        // onnx caffe reshape(nchw): 0
        // Tensorflow TFLite reshape(nhwc): 1
        auto reshape_type = reshape_param->reshape_type;
        if (output_shape_size_ <= 0) {
            output_shape_size_ = (int)shape.size();
        }
    
        if ((reshape_type == 1 && input_shape_size_ <= 0) || output_shape_size_ <= 0 || shape_size != output_shape_size_) {
            return Status(TNNERR_MODEL_ERR, "CoreMLReshapeLayer has invalid input shape, output shape, or ReshapeLayerParam");
        }
        
        // add permute to convert nchw to nhwc, when reshape_type = 1
        if (reshape_type == 1) {
            if(input_shape_size_ > 4 || output_shape_size_ > 4) {
                return Status(TNNERR_MODEL_ERR, "CoreMLReshapeLayer input rank and output rank must be smaller or equal to 4 , when reshape_type = 1");
            }
            RETURN_ON_NEQ(BuildPermute0Layer(), TNN_OK);
            RETURN_ON_NEQ(BuildPermute1Layer(), TNN_OK);
        }
        
        coreml_layer_param_ = std::shared_ptr<CoreML__Specification__ReshapeStaticLayerParams>(new CoreML__Specification__ReshapeStaticLayerParams);
        coreml_layer_->reshapestatic = (CoreML__Specification__ReshapeStaticLayerParams *)coreml_layer_param_.get();
        core_ml__specification__reshape_static_layer_params__init(coreml_layer_->reshapestatic);
        coreml_layer_->reshapestatic->n_targetshape = output_shape_size_;
        coreml_layer_shape_ = std::shared_ptr<int64_t>(new int64_t [coreml_layer_->reshapestatic->n_targetshape], [](int64_t* p) { delete[] p; });
        coreml_layer_->reshapestatic->targetshape = (int64_t *)coreml_layer_shape_.get();
        
        if (reshape_type == 1) {
            std::vector<int> output_shape_permute;
            if(output_shape_size_ == 4) {
                output_shape_permute = {output_shape[0],output_shape[2],output_shape[3],output_shape[1]};
            } else if(output_shape_size_ == 3) {
                output_shape_permute = {output_shape[0],output_shape[2],output_shape[1]};
            } else if(output_shape_size_ == 2) {
                output_shape_permute = {output_shape[0],output_shape[1]};
            }
            for (int i=0;i<output_shape_size_;i++){
                coreml_layer_->reshapestatic->targetshape[i] = output_shape_permute[i];
            }
        } else {
            //Note reshape static layer cannot handle 0 or -1
            if (output_shape.size() == output_shape_size_) {
                for (int i=0;i<output_shape_size_;i++){
                    coreml_layer_->reshapestatic->targetshape[i] = output_shape[i];
                }
            } else {
                for (int i=0;i<output_shape_size_;i++){
                    coreml_layer_->reshapestatic->targetshape[i] = shape[i];
                }
            }
        }
    }
    
    return TNN_OK;
}

Status CoreMLReshapeLayer::BuildPermute0Layer() {
    auto param = layer_info_->param.get();
    auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
    auto permute0_layer = CreateCoreMLBaseLayer(LAYER_PERMUTE);
    permute0_layer->SetNetResource(net_resource_);
    permute0_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
    {
        permute0_layer_info_->type = LAYER_PERMUTE;
        permute0_layer_info_->name = reshape_param->name + "-permute0";
        permute0_layer_info_->inputs = layer_info_->inputs;
        permute0_layer_info_->outputs = {reshape_param->name + "-permute0-out"};
        auto permute0_param = std::shared_ptr<PermuteLayerParam>(new PermuteLayerParam);
        permute0_layer_info_->param = permute0_param;
        {
            if(input_shape_size_ == 4) {
                permute0_param->orders = {0,2,3,1};  // nchw2nhwc
            } else if(input_shape_size_ == 3) {
                permute0_param->orders = {0,2,1};  // nch2nhc
            } else if(input_shape_size_ == 2) {
                permute0_param->orders = {0,1};  // nc2nc
            }
        }
    }
    RETURN_ON_NEQ(permute0_layer->Init(permute0_layer_info_.get(), nullptr), TNN_OK);
    coreml_layers_before_ = {permute0_layer};
    
    return TNN_OK;
}

Status CoreMLReshapeLayer::BuildPermute1Layer() {
    auto param = layer_info_->param.get();
    auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
    auto permute1_layer = CreateCoreMLBaseLayer(LAYER_PERMUTE);
    permute1_layer->SetNetResource(net_resource_);
    permute1_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
    {
        permute1_layer_info_->type = LAYER_PERMUTE;
        permute1_layer_info_->name = reshape_param->name + "-permute1";
        permute1_layer_info_->inputs = {reshape_param->name + "-permute1-in"};
        permute1_layer_info_->outputs = layer_info_->outputs;
        auto permute1_param = std::shared_ptr<PermuteLayerParam>(new PermuteLayerParam);
        permute1_layer_info_->param = permute1_param;
        {
            if(output_shape_size_ == 4) {
                permute1_param->orders = {0,3,1,2};  // nhwc2nchw
            } else if(output_shape_size_ == 3) {
                permute1_param->orders = {0,2,1};  // nhc2nch
            } else if(output_shape_size_ == 2) {
                permute1_param->orders = {0,1};  // nc2nc
            }
        }
    }
    RETURN_ON_NEQ(permute1_layer->Init(permute1_layer_info_.get(), nullptr), TNN_OK);
    coreml_layers_after_ = {permute1_layer};
    
    return TNN_OK;
}

Status CoreMLReshapeLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLReshapeLayer::BuildLayerInputs() {
    if (!layer_info_) {
        return std::vector<std::string>();
    } else {
        if (IsDynamic()) {
            return CoreMLBaseLayer::BuildLayerInputs();
        } else {
            auto param = layer_info_->param.get();
            auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
            if(reshape_param->reshape_type == 1) {
                return {reshape_param->name + "-permute0-out"};
            }
            return {layer_info_->inputs[0]};
        }
    }
}

std::vector<std::string> CoreMLReshapeLayer::BuildLayerOutputs() {
    if (!layer_info_) {
        return std::vector<std::string>();
    } else {
        if (IsDynamic()) {
            return CoreMLBaseLayer::BuildLayerOutputs();
        } else {
            auto param = layer_info_->param.get();
            auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
            if(reshape_param->reshape_type == 1) {
                return {reshape_param->name + "-permute1-in"};
            }
            return layer_info_->outputs;
        }
    }
}

REGISTER_COREML_LAYER(Reshape, LAYER_RESHAPE);

}  // namespace TNN_NS


// Use ReshapeRankPreserving
/*
 namespace TNN_NS {

 DECLARE_COREML_LAYER_WITH_FUNC_DATA(Reshape, LAYER_RESHAPE,
                                      virtual Status BuildPermute0Layer();
                                      virtual Status BuildPermute1Layer();
                                      virtual Status BuildSqueezeLayer();
                                      virtual Status BuildUnsqueezeLayer();
                                      bool IsDynamic();,
                                      std::shared_ptr<void> coreml_layer_shape_;
                                      std::shared_ptr<CoreML__Specification__Tensor*> coreml_layer_inputtensor_arr_;
                                      std::vector<std::shared_ptr<CoreML__Specification__Tensor> > coreml_layer_inputtensor_;
                                      std::shared_ptr<CoreML__Specification__Tensor*> coreml_layer_outputtensor_arr_;
                                      std::vector<std::shared_ptr<CoreML__Specification__Tensor> > coreml_layer_outputtensor_;
                                      int input_shape_size_ = 0;
                                      int output_shape_size_ = 0;
                                      std::shared_ptr<LayerInfo> permute0_layer_info_;
                                      std::shared_ptr<LayerInfo> permute1_layer_info_;
                                      std::shared_ptr<LayerInfo> unsqueeze_layer_info_;
                                     std::shared_ptr<LayerInfo> squeeze_layer_info_;);

 bool CoreMLReshapeLayer::IsDynamic() {
     if (layer_info_ && layer_info_->inputs.size() >= 2 && net_resource_ &&
         net_resource_->constant_map.find(layer_info_->inputs[1]) == net_resource_->constant_map.end()) {
         return true;
     }
     return false;
 }

 Status CoreMLReshapeLayer::BuildLayerType() {
     //layer type
     if (IsDynamic()) {
         coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_RESHAPE_DYNAMIC;
     } else {
         coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_RANK_PRESERVING_RESHAPE;
     }
     
     return TNN_OK;
 }

 Status CoreMLReshapeLayer::BuildLayerParam() {
     if (net_resource_ && layer_info_->inputs.size()>0 && layer_info_->outputs.size()>0) {
         if (net_resource_->blob_shapes_map.find(layer_info_->inputs[0]) != net_resource_->blob_shapes_map.end()) {
             auto input_shape = net_resource_->blob_shapes_map[layer_info_->inputs[0]];
             input_shape_size_ = (int)input_shape.size();
         }
         
         if (net_resource_->blob_shapes_map.find(layer_info_->outputs[0]) != net_resource_->blob_shapes_map.end()) {
             auto output_shape = net_resource_->blob_shapes_map[layer_info_->outputs[0]];
             output_shape_size_ = (int)output_shape.size();
         }
     }
     
     //reshape mode dynamic
     if (IsDynamic()) {
         coreml_layer_param_ = std::shared_ptr<CoreML__Specification__ReshapeDynamicLayerParams>(new CoreML__Specification__ReshapeDynamicLayerParams);
         coreml_layer_->reshapedynamic = (CoreML__Specification__ReshapeDynamicLayerParams *)coreml_layer_param_.get();
         core_ml__specification__reshape_dynamic_layer_params__init(coreml_layer_->reshapedynamic);
     } else {
         //reshape mode default
         //layer param
         auto param = layer_info_->param.get();
         auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
         CHECK_PARAM_NULL(reshape_param);
         auto input_size = 1;
         auto output_size = 1;
         auto shape_size = reshape_param->shape.size();
         auto shape = reshape_param->shape;
         // reshape_type:
         // onnx caffe reshape(nchw): 0
         // Tensorflow TFLite reshape(nhwc): 1
         auto reshape_type = reshape_param->reshape_type;
     
         if (input_shape_size_ <= 0 || output_shape_size_ <= 0 || shape_size != output_shape_size_) {
             return Status(TNNERR_MODEL_ERR, "CoreMLReshapeLayer has invalid input shape, output shape, or ReshapeLayerParam");
         }
         
         // add permute to convert nchw to nhwc, when reshape_type = 1
         if (reshape_type == 1) {
             if ((input_shape_size_!=4) || (output_shape_size_!=4)) {
                 return Status(TNNERR_MODEL_ERR, "CoreMLReshapeLayer input rank and output rank must be equal to 4 , when reshape_type = 1");
             }
             RETURN_ON_NEQ(BuildPermute0Layer(), TNN_OK);
             RETURN_ON_NEQ(BuildPermute1Layer(), TNN_OK);
         }
         
         // add unsqueeze to expenad dims
         int reshape_size = MAX(output_shape_size_, input_shape_size_);
         if (output_shape_size_  > input_shape_size_) {
             RETURN_ON_NEQ(BuildUnsqueezeLayer(), TNN_OK);
         }
         
         coreml_layer_param_ = std::shared_ptr<CoreML__Specification__RankPreservingReshapeLayerParams>(new CoreML__Specification__RankPreservingReshapeLayerParams);
         coreml_layer_->rankpreservingreshape = (CoreML__Specification__RankPreservingReshapeLayerParams *)coreml_layer_param_.get();
         core_ml__specification__rank_preserving_reshape_layer_params__init(coreml_layer_->rankpreservingreshape);
         coreml_layer_->rankpreservingreshape->n_targetshape = reshape_size;
         coreml_layer_shape_ = std::shared_ptr<int64_t>(new int64_t [coreml_layer_->rankpreservingreshape->n_targetshape], [](int64_t* p) { delete[] p; });
         coreml_layer_->rankpreservingreshape->targetshape = (int64_t *)coreml_layer_shape_.get();
         if(output_shape_size_  < input_shape_size_){
             auto reduce_dims = input_shape_size_ - output_shape_size_;
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
         if (output_shape_size_  < input_shape_size_) {
             RETURN_ON_NEQ(BuildSqueezeLayer(), TNN_OK);
         }
     }
     
     return TNN_OK;
 }

 Status CoreMLReshapeLayer::BuildPermute0Layer() {
     auto param = layer_info_->param.get();
     auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
     auto permute0_layer = CreateCoreMLBaseLayer(LAYER_PERMUTE);
     permute0_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
     {
         permute0_layer_info_->type = LAYER_PERMUTE;
         permute0_layer_info_->name = reshape_param->name + "-permute0";
         permute0_layer_info_->inputs = layer_info_->inputs;
         permute0_layer_info_->outputs = {reshape_param->name + "-permute0-out"};
         auto permute0_param = std::shared_ptr<PermuteLayerParam>(new PermuteLayerParam);
         permute0_layer_info_->param = permute0_param;
         {
             std::vector<int> orders_ = {0,2,3,1};  // nchw2nhwc
             permute0_param->orders = orders_;
         }
     }
     RETURN_ON_NEQ(permute0_layer->Init(permute0_layer_info_.get(), nullptr), TNN_OK);
     coreml_layer_before_ = permute0_layer;
     
     return TNN_OK;
 }

 Status CoreMLReshapeLayer::BuildPermute1Layer() {
     auto param = layer_info_->param.get();
     auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
     auto permute1_layer = CreateCoreMLBaseLayer(LAYER_PERMUTE);
     permute1_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
     {
         permute1_layer_info_->type = LAYER_PERMUTE;
         permute1_layer_info_->name = reshape_param->name + "-permute1";
         permute1_layer_info_->inputs = {reshape_param->name + "-permute1-in"};
         permute1_layer_info_->outputs = layer_info_->outputs;
         auto permute1_param = std::shared_ptr<PermuteLayerParam>(new PermuteLayerParam);
         permute1_layer_info_->param = permute1_param;
         {
             std::vector<int> orders_ = {0,3,1,2};  // nhwc2nchw
             permute1_param->orders = orders_;
         }
     }
     RETURN_ON_NEQ(permute1_layer->Init(permute1_layer_info_.get(), nullptr), TNN_OK);
     coreml_layer_after_ = permute1_layer;
     
     return TNN_OK;
 }

 Status CoreMLReshapeLayer::BuildSqueezeLayer() {
     auto param = layer_info_->param.get();
     auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
     auto squeeze_layer = CreateCoreMLBaseLayer(LAYER_SQUEEZE);
     squeeze_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
     {
         squeeze_layer_info_->type = LAYER_SQUEEZE;
         squeeze_layer_info_->name = reshape_param->name + "-squeeze";
         squeeze_layer_info_->inputs = {reshape_param->name + "-squeeze-in"};
         squeeze_layer_info_->outputs = layer_info_->outputs;
         auto squeeze_param = std::shared_ptr<SqueezeLayerParam>(new SqueezeLayerParam);
         squeeze_layer_info_->param = squeeze_param;
         {
             std::vector<int> axes = {};
             auto  reduce_dims = input_shape_size_ - output_shape_size_;
             for(int i=0;i<reduce_dims;i++){
                 axes.push_back(i-reduce_dims);
             }
             squeeze_param->axes = axes;
         }
     }
     RETURN_ON_NEQ(squeeze_layer->Init(squeeze_layer_info_.get(), nullptr), TNN_OK);
     coreml_layer_after_ = squeeze_layer;
     
     return TNN_OK;
 }

 Status CoreMLReshapeLayer::BuildUnsqueezeLayer() {
     auto param = layer_info_->param.get();
     auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
     auto unsqueeze_layer = CreateCoreMLBaseLayer(LAYER_UNSQUEEZE);
     unsqueeze_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
     {
         unsqueeze_layer_info_->type = LAYER_UNSQUEEZE;
         unsqueeze_layer_info_->name = reshape_param->name + "-unsqueeze";
         unsqueeze_layer_info_->inputs = layer_info_->inputs;
         unsqueeze_layer_info_->outputs =  {reshape_param->name + "-unsqueeze-out"};
         auto unsqueeze_param = std::shared_ptr<UnsqueezeLayerParam>(new UnsqueezeLayerParam);
         unsqueeze_layer_info_->param = unsqueeze_param;
         {
             std::vector<int> axes = {};
             auto expand_dims = output_shape_size_ - input_shape_size_;
             for(int i=0;i<expand_dims;i++){
                 axes.push_back(i-expand_dims);
             }
             unsqueeze_param->axes = axes;
         }
     }
     RETURN_ON_NEQ(unsqueeze_layer->Init(unsqueeze_layer_info_.get(), nullptr), TNN_OK);
     coreml_layer_before_ = unsqueeze_layer;

     return TNN_OK;
 }

 Status CoreMLReshapeLayer::BuildConstantWeightsLayer() {
     return CoreMLBaseLayer::BuildConstantWeightsLayer();
 }

 std::vector<std::string> CoreMLReshapeLayer::BuildLayerInputs() {
     if (!layer_info_) {
         return std::vector<std::string>();
     } else {
         if (IsDynamic()) {
             return CoreMLBaseLayer::BuildLayerInputs();
         } else {
             auto param = layer_info_->param.get();
             auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
             if(output_shape_size_  > input_shape_size_) {
                 return {reshape_param->name + "-unsqueeze-out"};
             }
             if(reshape_param->reshape_type == 1) {
                 return {reshape_param->name + "-permute0-out"};
             }
             return {layer_info_->inputs[0]};
         }
     }
 }

 std::vector<std::string> CoreMLReshapeLayer::BuildLayerOutputs() {
     if (!layer_info_) {
         return std::vector<std::string>();
     } else {
         if (IsDynamic()) {
             return CoreMLBaseLayer::BuildLayerOutputs();
         } else {
             auto param = layer_info_->param.get();
             auto reshape_param = dynamic_cast<ReshapeLayerParam *>(param);
             if(output_shape_size_  < input_shape_size_) {
                 return {reshape_param->name + "-squeeze-in"};
             }
             if(reshape_param->reshape_type == 1) {
                 return {reshape_param->name + "-permute1-in"};
             }
             return layer_info_->outputs;
         }
     }
 }

 REGISTER_COREML_LAYER(Reshape, LAYER_RESHAPE);

 }  // namespace TNN_NS
*/
