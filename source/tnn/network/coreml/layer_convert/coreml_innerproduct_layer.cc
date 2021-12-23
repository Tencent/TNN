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

namespace TNN_NS {

DECLARE_COREML_LAYER_WITH_FUNC_DATA(Innerproduct, LAYER_INNER_PRODUCT,
                                    virtual Status BuildSqueezeLayer();,
                                    std::shared_ptr<CoreML__Specification__WeightParams> bias_param_;
                                    std::shared_ptr<CoreML__Specification__WeightParams> weight_param_;
                                    std::shared_ptr<LayerInfo> squeeze_layer_info_;
                                    int input_shape_size = 0;
                                    int output_shape_size = 0;
                                    std::shared_ptr<float> weight_fp32_ptr_;
                                    std::shared_ptr<float> bias_fp32_ptr_;);

Status CoreMLInnerproductLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_INNER_PRODUCT;
    return TNN_OK;
}

Status CoreMLInnerproductLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto layer_param = dynamic_cast<InnerProductLayerParam *>(param);
    CHECK_PARAM_NULL(layer_param);
    auto num_output = layer_param->num_output;
    auto has_bias = layer_param->has_bias;
    auto transpose = layer_param->transpose;
    auto axis = layer_param->axis;
    
    auto layer_res = dynamic_cast<InnerProductLayerResource *>(layer_resource_);
    CHECK_PARAM_NULL(layer_res);
    auto weight_size = layer_res->weight_handle.GetDataCount();
    auto weight_type = layer_res->weight_handle.GetDataType();
    auto weight_dims = layer_res->weight_handle.GetBufferDims();
    auto bias_size = layer_res->bias_handle.GetDataCount();
    auto bias_type = layer_res->bias_handle.GetDataType();
    
    std::vector<int> input_shape;
    std::vector<int> output_shape;
    if (net_resource_ && layer_info_->inputs.size()>0 && layer_info_->outputs.size()>0) {
        if (net_resource_->blob_shapes_map.find(layer_info_->inputs[0]) != net_resource_->blob_shapes_map.end()) {
            input_shape = net_resource_->blob_shapes_map[layer_info_->inputs[0]];
            input_shape_size = (int)input_shape.size();
        }
        
        if (net_resource_->blob_shapes_map.find(layer_info_->outputs[0]) != net_resource_->blob_shapes_map.end()) {
            output_shape = net_resource_->blob_shapes_map[layer_info_->outputs[0]];
            output_shape_size = (int)output_shape.size();
        }
    }
    
    // in order to match old TNN model, add squeeze to reduce dims
    if(input_shape_size > output_shape_size) {
        RETURN_ON_NEQ(BuildSqueezeLayer(), TNN_OK);
    }
    
    int input_dims;
    if(weight_dims.size() == 0){
        if (input_shape.size() <= 0) {
            return Status(TNNERR_MODEL_ERR, "CoreMLInnerproductLayer has invalid input shape");
        }
        if(input_shape_size > output_shape_size) {
            auto reduce_dims = input_shape_size - output_shape_size;
            input_dims = input_shape[input_shape_size - 1 - reduce_dims];
        } else {
            input_dims = input_shape.back();
        }
    } else {
        input_dims = weight_dims.back();
    }
   
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__InnerProductLayerParams>(new CoreML__Specification__InnerProductLayerParams);
    coreml_layer_->innerproduct = (CoreML__Specification__InnerProductLayerParams *)coreml_layer_param_.get();
    core_ml__specification__inner_product_layer_params__init(coreml_layer_->innerproduct);
    coreml_layer_->innerproduct->inputchannels = input_dims;
    coreml_layer_->innerproduct->outputchannels = num_output;
    
    weight_param_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
    coreml_layer_->innerproduct->weights = weight_param_.get();
        core_ml__specification__weight_params__init(coreml_layer_->innerproduct->weights);
    switch (weight_type) {
        case DATA_TYPE_FLOAT:
            coreml_layer_->innerproduct->weights->n_floatvalue = weight_size;
            coreml_layer_->innerproduct->weights->floatvalue = layer_res->weight_handle.force_to<float *>();
            break;
        case DATA_TYPE_HALF:
            {
#if TNN_COREML_FULL_PRECISION
                coreml_layer_->innerproduct->weights->n_floatvalue = weight_size;
                void *weight_data_ptr = layer_res->weight_handle.force_to<void *>();
                weight_fp32_ptr_ = std::shared_ptr<float>(new float [weight_size], [](float* p) { delete[] p; });
                auto weight_fp32_ptr = weight_fp32_ptr_.get();
                RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)weight_data_ptr, (float *)weight_fp32_ptr, weight_size),TNN_OK);
                coreml_layer_->innerproduct->weights->floatvalue = weight_fp32_ptr;
#else
                coreml_layer_->innerproduct->weights->float16value.len = layer_res->weight_handle.GetBytesSize();
                coreml_layer_->innerproduct->weights->float16value.data = layer_res->weight_handle.force_to<uint8_t *>();
#endif
            }
            break;
        default:
            LOGE("CoreMLInnerproductLayer dont support data type (%d)\n", weight_type);
            return Status(TNNERR_MODEL_ERR, "CoreMLConvLayer dont support this weight data type");
            break;
    }
    if(bias_size) {
        coreml_layer_->innerproduct->hasbias = true;
        bias_param_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
        coreml_layer_->innerproduct->bias = bias_param_.get();
        core_ml__specification__weight_params__init(coreml_layer_->innerproduct->bias);
        switch (bias_type) {
            case DATA_TYPE_FLOAT:
                coreml_layer_->innerproduct->bias->n_floatvalue = bias_size;
                coreml_layer_->innerproduct->bias->floatvalue = layer_res->bias_handle.force_to<float *>();
                break;
            case DATA_TYPE_HALF:
                {
#if TNN_COREML_FULL_PRECISION
                    coreml_layer_->innerproduct->bias->n_floatvalue = bias_size;
                    void *bias_data_ptr = layer_res->bias_handle.force_to<void *>();
                    bias_fp32_ptr_ = std::shared_ptr<float>(new float [bias_size], [](float* p) { delete[] p; });
                    auto bias_fp32_ptr = bias_fp32_ptr_.get();
                    RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)bias_data_ptr, (float *)bias_fp32_ptr, bias_size),TNN_OK);
                    coreml_layer_->innerproduct->bias->floatvalue = bias_fp32_ptr;
#else
                    coreml_layer_->innerproduct->bias->float16value.len = layer_res->bias_handle.GetBytesSize();
                    coreml_layer_->innerproduct->bias->float16value.data = layer_res->bias_handle.force_to<uint8_t *>();
#endif
                }
                break;
            default:
                LOGE("CoreMLInnerproductLayer dont support data type (%d)\n", bias_type);
                return Status(TNNERR_MODEL_ERR, "CoreMLConvLayer dont support this bias data type");
                break;
        }
    }
    
    return TNN_OK;
}

Status CoreMLInnerproductLayer::BuildSqueezeLayer() {
    auto param = layer_info_->param.get();
    auto innerproduct_param = dynamic_cast<InnerProductLayerParam *>(param);
    auto squeeze_layer = CreateCoreMLBaseLayer(LAYER_SQUEEZE);
    squeeze_layer_info_ = std::shared_ptr<LayerInfo>(new LayerInfo);
    {
        squeeze_layer_info_->type = LAYER_SQUEEZE;
        squeeze_layer_info_->name = innerproduct_param->name + "-squeeze";
        squeeze_layer_info_->inputs = layer_info_->inputs;
        squeeze_layer_info_->outputs = {innerproduct_param->name + "-squeeze-out"};
        auto squeeze_param = std::shared_ptr<SqueezeLayerParam>(new SqueezeLayerParam);
        squeeze_layer_info_->param = squeeze_param;
        {
            std::vector<int> axes = {};
            auto  reduce_dims = input_shape_size - output_shape_size;
            for(int i=0;i<reduce_dims;i++){
                axes.push_back(i-reduce_dims);
            }
            squeeze_param->axes = axes;
        }
    }
    RETURN_ON_NEQ(squeeze_layer->Init(squeeze_layer_info_.get(), nullptr), TNN_OK);
    coreml_layer_before_ = squeeze_layer;
    
    return TNN_OK;
}

Status CoreMLInnerproductLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLInnerproductLayer::BuildLayerInputs() {
    if (!layer_info_) {
        return std::vector<std::string>();
    } else {
        if(input_shape_size > output_shape_size) {
            auto param = layer_info_->param.get();
            auto innerproduct_param = dynamic_cast<InnerProductLayerParam *>(param);
            return {innerproduct_param->name + "-squeeze-out"};
        }
        return layer_info_->inputs;
    }
}

std::vector<std::string> CoreMLInnerproductLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Innerproduct, LAYER_INNER_PRODUCT);

}  // namespace TNN_NS
