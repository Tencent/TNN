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

DECLARE_COREML_LAYER_WITH_DATA(Deconv, LAYER_DECONVOLUTION,
                                std::shared_ptr<uint64_t> stride_;
                                std::shared_ptr<uint64_t> dilationfactor_;
                                std::shared_ptr<uint64_t> kernelsize_;
                                std::shared_ptr<CoreML__Specification__ValidPadding> valid_;
                                std::shared_ptr<CoreML__Specification__SamePadding> same_;
                                std::shared_ptr<CoreML__Specification__BorderAmounts> paddingamounts_;
                                std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes*> borderamounts_;
                                std::vector<std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes> > borderamounts_arr_;
                                std::shared_ptr<CoreML__Specification__WeightParams> weight_param_;
                                std::shared_ptr<CoreML__Specification__WeightParams> bias_param_;
                                std::shared_ptr<float> weight_fp32_ptr_;
                                std::shared_ptr<float> bias_fp32_ptr_;);

Status CoreMLDeconvLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_CONVOLUTION;
    return TNN_OK;
}

Status CoreMLDeconvLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto conv_param = dynamic_cast<ConvLayerParam *>(param);
    CHECK_PARAM_NULL(conv_param);
    auto kernel_x = conv_param->kernels[0];
    auto kernel_y = conv_param->kernels[1];
    auto stride_x = conv_param->strides[0];
    auto stride_y = conv_param->strides[1];
    auto dilate_x = conv_param->dialations[0];
    auto dilate_y = conv_param->dialations[1];
    auto has_bias = conv_param->bias;
    auto n_group = conv_param->group;
    auto output_channels = conv_param->output_channel;
    auto pad_type = conv_param->pad_type;
    auto conv_res = dynamic_cast<ConvLayerResource *>(layer_resource_);
    CHECK_PARAM_NULL(conv_res);
    auto weight_size = conv_res->filter_handle.GetDataCount();
    auto weight_type = conv_res->filter_handle.GetDataType();
    auto bias_size = conv_res->bias_handle.GetDataCount();
    auto bias_type = conv_res->bias_handle.GetDataType();
    output_channels = std::max(output_channels, bias_size);
    auto kernel_channels = n_group*weight_size/kernel_x/kernel_y/output_channels;
    kernel_channels = std::max(kernel_channels, conv_param->input_channel);
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__ConvolutionLayerParams>(new CoreML__Specification__ConvolutionLayerParams);
    coreml_layer_->convolution = (CoreML__Specification__ConvolutionLayerParams *)coreml_layer_param_.get();
    core_ml__specification__convolution_layer_params__init(coreml_layer_->convolution);
    // deconvolution
    coreml_layer_->convolution->isdeconvolution = true;
    coreml_layer_->convolution->ngroups = n_group;
    coreml_layer_->convolution->n_stride = 2;
    stride_ = std::shared_ptr<uint64_t>(new uint64_t [coreml_layer_->convolution->n_stride], [](uint64_t* p) { delete[] p; });
    coreml_layer_->convolution->stride = stride_.get();
    coreml_layer_->convolution->stride[0] = stride_y;
    coreml_layer_->convolution->stride[1] = stride_x;
    coreml_layer_->convolution->n_dilationfactor = 2;
    dilationfactor_ = std::shared_ptr<uint64_t>(new uint64_t [coreml_layer_->convolution->n_dilationfactor], [](uint64_t* p) { delete[] p; });
    coreml_layer_->convolution->dilationfactor = dilationfactor_.get();
    coreml_layer_->convolution->dilationfactor[0] = dilate_y;
    coreml_layer_->convolution->dilationfactor[1] = dilate_x;
    coreml_layer_->convolution->kernelchannels = kernel_channels;
    coreml_layer_->convolution->outputchannels = output_channels;
    coreml_layer_->convolution->n_kernelsize = 2;
    kernelsize_ = std::shared_ptr<uint64_t>(new uint64_t [coreml_layer_->convolution->n_kernelsize], [](uint64_t* p) { delete[] p; });
    coreml_layer_->convolution->kernelsize = kernelsize_.get();
    coreml_layer_->convolution->kernelsize[0] = kernel_y;
    coreml_layer_->convolution->kernelsize[1] = kernel_x;

    weight_param_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
    coreml_layer_->convolution->weights = weight_param_.get();
        core_ml__specification__weight_params__init(coreml_layer_->convolution->weights);
    switch (weight_type) {
        case DATA_TYPE_FLOAT:
            coreml_layer_->convolution->weights->n_floatvalue = weight_size;
            coreml_layer_->convolution->weights->floatvalue = conv_res->filter_handle.force_to<float *>();
            break;
        case DATA_TYPE_HALF:
            {
#if TNN_COREML_FULL_PRECISION
                coreml_layer_->convolution->weights->n_floatvalue = weight_size;
                void *weight_data_ptr = conv_res->filter_handle.force_to<void *>();
                weight_fp32_ptr_ = std::shared_ptr<float>(new float [weight_size], [](float* p) { delete[] p; });
                auto weight_fp32_ptr = weight_fp32_ptr_.get();
                RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)weight_data_ptr, (float *)weight_fp32_ptr, weight_size),TNN_OK);
                coreml_layer_->convolution->weights->floatvalue = weight_fp32_ptr;
#else
                coreml_layer_->convolution->weights->float16value.len = conv_res->filter_handle.GetBytesSize();
                coreml_layer_->convolution->weights->float16value.data = conv_res->filter_handle.force_to<uint8_t *>();
#endif
            }
            break;
        default:
            LOGE("CoreMLDeconvLayer dont support data type (%d)\n", weight_type);
            return Status(TNNERR_MODEL_ERR, "CoreMLDeconvLayer dont support this weight data type");
            break;
    }
    if (bias_size) {
        coreml_layer_->convolution->hasbias = true;
        bias_param_ = std::shared_ptr<CoreML__Specification__WeightParams>(new CoreML__Specification__WeightParams);
        coreml_layer_->convolution->bias = bias_param_.get();
        core_ml__specification__weight_params__init(coreml_layer_->convolution->bias);
       
        switch (bias_type) {
            case DATA_TYPE_FLOAT:
                coreml_layer_->convolution->bias->n_floatvalue = bias_size;
                coreml_layer_->convolution->bias->floatvalue = conv_res->bias_handle.force_to<float *>();
                break;
            case DATA_TYPE_HALF:
                {
#if TNN_COREML_FULL_PRECISION
                    coreml_layer_->convolution->bias->n_floatvalue = bias_size;
                    void *bias_data_ptr = conv_res->bias_handle.force_to<void *>();
                    bias_fp32_ptr_ = std::shared_ptr<float>(new float [bias_size], [](float* p) { delete[] p; });
                    auto bias_fp32_ptr = bias_fp32_ptr_.get();
                    RETURN_ON_NEQ(ConvertFromHalfToFloat((void *)bias_data_ptr, (float *)bias_fp32_ptr, bias_size),TNN_OK);
                    coreml_layer_->convolution->bias->floatvalue = bias_fp32_ptr;
#else
                    coreml_layer_->convolution->bias->float16value.len = conv_res->bias_handle.GetBytesSize();
                    coreml_layer_->convolution->bias->float16value.data = conv_res->bias_handle.force_to<uint8_t *>();
#endif
                }
                break;
            default:
                LOGE("CoreMLDeconvLayer dont support data type (%d)\n", bias_type);
                return Status(TNNERR_MODEL_ERR, "CoreMLDeconvLayer dont support this bias data type");
                break;
        }
    }
    
    if (pad_type == -1) { // default padding following the proto setting
        //[w_begin w_end h_begin h_end d_begin d_end]
        auto pad_left = conv_param->pads[0];
        auto pad_right = conv_param->pads[1];
        auto pad_top = conv_param->pads[2];
        auto pad_bottom = conv_param->pads[3];

        coreml_layer_->convolution->convolution_padding_type_case = CORE_ML__SPECIFICATION__CONVOLUTION_LAYER_PARAMS__CONVOLUTION_PADDING_TYPE_VALID;
        valid_ = std::shared_ptr<CoreML__Specification__ValidPadding>(new CoreML__Specification__ValidPadding);
        coreml_layer_->convolution->valid = valid_.get();
        core_ml__specification__valid_padding__init(coreml_layer_->convolution->valid);
        paddingamounts_ = std::shared_ptr<CoreML__Specification__BorderAmounts>(new CoreML__Specification__BorderAmounts);
        coreml_layer_->convolution->valid->paddingamounts = paddingamounts_.get();
        core_ml__specification__border_amounts__init(coreml_layer_->convolution->valid->paddingamounts);
        coreml_layer_->convolution->valid->paddingamounts->n_borderamounts = 2;
        borderamounts_ = std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes*>(new CoreML__Specification__BorderAmounts__EdgeSizes* [2], [](CoreML__Specification__BorderAmounts__EdgeSizes** p) { delete[] p; });
        coreml_layer_->convolution->valid->paddingamounts->borderamounts = borderamounts_.get();
        borderamounts_arr_.push_back(std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes>(new CoreML__Specification__BorderAmounts__EdgeSizes));
        coreml_layer_->convolution->valid->paddingamounts->borderamounts[0] = borderamounts_arr_[0].get();
        core_ml__specification__border_amounts__edge_sizes__init(coreml_layer_->convolution->valid->paddingamounts->borderamounts[0]);
        // This must be length 2 in the order ``[H, W]``.
        coreml_layer_->convolution->valid->paddingamounts->borderamounts[0]->startedgesize = pad_top;
        coreml_layer_->convolution->valid->paddingamounts->borderamounts[0]->endedgesize = pad_bottom;
        borderamounts_arr_.push_back(std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes>(new CoreML__Specification__BorderAmounts__EdgeSizes));
        coreml_layer_->convolution->valid->paddingamounts->borderamounts[1] = borderamounts_arr_[1].get();
        core_ml__specification__border_amounts__edge_sizes__init(coreml_layer_->convolution->valid->paddingamounts->borderamounts[1]);
        coreml_layer_->convolution->valid->paddingamounts->borderamounts[1]->startedgesize = pad_left;
        coreml_layer_->convolution->valid->paddingamounts->borderamounts[1]->endedgesize = pad_right;
    } else if (pad_type == 0) { // SAME type
        coreml_layer_->convolution->convolution_padding_type_case = CORE_ML__SPECIFICATION__CONVOLUTION_LAYER_PARAMS__CONVOLUTION_PADDING_TYPE_SAME;
        same_ = std::shared_ptr<CoreML__Specification__SamePadding>(new CoreML__Specification__SamePadding);
        coreml_layer_->convolution->same = same_.get();
        core_ml__specification__same_padding__init(coreml_layer_->convolution->same);
    } else if (pad_type == 1) { // VALID type
        coreml_layer_->convolution->convolution_padding_type_case = CORE_ML__SPECIFICATION__CONVOLUTION_LAYER_PARAMS__CONVOLUTION_PADDING_TYPE_VALID;
        valid_ = std::shared_ptr<CoreML__Specification__ValidPadding>(new CoreML__Specification__ValidPadding);
        coreml_layer_->convolution->valid = valid_.get();
        core_ml__specification__valid_padding__init(coreml_layer_->convolution->valid);
    }
    
    return TNN_OK;
}

Status CoreMLDeconvLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLDeconvLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLDeconvLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Deconv, LAYER_DECONVOLUTION);

}  // namespace TNN_NS
