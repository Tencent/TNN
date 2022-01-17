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

DECLARE_COREML_LAYER_WITH_DATA(Pool, LAYER_POOLING,
                               std::shared_ptr<uint64_t> stride_;
                               std::shared_ptr<uint64_t> kernelsize_;
                               std::shared_ptr<CoreML__Specification__ValidPadding> valid_;
                               std::shared_ptr<CoreML__Specification__SamePadding> same_;
                               std::shared_ptr<CoreML__Specification__BorderAmounts> paddingamounts_;
                               std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes*> borderamounts_;
                               std::vector<std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes> > borderamounts_arr_;);

Status CoreMLPoolLayer::BuildLayerType() {
    //layer type
    coreml_layer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_POOLING;
    return TNN_OK;
}

Status CoreMLPoolLayer::BuildLayerParam() {
    //layer param
    auto param = layer_info_->param.get();
    auto pool_param = dynamic_cast<PoolingLayerParam *>(param);
    CHECK_PARAM_NULL(pool_param);
    
    auto pool_type = pool_param->pool_type;
    auto pad_type = pool_param->pad_type;
    auto is_adaptive_pool = pool_param->is_adaptive_pool;
    auto is_global_pool = pool_param->is_global_pool;
    auto kernel_x = pool_param->kernels[0];
    auto kernel_y = pool_param->kernels[1];
    auto stride_x = pool_param->strides[0];
    auto stride_y = pool_param->strides[1];
    
    coreml_layer_param_ = std::shared_ptr<CoreML__Specification__PoolingLayerParams>(new CoreML__Specification__PoolingLayerParams);
    coreml_layer_->pooling = (CoreML__Specification__PoolingLayerParams *)coreml_layer_param_.get();
    core_ml__specification__pooling_layer_params__init(coreml_layer_->pooling);
    coreml_layer_->pooling->globalpooling = is_global_pool;
    coreml_layer_->pooling->n_stride = 2;
    stride_ = std::shared_ptr<uint64_t>(new uint64_t [coreml_layer_->pooling->n_stride], [](uint64_t* p) { delete[] p; });
    coreml_layer_->pooling->stride = stride_.get();
    coreml_layer_->pooling->stride[0] = stride_y;
    coreml_layer_->pooling->stride[1] = stride_x;
    coreml_layer_->pooling->n_kernelsize = 2;
    kernelsize_ = std::shared_ptr<uint64_t>(new uint64_t [coreml_layer_->pooling->n_kernelsize], [](uint64_t* p) { delete[] p; });
    coreml_layer_->pooling->kernelsize = kernelsize_.get();
    coreml_layer_->pooling->kernelsize[0] = kernel_y;
    coreml_layer_->pooling->kernelsize[1] = kernel_x;
    
    if (pool_type == 0) {  // MaxPooling
        coreml_layer_->pooling->type = CORE_ML__SPECIFICATION__POOLING_LAYER_PARAMS__POOLING_TYPE__MAX;
    } else if (pool_type == 1) {  // AveragePooling
        coreml_layer_->pooling->type = CORE_ML__SPECIFICATION__POOLING_LAYER_PARAMS__POOLING_TYPE__AVERAGE;
        coreml_layer_->pooling->avgpoolexcludepadding = true;
    }
    
    if (pad_type == -1) { // default padding following the proto setting
        //[w_begin w_end h_begin h_end d_begin d_end]
        auto pad_left = pool_param->pads[0];
        auto pad_right = pool_param->pads[1];
        auto pad_top = pool_param->pads[2];
        auto pad_bottom = pool_param->pads[3];
    
        coreml_layer_->pooling->pooling_padding_type_case = CORE_ML__SPECIFICATION__POOLING_LAYER_PARAMS__POOLING_PADDING_TYPE_VALID;
        valid_ = std::shared_ptr<CoreML__Specification__ValidPadding>(new CoreML__Specification__ValidPadding);
        coreml_layer_->pooling->valid = valid_.get();
        core_ml__specification__valid_padding__init(coreml_layer_->pooling->valid);
        paddingamounts_ = std::shared_ptr<CoreML__Specification__BorderAmounts>(new CoreML__Specification__BorderAmounts);
        coreml_layer_->pooling->valid->paddingamounts = paddingamounts_.get();
        core_ml__specification__border_amounts__init(coreml_layer_->pooling->valid->paddingamounts);
        coreml_layer_->pooling->valid->paddingamounts->n_borderamounts = 2;
        borderamounts_ = std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes*>(new CoreML__Specification__BorderAmounts__EdgeSizes* [2], [](CoreML__Specification__BorderAmounts__EdgeSizes** p) { delete[] p; });
        coreml_layer_->pooling->valid->paddingamounts->borderamounts = borderamounts_.get();
        borderamounts_arr_.push_back(std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes>(new CoreML__Specification__BorderAmounts__EdgeSizes));
        coreml_layer_->pooling->valid->paddingamounts->borderamounts[0] = borderamounts_arr_[0].get();
        core_ml__specification__border_amounts__edge_sizes__init(coreml_layer_->pooling->valid->paddingamounts->borderamounts[0]);
        coreml_layer_->pooling->valid->paddingamounts->borderamounts[0]->startedgesize = pad_top;
        coreml_layer_->pooling->valid->paddingamounts->borderamounts[0]->endedgesize = pad_bottom;
        borderamounts_arr_.push_back(std::shared_ptr<CoreML__Specification__BorderAmounts__EdgeSizes>(new CoreML__Specification__BorderAmounts__EdgeSizes));
        coreml_layer_->pooling->valid->paddingamounts->borderamounts[1] = borderamounts_arr_[1].get();
        core_ml__specification__border_amounts__edge_sizes__init(coreml_layer_->pooling->valid->paddingamounts->borderamounts[1]);
        coreml_layer_->pooling->valid->paddingamounts->borderamounts[1]->startedgesize = pad_left;
        coreml_layer_->pooling->valid->paddingamounts->borderamounts[1]->endedgesize = pad_right;
    } else if (pad_type == 0) { // SAME type
        coreml_layer_->pooling->pooling_padding_type_case = CORE_ML__SPECIFICATION__POOLING_LAYER_PARAMS__POOLING_PADDING_TYPE_SAME;
        same_ = std::shared_ptr<CoreML__Specification__SamePadding>(new CoreML__Specification__SamePadding);
        coreml_layer_->pooling->same = same_.get();
        core_ml__specification__same_padding__init(coreml_layer_->pooling->same);
    } else if (pad_type == 1) { // VALID type
        coreml_layer_->pooling->pooling_padding_type_case = CORE_ML__SPECIFICATION__POOLING_LAYER_PARAMS__POOLING_PADDING_TYPE_VALID;
        valid_ = std::shared_ptr<CoreML__Specification__ValidPadding>(new CoreML__Specification__ValidPadding);
        coreml_layer_->pooling->valid = valid_.get();
        core_ml__specification__valid_padding__init(coreml_layer_->pooling->valid);
    }
    
    return TNN_OK;
}

Status CoreMLPoolLayer::BuildConstantWeightsLayer() {
    return CoreMLBaseLayer::BuildConstantWeightsLayer();
}

std::vector<std::string> CoreMLPoolLayer::BuildLayerInputs() {
    return CoreMLBaseLayer::BuildLayerInputs();
}

std::vector<std::string> CoreMLPoolLayer::BuildLayerOutputs() {
    return CoreMLBaseLayer::BuildLayerOutputs();
}

REGISTER_COREML_LAYER(Pool, LAYER_POOLING);

}  // namespace TNN_NS
