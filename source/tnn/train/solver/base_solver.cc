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

// author: sanerzheng@tencent.com

#include "tnn/train/solver/base_solver.h"
#include "tnn/utils/bfp16.h"


namespace TNN_NS {
namespace train {
template<typename T> void UpdateVariable(T* dst_ptr, const T* src, int count);
template <>
void UpdateVariable<float>(float* dst_ptr, const float* src, int count){
    //NOTE: don't deal with dataformat 
    for(int i=0; i<count; i++)
        dst_ptr[i] -= src[i];
}
template <>
void UpdateVariable<bfp16_t>(bfp16_t* dst_ptr, const bfp16_t* src, int count){
    //NOTE: don't deal with dataformat
    cvt_32b c1; 
    cvt_32b c2; 
    for(int i=0; i<count; i++) {
        c1.u = dst_ptr[i].w << 16;
        c2.u = src[i].w << 16;
        c1.f -= c2.f;
        dst_ptr[i] = c1.u >> 16;
    }
}


Status BaseSolver::ComputeUpdateValue(RawBuffer* resource_param, std::shared_ptr<RawBuffer>& resource_param_grad) {
    return Status(TNN_TRAIN_ERROR, "ComputeUpdateValue not implement in this solver");
}

void BaseSolver::SetNeedGradLayers(const std::set<std::string>& need_grad_layers) {
    grad_manager_.SetNeedGradLayers(need_grad_layers);
    return;
}

Status BaseSolver::UpdateTrainableVariable(RawBuffer* resource_param, const std::shared_ptr<RawBuffer>& resource_param_grad) {
    if(resource_param->GetDataType() != resource_param_grad->GetDataType() || resource_param->GetDataCount() != resource_param_grad->GetDataCount()) {
        return Status(TNN_TRAIN_ERROR, "grad data type or dims not match"); 
    }
    int count = resource_param->GetDataCount();
    if(count <= 0) 
        return Status(TNN_TRAIN_ERROR, "grad data count error");  
    if(resource_param->GetDataType() == DATA_TYPE_FLOAT) 
        UpdateVariable<float>(resource_param->force_to<float *>(), resource_param_grad->force_to<const float*>(), count);
    else if(resource_param->GetDataType() == DATA_TYPE_BFP16)
        UpdateVariable<bfp16_t>(resource_param->force_to<bfp16_t* >(), resource_param_grad->force_to<const bfp16_t*>(), count);
    else
        return Status(TNN_TRAIN_ERROR, "DATA TYPE NOT SUPPORT");
    return Status(TNN_OK);
}
Status BaseSolver::step() {
    RETURN_ON_NEQ(grad_manager_.IsSupport(), TNN_OK);
    RETURN_ON_NEQ(grad_manager_.CalcuteGrads(), TNN_OK);
    auto& resource_grads = grad_manager_.GetContext().backward_grads_resource;
    for(auto iter: resource_grads){
        if(iter.first->GetTrainable()) {
            Status status = ComputeUpdateValue(iter.first, iter.second);
            RETURN_ON_NEQ(status, TNN_OK);
            status = UpdateTrainableVariable(iter.first, iter.second);
            RETURN_ON_NEQ(status, TNN_OK);
        }
    }
    return TNN_OK;
}

}// namespace trian
}// namespace TNN_NS