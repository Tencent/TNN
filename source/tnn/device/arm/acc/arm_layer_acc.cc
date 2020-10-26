//
//  arm_layer_acc.cpp
//  tnn
//
//  Created by seanxcwang on 2019/9/17.
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
//

#include "tnn/core/profile.h"
#include "tnn/device/arm/acc/arm_layer_acc.h"
#include "tnn/device/arm/arm_context.h"

namespace TNN_NS {

Status ArmLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                         const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    AbstractLayerAcc::Init(context, param, resource, inputs, outputs);
    context_ = reinterpret_cast<ArmContext *>(context);

    param_    = param;
    resource_ = resource;
    k_param_  = std::make_shared<ArmKernelParam>();

    // init base k_param_
    auto input_dim  = inputs[0]->GetBlobDesc().dims;
    auto output_dim = outputs[0]->GetBlobDesc().dims;
    k_param_->ic_r4 = ROUND_UP(input_dim[1], 4);
    k_param_->ih    = input_dim[2];
    k_param_->iw    = input_dim[3];
    k_param_->oc_r4 = ROUND_UP(output_dim[1], 4);
    k_param_->oh    = output_dim[2];
    k_param_->ow    = output_dim[3];

    return TNN_OK;
}

std::vector<DataFormat> ArmLayerAcc::SupportDataFormat(DataType data_type, int dims_size) {
    std::vector<DataFormat> support_list;
    if (dims_size == 4) {
        if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_BFP16 || data_type == DATA_TYPE_HALF)
            support_list.push_back(DATA_FORMAT_NC4HW4);
        else if (data_type == DATA_TYPE_INT8)
            support_list.push_back(DATA_FORMAT_NHWC4);
    }
    return support_list;
}

ArmLayerAcc::~ArmLayerAcc() {}

Status ArmLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    // reinit k_param_ h,w
    auto input_dim  = inputs[0]->GetBlobDesc().dims;
    auto output_dim = outputs[0]->GetBlobDesc().dims;
    k_param_->ic_r4 = ROUND_UP(input_dim[1], 4);
    k_param_->ih    = input_dim[2];
    k_param_->iw    = input_dim[3];
    k_param_->oc_r4 = ROUND_UP(output_dim[1], 4);
    k_param_->oh    = output_dim[2];
    k_param_->ow    = output_dim[3];
    return TNN_OK;
}

bool ArmLayerAcc::DataTypeSupported(DataType data_type) {
    if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_BFP16 || data_type == DATA_TYPE_INT8) {
        return true;
    } else { 
        return false;
    }
}

Status ArmLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status;
#if TNN_PROFILE
    auto pdata = std::make_shared<ProfilingData>();
    UpdateProfilingData(pdata.get(), param_, inputs[0]->GetBlobDesc().dims, outputs[0]->GetBlobDesc().dims);
    timer.Start();
#endif

    auto in_data_type = inputs[0]->GetBlobDesc().data_type;
    if (DataTypeSupported(in_data_type)) {
        status = this->DoForward(inputs, outputs);
    } else {
        LOGE("Error : arm layer acc got unsupported data type %d\n", in_data_type);
        return Status(TNNERR_LAYER_ERR, "Error: arm layer acc got unsupported data type.");
    }

#if TNN_PROFILE
    pdata->kernel_time = timer.TimeEclapsed();
    context_->AddProfilingData(pdata);
#endif

    RETURN_ON_NEQ(status, TNN_OK);

    return TNN_OK;
}

Status ArmLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return Status(TNNERR_LAYER_ERR, "DoForward not implement");
}

}  // namespace TNN_NS
