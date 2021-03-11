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

#include "tnn/device/arm/acc/arm_layer_acc.h"

#include "tnn/core/profile.h"
#include "tnn/device/arm/arm_context.h"
#include "tnn/utils/blob_transfer_utils.h"
#include "tnn/utils/data_type_utils.h"

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
    int ic          = input_dim[1];
    int ih          = input_dim.size() > 2 ? input_dim[2] : 1;
    int iw          = input_dim.size() > 3 ? input_dim[3] : 1;
    int oc          = output_dim[1];
    int oh          = output_dim.size() > 2 ? output_dim[2] : 1;
    int ow          = output_dim.size() > 3 ? output_dim[3] : 1;
    k_param_->set_dims(ROUND_UP(ic, 4), ROUND_UP(ic, 8), ih, iw, ROUND_UP(oc, 4), ROUND_UP(oc, 8), oh, ow);

    RETURN_ON_NEQ(ReloadConstantBlobs(inputs), TNN_OK);

    return TNN_OK;
}

std::vector<DataFormat> ArmLayerAcc::SupportDataFormat(DataType data_type, int dims_size) {
    std::vector<DataFormat> support_list;
    if (dims_size == 4) {
        if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_BFP16)
            support_list.push_back(DATA_FORMAT_NC4HW4);
        else if (data_type == DATA_TYPE_INT8)
            support_list.push_back(DATA_FORMAT_NHWC4);
        else if (data_type == DATA_TYPE_HALF) {
            support_list.push_back(DATA_FORMAT_NC8HW8);
        }
    }
    return support_list;
}

ArmLayerAcc::~ArmLayerAcc() {}

Status ArmLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    // reinit k_param_ h,w
    auto input_dim  = inputs[0]->GetBlobDesc().dims;
    auto output_dim = outputs[0]->GetBlobDesc().dims;
    int ic          = input_dim[1];
    int ih          = input_dim.size() > 2 ? input_dim[2] : 1;
    int iw          = input_dim.size() > 3 ? input_dim[3] : 1;
    int oc          = output_dim[1];
    int oh          = output_dim.size() > 2 ? output_dim[2] : 1;
    int ow          = output_dim.size() > 3 ? output_dim[3] : 1;
    k_param_->set_dims(ROUND_UP(ic, 4), ROUND_UP(ic, 8), ih, iw, ROUND_UP(oc, 4), ROUND_UP(oc, 8), oh, ow);
    return TNN_OK;
}

Status ArmLayerAcc::ReloadConstantBlobs(const std::vector<Blob *> &inputs) {
    auto const_resource = const_resource_;
    auto const_blob_map = const_blob_map_;
    for (auto iter : inputs) {
        auto name = iter->GetBlobDesc().name;
        if (const_resource == nullptr || const_resource->find(name) == const_resource->end()) {
            continue;
        }

        auto buffer = (*const_resource)[name];
        std::shared_ptr<Blob> blob = nullptr;
        if (const_blob_map.find(name) != const_blob_map.end()) {
            blob = const_blob_map[name];
        }
        auto status = RawBuffer2Blob(buffer.get(), blob);
        RETURN_ON_NEQ(status, TNN_OK);

        blob->flag = DATA_FLAG_CHANGE_NEVER;
        const_blob_map[name] = blob;
        iter->SetHandle(blob->GetHandle());
        iter->GetBlobDesc() = blob->GetBlobDesc();
        LOGD("Reload constant blob: %s\n", name.c_str());
    }
    const_blob_map_ = const_blob_map;
    return TNN_OK;
}

bool ArmLayerAcc::DataTypeSupported(DataType data_type) {
    if (data_type == DATA_TYPE_FLOAT || data_type == DATA_TYPE_BFP16 || data_type == DATA_TYPE_INT8 ||
        data_type == DATA_TYPE_HALF) {
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
