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

#include "tnn/device/cpu/acc/cpu_layer_acc.h"
#include "tnn/utils/blob_transfer_utils.h"

namespace TNN_NS {

CpuLayerAcc::~CpuLayerAcc() {}

Status CpuLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                         const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    param_    = param;
    resource_ = resource;
    
    auto status = AbstractLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(status, TNN_OK);
    
    status = ReloadConstantBlobs(inputs, false);

    RETURN_ON_NEQ(status, TNN_OK);

    return Reshape(inputs, outputs);
}

Status CpuLayerAcc::ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob) {

    auto const_resource = const_resource_;
    auto const_resource_flag = const_resource_flag_;
    auto const_blob_map = const_blob_map_;
    for (auto iter : inputs) {
        auto name = iter->GetBlobDesc().name;
        if (const_resource == nullptr || const_resource->find(name) == const_resource->end()) {
            continue;
        }
        
        if (only_reload_shape_differ_blob && const_resource_flag &&
            const_resource_flag->find(name) == const_resource_flag->end()) {
            continue;
        }
        
        auto buffer = (*const_resource)[name];
        std::shared_ptr<Blob> blob = nullptr;
        if (const_blob_map.find(name) != const_blob_map.end()) {
            blob = const_blob_map[name];
        }
        auto status = RawBuffer2Blob(buffer.get(), blob);
        RETURN_ON_NEQ(status, TNN_OK);

        blob->SetFlag(DATA_FLAG_CHANGE_NEVER);
        const_blob_map[name] = blob;
        iter->SetHandle(blob->GetHandle());
        LOGD("Reload constant blob: %s %p\n", name.c_str(), &blob);
    }
    const_blob_map_ = const_blob_map;
    return TNN_OK;
}

std::vector<DataFormat> CpuLayerAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    std::vector<DataFormat> support_list;
    if (dims_size == 5) {
        support_list.push_back(DATA_FORMAT_NCDHW);
    } else if (dims_size >= 0) {
        support_list.push_back(DATA_FORMAT_NCHW);
    }
    return support_list;
}

}  // namespace TNN_NS
