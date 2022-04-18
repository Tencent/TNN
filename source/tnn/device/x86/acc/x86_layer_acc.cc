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

#include "tnn/device/x86/acc/x86_layer_acc.h"
#include "tnn/utils/blob_transfer_utils.h"

namespace TNN_NS {

X86LayerAcc::~X86LayerAcc() {}

Status X86LayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                         const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    AbstractLayerAcc::Init(context, param, resource, inputs, outputs);
    context_ = reinterpret_cast<X86Context *>(context);

    param_    = param;
    resource_ = resource;

    RETURN_ON_NEQ(ReloadConstantBlobs(inputs, false), TNN_OK);

    // for layer use intrinsic, avx2 and avx use the same impl
    if (cpu_with_isa(avx2) || cpu_with_isa(avx)) {
        arch_ = avx2;
    } else if (cpu_with_isa(sse42)) {
        arch_ = sse42;
    } else {
        return Status(TNNERR_DEVICE_NOT_SUPPORT, "Cat not support X86 arch before SSE4.2");
    }

    return Reshape(inputs, outputs);
}

std::vector<DataFormat> X86LayerAcc::SupportDataFormat(DataType data_type, int dims_size, BlobType blob_type) {
    std::vector<DataFormat> support_list;
    if (dims_size == 4) {
        if (data_type == DATA_TYPE_FLOAT)
            support_list.push_back(DATA_FORMAT_NCHW);
        else if (data_type == DATA_TYPE_INT8)
            support_list.push_back(DATA_FORMAT_NHWC4);
    }
    return support_list;
}

Status X86LayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    Status status;
#if TNN_PROFILE
    auto pdata = std::make_shared<ProfilingData>();
    UpdateProfilingData(pdata.get(), param_, inputs[0]->GetBlobDesc().dims, outputs[0]->GetBlobDesc().dims);
    timer.Start();
#endif

    status = this->DoForward(inputs, outputs);

#if TNN_PROFILE
    pdata->kernel_time = timer.TimeEclapsed();
    context_->AddProfilingData(pdata);
#endif

    RETURN_ON_NEQ(status, TNN_OK);

    return TNN_OK;
}

Status X86LayerAcc::Reshape(const std::vector<Blob*> &inputs, const std::vector<Blob*> &outputs) {
    return TNN_OK;
}

Status X86LayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return Status(TNNERR_LAYER_ERR, "DoForward not implement");
}

Status X86LayerAcc::ReloadConstantBlobs(const std::vector<Blob *> &inputs, bool only_reload_shape_differ_blob) {
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
        LOGD("Reload constant blob: %s\n", name.c_str());
    }
    const_blob_map_ = const_blob_map;
    return TNN_OK;
}

}  // namespace TNN_NS
