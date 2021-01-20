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

namespace TNN_NS {

X86LayerAcc::~X86LayerAcc() {}

Status X86LayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                         const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    AbstractLayerAcc::Init(context, param, resource, inputs, outputs);
    context_ = context;

    param_    = param;
    resource_ = resource;

    if (cpu_with_isa(avx2)) {
        arch_ = avx2;
    } else if (cpu_with_isa(sse42)) {
        arch_ = sse42;
    }

    return Reshape(inputs, outputs);
}

std::vector<DataFormat> X86LayerAcc::SupportDataFormat(DataType data_type, int dims_size) {
    std::vector<DataFormat> support_list;
    if (dims_size == 4) {
        support_list.push_back(DATA_FORMAT_NCHW);
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

Status X86LayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return Status(TNNERR_LAYER_ERR, "DoForward not implement");
}

}  // namespace TNN_NS
