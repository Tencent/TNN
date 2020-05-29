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

namespace TNN_NS {

CpuLayerAcc::~CpuLayerAcc() {}

Status CpuLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                         const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    AbstractLayerAcc::Init(context, param, resource, inputs, outputs);

    param_    = param;
    resource_ = resource;
    return Reshape(inputs, outputs);
}

std::vector<DataFormat> CpuLayerAcc::SupportDataFormat(DataType data_type, int dims_size) {
    std::vector<DataFormat> support_list;
    if (dims_size == 4) {
        support_list.push_back(DATA_FORMAT_NCHW);
    }
    return support_list;
}

}  // namespace TNN_NS
