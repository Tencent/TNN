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

#include "tnn/core/mat.h"

#include "tnn/core/abstract_device.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

Mat::~Mat() {
    data_alloc_ = nullptr;
    data_       = nullptr;
}

DeviceType Mat::GetDeviceType() {
    return device_type_;
}

MatType Mat::GetMatType() {
    return mat_type_;
}

void* Mat::GetData() {
    return data_;
}

DimsVector Mat::GetDims() {
    return dims_;
}

int Mat::GetDim(int index) {
    if (index >= 0 && index <dims_.size()) {
        return dims_[index];
    } else {
        return 0;
    }
}

int Mat::GetBatch() {
    return GetDim(0);
}

int Mat::GetChannel() {
    return GetDim(1);
}

int Mat::GetHeight() {
    return GetDim(2);
}

int Mat::GetWidth() {
    return GetDim(3);
}

Mat::Mat(DeviceType device_type, MatType mat_type, DimsVector dims) {
    dims_ = dims;
    
    auto device = GetDevice(device_type);
    ASSERT(device != NULL);

    int count = DimsVectorUtils::Count(dims);
    ASSERT(count > 0);

    device_type_     = device_type;
    mat_type_        = mat_type;
    void* data_alloc = nullptr;
    auto status      = device->Allocate(&data_alloc, mat_type, dims);
    if (status == TNN_OK) {
        data_alloc_ = std::shared_ptr<void>(data_alloc, [=](void* p) {
            auto device = GetDevice(device_type);
            if (device) {
                device->Free(p);
            }
        });
        data_       = data_alloc_.get();
    } else {
        data_       = nullptr;
        data_alloc_ = nullptr;
    }
}

Mat::Mat(DeviceType device_type, MatType mat_type, DimsVector dims, void* data) {
    dims_ = dims;
    
    data_alloc_ = nullptr;

    device_type_ = device_type;
    mat_type_    = mat_type;
    data_        = data;
}

}  // namespace TNN_NS
