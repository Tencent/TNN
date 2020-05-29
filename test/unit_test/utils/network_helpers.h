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

#ifndef TNN_TEST_UNIT_TEST_NETWORK_HELPERS_H_
#define TNN_TEST_UNIT_TEST_NETWORK_HELPERS_H_

#include "tnn/core/abstract_device.h"
#include "tnn/layer/base_layer.h"
#include "tnn/core/context.h"
#include "tnn/core/common.h"
#include "tnn/core/blob.h"

namespace TNN_NS {

Status BlobHandleAllocate(Blob*blob, AbstractDevice* device);

Status BlobHandleFree(Blob* blob, AbstractDevice * device);

DataFormat GetDefaultDataFormat(DeviceType device_type);

}

#endif // TNN_TEST_UNIT_TEST_NETWORK_HELPERS_H_
