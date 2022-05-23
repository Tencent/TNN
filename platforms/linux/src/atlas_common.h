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

#ifndef TNN_PLATFORM_LINUX_SRC_ATLAS_COMMON_H_
#define TNN_PLATFORM_LINUX_SRC_ATLAS_COMMON_H_

#include <string>
#include "tnn/core/tnn.h"

struct TNNParam {
    std::string input_file;
    int device_id = 0;
    int thread_id = 0;
    int batch_size = 1;
    TNN_NS::TNN* tnn_net;
    TNN_NS::NetworkType network_type = TNN_NS::NETWORK_TYPE_ATLAS;
    TNN_NS::DeviceType device_type   = TNN_NS::DEVICE_ATLAS;
};

void* RunTNN(void* param);

#endif  // end of TNN_PLATFORM_LINUX_SRC_ATLAS_COMMON_H_
