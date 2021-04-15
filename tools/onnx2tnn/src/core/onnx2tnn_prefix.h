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

#ifndef onnx2tnn_prefix_h
#define onnx2tnn_prefix_h
#include "tnn/core/macro.h"
#include "tnn/core/common.h"
#include "tnn/utils/half_utils.h"


#define LOG_LEVEL 2
#define LOG_FUNCTION (LOG_LEVEL >= 3 ? __PRETTY_FUNCTION__ : __FUNCTION__)
#define DLog(fmt, ...)                                                         \
    printf(("%s [Line %d] " fmt), LOG_FUNCTION, __LINE__, ##__VA_ARGS__)


#define k_tnn_noop_type "tnn.noop"
#define k_layout_order_nchw "nchw"
#define k_layout_order_nhwc "nhwc"

#define k_onnx_from_tensorflow "tensorflow"
#define k_onnx_from_pytorch "pytorch"

#define k_device_gpu "gpu"
#define k_device_cpu "cpu"

#define ERROR(fmt, ...)                                             \
    do {                                                            \
        const int _MAX = 2000;                                      \
        char _ss[_MAX];                                             \
        auto bt = get_backtrack();                                  \
        snprintf(_ss, _MAX, fmt "\tin:\n%s:%d\nbacktrace:\n:%s",    \
            ##__VA_ARGS__ , __FILE__, __LINE__, bt.c_str());        \
        throw std::runtime_error(_ss);                              \
    } while(0)
    
#endif /* prefix_h */
