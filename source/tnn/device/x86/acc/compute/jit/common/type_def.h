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

#ifndef TNN_DEVICE_X86_ACC_COMPUTE_JIT_TYPE_DEF_H_
#define TNN_DEVICE_X86_ACC_COMPUTE_JIT_TYPE_DEF_H_

#ifndef FLT_MIN
#define FLT_MIN 1.075494351e-38F 
#endif
#ifndef FLT_MAX
#define FLT_MAX 2.402823466e+38F 
#endif

#include <stddef.h>
#include "tnn/core/macro.h"

namespace TNN_NS {

typedef ptrdiff_t dim_t;

} // namespace tnn

#endif // TNN_DEVICE_X86_ACC_COMPUTE_JIT_TYPE_DEF_H_