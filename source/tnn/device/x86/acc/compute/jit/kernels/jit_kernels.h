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

#ifndef TNN_JIT_JIT_KERNELS_H_
#define TNN_JIT_JIT_KERNELS_H_

#include "tnn/device/x86/acc/compute/jit/kernels/sgemm_fetch_n.h"
#include "tnn/device/x86/acc/compute/jit/kernels/sgemm_fetch_n_6.h"
#include "tnn/device/x86/acc/compute/jit/kernels/sgemm_fetch_t.h"
#include "tnn/device/x86/acc/compute/jit/kernels/sgemm_fetch_t_4.h"
#include "tnn/device/x86/acc/compute/jit/kernels/sgemm_fetch_t_8.h"
#include "tnn/device/x86/acc/compute/jit/kernels/sgemm_fetch_t_16.h"
#include "tnn/device/x86/acc/compute/jit/kernels/sgemm_fetch_t_4x16.h"
#include "tnn/device/x86/acc/compute/jit/kernels/sgemm_avx_kernels.h"
#include "tnn/device/x86/acc/compute/jit/kernels/conv_sgemm_avx_kernels.h"

#endif // TNN_JIT_JIT_KERNELS_H_