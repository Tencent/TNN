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

#ifndef TNN_TEST_FLAGS_H_
#define TNN_TEST_FLAGS_H_

#include "gflags/gflags.h"
#include "tnn/core/macro.h"

namespace TNN_NS {

static const char help_message[] = "print a usage message.";

static const char model_type_message[] = "specify model type: TNN, OPENVINO, COREML, SNPE, NCNN, RKCACHE.";

static const char model_path_message[] =
    "specify model path: tnn proto path, openvino xml path, coreml "
    "mlmodel path, snpe dlc path.";

static const char device_type_message[] = "(required) specify tnn device type: NAIVE, X86, ARM, CUDA, METAL, OPENCL, HUAWEI_NPU, APPLE_NPU, default is ARM.";

static const char library_path_message[] =
    "specify tnn NetworkConfig library_path. For metal, it is the tnn.metallib full path";

static const char device_id_message[] = "specify device id(default 0).";

static const char iterations_count_message[] = "iterations count (default 1).";

static const char warm_up_count_message[] = "warm up count (default 0).";

static const char input_path_message[] = "input file path";

static const char output_path_message[] = "output file path";

static const char output_format_cmp_message[] = "output format for comparison";

static const char device_list_message[] = "device list(eg: 0,1,2,3)";

static const char unit_test_benchmark_message[] = "enable unit benchmark(default false)";

static const char cpu_thread_num_message[] = "cpu thread num(eg: 0,1,2,3, default 1)";

static const char input_format_message[] = "input format(0: nchw float; 1: bgr u8; 2: gray u8; 3: int32; 4: int8;), default nchw float";

static const char precision_message[] = "compute precision(HIGH, NORMAL, LOW)";

static const char input_shape_message[] = "input shape: name[n,c,h,w]";

static const char network_type_message[] = "network type: NAIVE, NPU, COREML, SNPE, OPENVINO, default NAIVE";

static const char enable_tune_message[] = "enable tune kernel(default false)";

static const char scale_message[] = "input scale: s0,s1,s2,...)";

static const char bias_message[] = "input bias: b0,b1,b2,...)";

DECLARE_bool(h);

DECLARE_string(mt);

DECLARE_string(nt);

DECLARE_string(mp);

DECLARE_string(dt);

DECLARE_string(lp);

DECLARE_int32(di);

DECLARE_int32(ic);

DECLARE_int32(wc);

DECLARE_string(ip);

DECLARE_string(op);

DECLARE_bool(fc);

DECLARE_string(dl);

DECLARE_bool(ub);

DECLARE_int32(th);

DECLARE_int32(it);

DECLARE_string(pr);

DECLARE_string(is);

DECLARE_bool(et);

DECLARE_string(sc);

DECLARE_string(bi);

}  // namespace TNN_NS

#endif  // TNN_TEST_FLAGS_H_
