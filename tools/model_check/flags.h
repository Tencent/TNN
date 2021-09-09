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

#ifndef TNN_TOOLS_MODEL_CHECK_FLAGS_H_
#define TNN_TOOLS_MODEL_CHECK_FLAGS_H_

#include "gflags/gflags.h"
#include "tnn/core/macro.h"

namespace TNN_NS {

static const char help_message[] = "show this message";

static const char proto_path_message[] = "(required) tnn proto file path";

static const char model_path_message[] = "(required) tnn model file path";

static const char device_type_message[] = "(required) specify tnn device type: NAIVE, X86, ARM, CUDA, METAL, OPENCL, HUAWEI_NPU, default is ARM.";

static const char input_path_message[] = "(optional) input file path";

static const char output_ref_path_message[] = "(optional) the reference output to compare";

static const char cmp_end_message[] = "(optional) compare output only";

static const char bias_message[] = "(optional) bias val when preprocess image input, ie, 0.0,0.0,0.0";

static const char scale_message[] = "(optional) scale val when preprocess image input, ie, 1.0,1.0,1.0";

static const char dump_output_path_message[] = "(optional) specify the path for dump output";

static const char check_batch_message[] = "(optional) check result of multi batch";

static const char align_all_message[] = "(optional) dump folder path to compare the all model";

static const char set_precision_message[] = "(optional) specify tnn precision type(default HIGH): AUTO, NORMAL, HIGH, LOW";

static const char dump_unaligned_layer_path_message[] = "(optional) specify the path for dump unaligned layer";

DECLARE_bool(h);

DECLARE_string(p);

DECLARE_string(m);

DECLARE_string(d);

DECLARE_string(i);

DECLARE_string(f);

DECLARE_bool(e);

DECLARE_string(n);

DECLARE_string(s);

DECLARE_bool(b);

DECLARE_string(a);

DECLARE_string(sp);

DECLARE_string(do);

DECLARE_string(du);
}  // namespace TNN_NS

#endif  // TNN_TOOLS_MODEL_CHECK_FLAGS_H_
