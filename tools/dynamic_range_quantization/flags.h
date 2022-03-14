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

#ifndef TNN_TOOLS_DYNAMIC_RANGE_FLAGS_H
#define TNN_TOOLS_DYNAMIC_RANGE_FLAGS_H

#include "gflags/gflags.h"
#include "tnn/core/macro.h"

namespace TNN_NS {

static const char help_message[] = "show this message";

static const char proto_message[] = "(required) tnn proto file path";

static const char model_message[] = "(required) tnn model file path";

static const char quant_proto_message[] = "(required) the path to save quant tnnproto file";

static const char quant_model_message[] = "(required) the path to save quant tnnmodel file";

DECLARE_bool(h);

DECLARE_string(p);

DECLARE_string(m);

DECLARE_string(qp);

DECLARE_string(qm);

}  // namespace TNN_NS

#endif  // TNN_TOOLS_DYNAMIC_RANGE_FLAGS_H
