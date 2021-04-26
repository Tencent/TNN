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

#ifndef TNN_EXAMPLES_LINUX_SRC_FLAGS_H_
#define TNN_EXAMPLES_LINUX_SRC_FLAGS_H_

#include "gflags/gflags.h"

DECLARE_bool(h);
DECLARE_string(p);
DECLARE_string(m);
DECLARE_string(i);

static const char help_message[] = "print a usage message.";
static const char proto_path_message[] = "(required) tnn proto file path";
static const char model_path_message[] = "(required) tnn model file path";
static const char input_path_message[] = "(required) input file path";

void ShowUsage(const char* exe, bool input_requird=true);
bool ParseAndCheckCommandLine(int argc, char* argv[], bool input_require=true);

#endif