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

#ifndef TNNCONVERTER_SRC_FLAGS_H_
#define TNNCONVERTER_SRC_FLAGS_H_
#include "gflags/gflags.h"

namespace TNN_CONVERTER {

static const char help_message[] = "print a usage message.";

static const char tf_path_message[] = "specify model path: <the>/<path>/<to>/<test.tflite>.";

static const char output_dir_message[] =
    "Specify the output directory of the converted model: <the>/<path>/<to>/<directory>.";

static const char model_type_message[] = "specify model type: Caffe, TF, TFLite.";

static const char save_path_message[] = "Specify the save path of the results after TNN inference";

DECLARE_bool(h);

DECLARE_string(mp);

DECLARE_string(od);

DECLARE_string(mt);

DECLARE_string(sp);

}  // namespace TNN_CONVERTER

#endif  // TNNCONVERTER_SRC_FLAGS_H_
