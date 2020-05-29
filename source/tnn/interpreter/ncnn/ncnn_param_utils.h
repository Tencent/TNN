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

#ifndef TNN_SOURCE_TNN_INTERPRETER_NCNN_NCNN_PARAM_UTILS_H_
#define TNN_SOURCE_TNN_INTERPRETER_NCNN_NCNN_PARAM_UTILS_H_

#include "tnn/utils/split_utils.h"

#include <string>

namespace TNN_NS {

namespace ncnn {

    int GetInt(str_dict param, int index, int default_value = 0);

    float GetFloat(str_dict param, int index, float default_value = 0.f);

    std::vector<float> GetFloatList(str_dict param, int index);

    std::vector<int> GetIntList(str_dict param, int index);

    bool HasField(str_dict param, int index);

}  // namespace ncnn

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_NCNN_NCNN_PARAM_UTILS_H_
