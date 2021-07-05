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

#include "ncnn_param_utils.h"

#include <stdlib.h>
#include <string>

#include "tnn/core/status.h"
#include "tnn/utils/split_utils.h"

namespace TNN_NS {

namespace ncnn {

    int GetInt(str_dict param, int index, int default_value) {
        if (param.find(index) == param.end()) {
            return default_value;
        }
        return atoi(param[index].c_str());
    }

    float GetFloat(str_dict param, int index, float default_value) {
        if (param.find(index) == param.end()) {
            return default_value;
        }
        return static_cast<float>(atof(param[index].c_str()));
    }

    str_arr GetStrList(str_dict param, int index) {
        str_arr param_vec;
        if (param.find(index) == param.end()) {
            return param_vec;
        }

        char *str = const_cast<char *>(param[index].c_str());

        Status ret = SplitUtils::SplitStr(str, param_vec, ",", true, true);
        if (ret != TNN_OK) {
            LOGE("split param list failed\n");
            return param_vec;
        }
        return param_vec;
    }

    std::vector<float> GetFloatList(str_dict param, int index) {
        std::vector<float> float_result;
        str_arr param_vec = GetStrList(param, index);
        // start from offset 1; first element is the length
        for (size_t i = 1; i < param_vec.size(); i++) {
            float_result.push_back(atof(param_vec[i].c_str()));
        }
        return float_result;
    }

    std::vector<int> GetIntList(str_dict param, int index) {
        std::vector<int> int_result;
        str_arr param_vec = GetStrList(param, index);
        // start from the offset 1; first element is the length
        for (size_t i = 1; i < param_vec.size(); i++) {
            int_result.push_back(atoi(param_vec[i].c_str()));
        }
        return int_result;
    }

    bool HasField(str_dict param, int index) {
        return param.find(index) != param.end();
    }

}  // namespace ncnn

}  // namespace TNN_NS
