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

#ifndef TNN_SOURCE_TNN_UTILS_STRING_FORMATTER_H_
#define TNN_SOURCE_TNN_UTILS_STRING_FORMATTER_H_

#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "tnn/core/macro.h"
#include "tnn/core/mat.h"

namespace TNN_NS {

std::string DoubleToString(double val);

std::string DoubleToStringFilter(double val);

std::string MatTypeToString(MatType mat_type);

std::string DimsToString(std::vector<int> dims);

template <typename Int>
std::string IntToString(Int val) {
    static_assert(std::is_integral<Int>::value, "Integral type required!");
    std::stringstream stream;
    stream << val;
    return stream.str();
}

template <typename Int>
std::string IntToStringFilter(Int val) {
    if (static_cast<Int>(0) == val) {
        return "";
    } else {
        return IntToString(val);
    }
}

template <typename T>
std::string VectorToString(std::vector<T> val) {
    if (val.empty())
        return "";

    std::stringstream stream;
    stream << "[";
    for (int i = 0; i < val.size(); ++i) {
        stream << val[i];
        if (i != val.size() - 1)
            stream << ",";
    }
    stream << "]";
    return stream.str();
}

std::vector<std::pair<std::string, std::vector<float>>> SortMapByValue(std::map<std::string, std::vector<float>> &map);

class StringFormatter {
public:
    static std::string Table(const std::string &title, const std::vector<std::string> &header,
                             const std::vector<std::vector<std::string>> &data);
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_UTILS_STRING_FORMATTER_H_
