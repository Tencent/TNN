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

#include "tnn/utils/string_utils.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

std::string UcharToString(const unsigned char *buffer, int length){
    std::string param;
    for(int i = 0; i<length; i++){
        param += buffer[i];
    }
    return param;
}

template <>
std::string ToString<float>(float value) {
    std::ostringstream os;
    os << std::showpoint << value;
    return os.str();
}
}  // namespace TNN_NS
