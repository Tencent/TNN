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

#ifndef TNN_INCLUDE_TNN_UTILS_DATA_TYPE_UTILS_H_
#define TNN_INCLUDE_TNN_UTILS_DATA_TYPE_UTILS_H_

#include <string>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"

namespace TNN_NS {

class PUBLIC DataTypeUtils {
public:
    // @brief get bytes
    // @param data_tyep data type info
    static int GetBytesSize(DataType data_type);

    // @brief get string for DataType
    // @param data_tyep data type info
    static std::string GetDataTypeString(DataType data_type);
    
    // @brief safely cast int64 to int, int64_min to int_min and int64_max to int_max. avoid to cast int64_max to -1
    static int SaturateCast(long long int data);
};

}  // namespace TNN_NS

#endif  // TNN_INCLUDE_TNN_UTILS_DATA_TYPE_UTILS_H_
