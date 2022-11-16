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

#include "tnn/utils/data_type_utils.h"
#include <limits.h>
#include "tnn/core/macro.h"

namespace TNN_NS {

int DataTypeUtils::GetBytesSize(DataType data_type) {
    if (data_type == DATA_TYPE_FLOAT) {
        return 4;
    } else if (data_type == DATA_TYPE_HALF) {
        return 2;
    } else if (data_type == DATA_TYPE_BFP16) {
        return 2;
    } else if (data_type == DATA_TYPE_INT8) {
        return 1;
    } else if (data_type == DATA_TYPE_INT32) {
        return 4;
    } else if (data_type == DATA_TYPE_INT64) {
        return 8;
    } else if (data_type == DATA_TYPE_UINT32) {
        return 4;
    } else {
        LOGE("GetBytes Undefined \n");
        return -1;
    }
}

std::string DataTypeUtils::GetDataTypeString(DataType data_type) {
    if (data_type == DATA_TYPE_FLOAT) {
        return "float";
    } else if (data_type == DATA_TYPE_HALF) {
        return "half";
    } else if (data_type == DATA_TYPE_BFP16) {
        return "bfp16";
    } else if (data_type == DATA_TYPE_INT8) {
        return "int8";
    } else if (data_type == DATA_TYPE_INT64) {
        return "int64";
    } else if (data_type == DATA_TYPE_INT32) {
        return "int32";
    } else {
        return "";
    }
}

int DataTypeUtils::SaturateCast(long long int data) {
    return (int)((uint64_t)(data - INT_MIN) <= (uint64_t)UINT_MAX ? data : data > 0 ? INT_MAX : INT_MIN);
}

}  // namespace TNN_NS
