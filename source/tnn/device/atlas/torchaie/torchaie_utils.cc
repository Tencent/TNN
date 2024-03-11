// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "tnn/device/atlas/torchaie/torchaie_utils.h"

namespace TNN_NS {

torch_aie::TensorFormat AieTensorFormatConverter(std::string format_str) {
    torch_aie::TensorFormat format = torch_aie::TensorFormat::UNKNOWN;

    if (format_str == "NCHW") {
        format = torch_aie::TensorFormat::NCHW;
    } else if (format_str == "NHWC") {
        format = torch_aie::TensorFormat::NHWC;
    } else if (format_str == "ND") {
        format = torch_aie::TensorFormat::ND;
    } else if (format_str == "NC1HWC0") {
        format = torch_aie::TensorFormat::NC1HWC0;
    } else {
        LOGE("not support torchaie tensor format (%s)\n", format_str.c_str());
    }

    return format;
}

torch_aie::DataType AieDataTypeConverter(DataType type) {
    torch_aie::DataType aie_type = torch_aie::DataType::UNKNOWN;
    switch (type)
    {
    case DATA_TYPE_FLOAT:
        aie_type = torch_aie::DataType::FLOAT;
        break;
    case DATA_TYPE_HALF:
        aie_type = torch_aie::DataType::FLOAT16;
        break;
    case DATA_TYPE_INT8:
        aie_type = torch_aie::DataType::INT8;
        break;
    case DATA_TYPE_INT32:
        aie_type = torch_aie::DataType::INT32;
        break;
    case DATA_TYPE_INT64:
        aie_type = torch_aie::DataType::INT64;
        break;
    case DATA_TYPE_UINT32:
        aie_type = torch_aie::DataType::UINT32;
        break;
    case DATA_TYPE_UINT8:
        aie_type = torch_aie::DataType::UINT8;
        break;
    default:
        LOGE("not support convert from data type (%d) to torchaie data type\n", type);
        break;
    }
    return aie_type;
}

}  //  namespace TNN_NS
