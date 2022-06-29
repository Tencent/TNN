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

#include "tnn/network/openvino/utils.h"

namespace TNN_NS {

ngraph::element::Type_t ConvertToOVDataType(DataType type) {
    switch (type) {
        case DATA_TYPE_FLOAT:
            return ngraph::element::Type_t::f32;
        case DATA_TYPE_HALF:
            return ngraph::element::Type_t::f16;
        case DATA_TYPE_INT64:
            return ngraph::element::Type_t::i64;
        case DATA_TYPE_INT32:
            return ngraph::element::Type_t::i32;
        case DATA_TYPE_INT8:
            return ngraph::element::Type_t::i8;
        default:
            return ngraph::element::Type_t::f32;
    }
}

DataType ConvertOVPrecisionToDataType(const InferenceEngine::Precision &precision) {
    switch (precision.getPrecVal()) {
        case InferenceEngine::Precision::FP32:
            return DATA_TYPE_FLOAT;
        case InferenceEngine::Precision::FP16:
            return DATA_TYPE_HALF;
        case InferenceEngine::Precision::I32:
            return DATA_TYPE_INT32;
        case InferenceEngine::Precision::I8:
            return DATA_TYPE_INT8;
        default:
            return DATA_TYPE_FLOAT;
    }
}

InferenceEngine::Precision ConvertOVTypeToPrecision(ngraph::element::Type_t type) {
    switch (type) {
        case ngraph::element::Type_t::f32:
            return InferenceEngine::Precision::FP32;
        case ngraph::element::Type_t::f16:
            return InferenceEngine::Precision::FP16;
        case ngraph::element::Type_t::i32:
            return InferenceEngine::Precision::I32;
        case ngraph::element::Type_t::i8:
            return InferenceEngine::Precision::I8;
        default:
            return InferenceEngine::Precision::FP32;
    }
}

std::shared_ptr<ngraph::op::Constant> ConvertToConstNode(RawBuffer *buffer) {
    ngraph::Shape constShape;

    if (buffer->GetBufferDims().size() == 0 && buffer->GetBytesSize() == 0) {
        return std::make_shared<ngraph::op::Constant>();
    }

    for (auto &iter : buffer->GetBufferDims()) {
        constShape.push_back(iter);
    }

    return std::make_shared<ngraph::op::Constant>(ConvertToOVDataType(buffer->GetDataType()), constShape,
                                                  buffer->force_to<void *>());
}

}  //  namespace TNN_NS
