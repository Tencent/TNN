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

#include "tf_lite_op_converter.h"

TFLiteOpConverterSuit* TFLiteOpConverterSuit::_unique_suit = nullptr;

TFLiteOpConverter* TFLiteOpConverterSuit::search(const tflite::BuiltinOperator op_index) {
    auto iter = _tf_lite_op_converters.find(op_index);
    if (iter == _tf_lite_op_converters.end()) {
        return nullptr;
    }

    return iter->second;
}

TFLiteOpConverterSuit* TFLiteOpConverterSuit::get() {
    if (_unique_suit == nullptr) {
        _unique_suit = new TFLiteOpConverterSuit;
    }

    return _unique_suit;
}

TFLiteOpConverterSuit::~TFLiteOpConverterSuit() {
    for (auto& it : _tf_lite_op_converters) {
        delete it.second;
    }
    _tf_lite_op_converters.clear();
}

void TFLiteOpConverterSuit::insert(TFLiteOpConverter* t, tflite::BuiltinOperator op_index) {
    _tf_lite_op_converters.insert(std::make_pair(op_index, t));
}
