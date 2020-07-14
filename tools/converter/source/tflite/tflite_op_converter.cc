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

#include "tflite_op_converter.h"

namespace TNN_CONVERTER {

TFLiteOpConverterManager* TFLiteOpConverterManager::get() {
    if (tf_lite_op_converter_manager_ == nullptr) {
        tf_lite_op_converter_manager_ = new TFLiteOpConverterManager;
    }
    return tf_lite_op_converter_manager_;
}
TFLiteOpConverter* TFLiteOpConverterManager::search(const tflite::BuiltinOperator op_index) {
    auto iter = tf_lite_op_converter_map_.find(op_index);
    if (iter == tf_lite_op_converter_map_.end()) {
        return nullptr;
    }
    return iter->second;
}

TFLiteOpConverterManager::~TFLiteOpConverterManager() {
    for (auto& it : tf_lite_op_converter_map_) {
        delete it.second;
    }
    tf_lite_op_converter_map_.clear();
}

void TFLiteOpConverterManager::insert(const tflite::BuiltinOperator op_index, TFLiteOpConverter* t) {
    tf_lite_op_converter_manager_->insert(std::make_pair(op_index, t));
}

}  // namespace TNN_CONVERTER
