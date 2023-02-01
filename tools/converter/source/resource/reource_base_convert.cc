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

#include "reource_base_convert.h"
namespace TNN_CONVERTER {

ResourceConvertManager *ResourceConvertManager::resource_convert_manager_ = nullptr;

TNN_CONVERTER::ResourceConvertManager *TNN_CONVERTER::ResourceConvertManager::get() {
    if (resource_convert_manager_ == nullptr) {
        resource_convert_manager_ = new ResourceConvertManager;
    }
    return resource_convert_manager_;
}
void TNN_CONVERTER::ResourceConvertManager::insert(const std::string &tnn_op_name,
                                                   TNN_CONVERTER::ResourceBaseConvert *resource_convert) {
    resource_convert_map_.insert(std::make_pair(tnn_op_name, resource_convert));
}

TNN_CONVERTER::ResourceBaseConvert *TNN_CONVERTER::ResourceConvertManager::search(const std::string &tnn_op_name) {
    auto iter = resource_convert_map_.find(tnn_op_name);
    if (iter == resource_convert_map_.end()) {
        return nullptr;
    }
    return iter->second;
}
ResourceConvertManager::~ResourceConvertManager() {
    for (auto &iter : resource_convert_map_) {
        delete iter.second;
    }
    resource_convert_map_.clear();
    delete resource_convert_manager_;
}
}  // namespace TNN_CONVERTER
