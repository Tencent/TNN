// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "tnn/core/any.h"

namespace TNN_NS {

Any::Any(Any const& other) : content_(other.content_ ? other.content_->clone() : nullptr) {}

Any::Any(Any&& other) noexcept : content_(std::move(other.content_)) {
    other.content_ = nullptr;
}

Any::~Any() {
    reset();
}

Any& Any::operator=(const Any& rhs) {
    Any(rhs).swap(*this);
    return *this;
}

Any& Any::operator=(Any&& rhs) noexcept {
    Any(std::move(rhs)).swap(*this);
    return *this;
}

void Any::reset() noexcept {
    delete content_;
    content_ = nullptr;
}

void Any::swap(Any& other) noexcept {
    std::swap(content_, other.content_);
}

bool Any::has_value() const noexcept {
    return content_ != nullptr;
}

const std::type_info & Any::type() const noexcept {
    return has_value() ? content_->type() : typeid(void);
}

}  // namespace TNN_NS
