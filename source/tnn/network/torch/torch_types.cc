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

#include "tnn/network/torch/torch_types.h"

#include <memory>
#include <map>
#include <set>
#include <regex>
#include <mutex>
#include <stdlib.h>

#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/core/macro.h"

#include <torch/script.h>

namespace TNN_NS {

static std::map<c10::TypeKind, TypeKindCloset> type_kind_closet = 
{
   {c10::TypeKind::TupleType, TypeKindCloset(c10::TypeKind::TupleType, '(', ')')},
   {c10::TypeKind::ListType,  TypeKindCloset(c10::TypeKind::ListType,  '[', ']')},
   {c10::TypeKind::DictType,  TypeKindCloset(c10::TypeKind::DictType,  '[', ']')},
};

static std::map<c10::TypeKind, const char *> type_kind_regex = 
{
   {c10::TypeKind::IntType,     "[0-9]+"},
   {c10::TypeKind::FloatType,   "[+-]?([0-9]*\\.?[0-9]+|[0-9]+\\.?[0-9]*)([eE][+-]?[0-9]+)?"},
   {c10::TypeKind::StringType,  "[^\\\\\\(\\)\\[\\]\\{\\}]+"},
};

std::mutex g_map_mutex;

std::string escape(char c) {
    switch (c) {
        case '(':
        case ')':
        case ']':
        case '[':
        case '{':
        case '}':
        case '\\':
        case '^':
        case '.':
        case '|':
            return std::string("\\") + std::string(1, c);
            break;
        default:
            break;
    }
    return std::string(1, c);
}

c10::TypeKind key_type(c10::TypePtr type) {
    switch(type->kind()) {
        case c10::TypeKind::ListType:
        case c10::TypeKind::TupleType:
            return c10::TypeKind::IntType; 
        case c10::TypeKind::DictType:
            return type->cast<c10::DictType>()->getKeyType()->kind();
    }
    return c10::TypeKind::StringType;
}

TypeKindMatcher::TypeKindMatcher(c10::TypePtr type, SubStr type_str) {
    std::lock_guard<std::mutex> guard(g_map_mutex);
    auto kind = type->kind();
    auto key_kind = key_type(type);
    valid_ = false;
    if (type_kind_closet.find(kind) == type_kind_closet.end()) {
        // TODO support other value types (e.g. single float or int type)
        valid_ = (kind == c10::TypeKind::TensorType 
                    && type_str.len() == 0);
        return;
    }
    if (type_kind_regex.find(key_kind) == type_kind_regex.end()) {
        return;
    }

    char pattern[200];
    snprintf(pattern, 200, "^(%s(%s)%s).*$", 
        escape(type_kind_closet[kind].start).c_str(),
        type_kind_regex[key_kind],
        escape(type_kind_closet[kind].end).c_str()
        );

    std::regex key_regex(pattern);

    auto s = type_str.str();
    // printf("TypeKindMatcher pattern:%s, type_str:%s\n", pattern, s.c_str());
    std::smatch key_match;
    if(std::regex_search(s, key_match, key_regex)) {
        valid_ = key_match.size() == 3;
    }

    if (valid_) {
        prefix_ = SubStr(type_str, 0, key_match[1].length());
        suffix_ = SubStr(type_str, key_match[1].length());
        key_    = SubStr(type_str, 1, key_match[2].length());
        if (key_kind == c10::TypeKind::IntType) {
            offset_ = atoi(key_.str().c_str());
        }
    }

    /*
    printf("prefix:%s key:%s int:%d suffix:%s\n",
        prefix_.str().c_str(),
        key_.str().c_str(),
        offset_,
        suffix_.str().c_str()
        );
        */
}

SubStr JitTypeMatcher::take_name(SubStr full_name) {
    std::lock_guard<std::mutex> guard(g_map_mutex);

    char pattern[200];
    snprintf(pattern, 200, "^(%s).*$", type_kind_regex[c10::TypeKind::StringType]);
    std::regex name_regex(pattern);

    auto s = full_name.str();
    // printf("pattern:%s, full_name:%s\n", pattern, s.c_str());
    std::smatch name_match;
    if(std::regex_search(s, name_match, name_regex)) {
        value_name_ = SubStr(full_name, 0, name_match[1].length());
    }

    auto ret = SubStr(full_name, value_name_.len());
    // printf("value_name:%s type_str:%s\n", value_name_.str().c_str(), ret.str().c_str());
    return ret;
}

void JitTypeMatcher::match(SubStr type_str) {
    if (type_str.len() == 0) {
        // TODO : add support for value_types ( eg. single float or int value)
        valid_ = type_->kind() == c10::TypeKind::TensorType;
        return;
    }

    TypeKindMatcher matcher(type_, type_str);
    key_ = matcher.key();
    suffix_ = matcher.suffix();
    // type_str reaches the end
    if (matcher.valid() && suffix_.len() == 0) {
        valid_ = true;
        return;
    }

    auto elems = elements();
    if (elems.size() == 0) {
        valid_ = false;
        return;
    }

    // found a match key, and type_str has remains
    if (matcher.valid()) {
        c10::TypePtr next_type = elements()[0];
        // direct use the targeting type 
        if (type_->kind() == c10::TypeKind::TupleType && matcher.offset() >= 0) {
            next_type = elements()[matcher.offset()];
        }
        next_ = std::make_shared<JitTypeMatcher>(next_type, suffix_, value_name_, matcher.prefix());
        valid_ = next_->valid();
    }
}

JitTypeMatcherPtr JitTypeMatcher::next() {
    return next_;
}

int JitTypeMatcher::id_from_name(std::string full_name) {

    auto matcher = JitTypeMatcher::create(c10::AnyType::create(), full_name);
    auto name = matcher->value_name_.str();

    std::lock_guard<std::mutex> guard(g_map_mutex);

    char pattern[200];
    snprintf(pattern, 200, "^%s_([0-9]+)$", type_kind_regex[c10::TypeKind::StringType]);
    std::regex id_regex(pattern);

    // printf("pattern:%s, value_name_:%s\n", pattern, name.c_str());
    std::smatch id_match;
    std::regex_search(name, id_match, id_regex);
    if (id_match.size() >= 2) {
        return atoi(id_match[1].str().c_str());
    }

    return -1;
}

std::vector<c10::TypePtr> JitTypeMatcher::elements() {
    std::vector<c10::TypePtr> ret;
    switch(type_->kind()) {
        case c10::TypeKind::TupleType : 
            for(auto t : type_->cast<c10::TupleType>()->elements()) {
                ret.push_back(t);
            }
            break;
        case c10::TypeKind::ListType: 
            ret.push_back(type_->cast<c10::ListType>()->getElementType());
            break;
        case c10::TypeKind::DictType: 
            ret.push_back(type_->cast<c10::DictType>()->getValueType());
            break;
        default:
            break;
    }

    return ret;
}

}