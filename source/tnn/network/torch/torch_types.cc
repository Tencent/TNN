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
#include <sstream>
#include <stdlib.h>

#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/core/macro.h"
#include "tnn/network/torch/torch_utils.h"

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
            return type->expect<c10::DictType>()->getKeyType()->kind();
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

int JitTypeMatcher::idFromName(std::string full_name) {

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
            for(auto t : type_->expect<c10::TupleType>()->elements()) {
                ret.push_back(t);
            }
            break;
        case c10::TypeKind::ListType: 
            ret.push_back(type_->expect<c10::ListType>()->getElementType());
            break;
        case c10::TypeKind::DictType: 
            ret.push_back(type_->expect<c10::DictType>()->getValueType());
            break;
        default:
            break;
    }

    return ret;
}

Status attach_tensor_to_ivalue(c10::IValue &ivalue, at::TensorPtr tensor, JitTypeMatcherPtr matcher) {
    TNN_CHECK(ivalue.type()->isSubtypeOf(matcher->type()), "IValue type %s != matcher type %s", 
                                                           ivalue.type()->annotation_str().c_str(), 
                                                           matcher->type()->annotation_str().c_str());
    switch (matcher->type()->kind()) {
        case c10::TypeKind::TupleType : 
            {
                TNN_CHECK(matcher->hasNext(), "Type(%s) has no next!", matcher->type()->annotation_str().c_str());
                auto tuple = ivalue.toTuple();
                int idx = matcher->intKey();
                while(tuple->elements().size() <= idx) {
                    c10::IValue iv;
                    RETURN_ON_FAIL(CreateIValueFromTypePtr(iv, matcher->next()->type()));
                    tuple->elements().push_back(iv);
                }
                auto element = tuple->elements()[idx];
                RETURN_ON_FAIL(attach_tensor_to_ivalue(element, tensor, matcher->next()));
                tuple->elements()[idx] = element;
                ivalue = tuple;
                break;
            }
        case c10::TypeKind::ListType: 
            {
                TNN_CHECK(matcher->hasNext(), "Type(%s) has no next!", matcher->type()->annotation_str().c_str());
                int idx = matcher->intKey();

                auto list = ivalue.toList();
                while(list.size() <= idx) {
                    c10::IValue iv;
                    RETURN_ON_FAIL(CreateIValueFromTypePtr(iv, matcher->next()->type()));
                    list.push_back(iv);
                }
                c10::IValue element = list[idx];
                RETURN_ON_FAIL(attach_tensor_to_ivalue(element, tensor, matcher->next()));
                list[idx] = element;
                ivalue = list;
                break;
            }
        case c10::TypeKind::DictType: 
            {
                c10::IValue key;
                if (key_type(matcher->type()) == c10::TypeKind::StringType) {
                    key = c10::IValue(matcher->strKey());
                } else if (key_type(matcher->type()) == c10::TypeKind::IntType) {
                    key = c10::IValue(matcher->intKey());
                } else if (key_type(matcher->type()) == c10::TypeKind::FloatType) {
                    key = c10::IValue(matcher->floatKey());
                }

                TNN_CHECK(matcher->hasNext(), "Type(%s) has no next!", matcher->type()->annotation_str().c_str());

                auto dict = ivalue.toGenericDict();
                if (!dict.contains(key)) {
                    c10::IValue iv;
                    RETURN_ON_FAIL(CreateIValueFromTypePtr(iv, matcher->next()->type()));
                    dict.insert(key, iv);
                }

                c10::IValue element = dict.at(key);
                RETURN_ON_FAIL(attach_tensor_to_ivalue(element, tensor, matcher->next()));
                dict.insert_or_assign(key, element);
                ivalue = dict;
                break;
            }
        case c10::TypeKind::TensorType:
            {
                ivalue = c10::IValue(std::move(*tensor));
            }
            break;
        default:
            // printf("matcher type:%s kind:%d, str:%s\n", matcher->type()->annotation_str().c_str(), matcher->type()->kind(), typeKindToString(matcher->type()->kind()));
            return Status(TNNERR_PARAM_ERR, "Unsupported type in function attach_tensor_to_ivalue");
    }

    return TNN_OK;
}

Status IValueRouter::attach(c10::IValue &ivalue, at::TensorPtr tensor) {
    TNN_CHECK(matcher_->valid(), "Input name \"%s\" not matched with the target type:\"%s\"", full_name_.c_str(), matcher_->type()->annotation_str().c_str());
    RETURN_ON_FAIL(attach_tensor_to_ivalue(ivalue, tensor, matcher_));
    return TNN_OK; 
}

template<typename T>
Status keyToString(T key, c10::TypeKind kind, std::string &str) {
    std::lock_guard<std::mutex> guard(g_map_mutex);

    std::stringstream ss;

    if (type_kind_closet.find(kind) != type_kind_closet.end()) {
        ss << type_kind_closet[kind].start << key <<type_kind_closet[kind].end;
        str = ss.str();
    } else {
        printf("keyToString typekind:%s\n", typeKindToString(kind));
        return Status(TNNERR_PARAM_ERR, "Unsupported key type from function keyToString.");
    }
    return TNN_OK;
}

Status eval(c10::IValue &ivalue, std::string &value) 
{
    std::stringstream ss;
    if (ivalue.isInt()) {
        ss << ivalue.toInt();
    } else if (ivalue.isString()) {
        ss << ivalue.toStringRef();
    } else if (ivalue.isDouble()) {
        ss << ivalue.toDouble();
    } else {
        return Status(TNNERR_PARAM_ERR, "Unsupport ivalue type from function eval");
    }
    value = ss.str();
    return TNN_OK;
}


#define IDX_TO_STR(key, kind, str)              \
    std::string str;                            \
    RETURN_ON_FAIL(keyToString(key, kind, str))

#define SUB_TRAVERSE(ele, prefix, names)        \
    std::vector<std::string> names;             \
    RETURN_ON_FAIL(traverse(ele, prefix, names))

Status traverse(c10::IValue &ivalue, std::string prefix, std::vector<std::string> &ret) {
    // printf("traverse got type:%s prefix:%s\n", ivalue.type()->annotation_str().c_str(), prefix.c_str());
    std::vector<std::string> names;
    switch (ivalue.type()->kind()) {
        case c10::TypeKind::TupleType:
        {
            auto tuple = ivalue.toTuple();
            auto elements = tuple->elements();
            for(int i=0;i<elements.size();i++) {
                IDX_TO_STR(i, c10::TypeKind::TupleType, path);
                SUB_TRAVERSE(elements[i], prefix + path, sub_names);
                names.insert(names.end(), sub_names.begin(), sub_names.end());
            }
            break;
        }
        case c10::TypeKind::ListType:
        {
            auto list = ivalue.toList();
            for(int i=0;i<list.size();i++) {
                c10::IValue ele = list[i];
                IDX_TO_STR(i, c10::TypeKind::ListType, path);
                SUB_TRAVERSE(ele, prefix + path, sub_names);
                names.insert(names.end(), sub_names.begin(), sub_names.end());
            }
            break;
        }
        case c10::TypeKind::DictType:
        {
            auto dict = ivalue.toGenericDict();
            int i = 0;
            for(auto it= dict.begin();it != dict.end();it++) 
            {
                auto key = it->key();
                auto value = it->value();

                std::string str_key;
                RETURN_ON_FAIL(eval(key, str_key));
                IDX_TO_STR(str_key, c10::TypeKind::DictType, path);

                SUB_TRAVERSE(value, prefix + path, sub_names);
                names.insert(names.end(), sub_names.begin(), sub_names.end());
            }
            break;
        }
        case c10::TypeKind::TensorType:
        {
            names.push_back(prefix);
            break;
        }
        default:
            return Status(TNNERR_PARAM_ERR, "Unsupported type from function traverse");
    }
    ret = names; 
    return TNN_OK;;
}

#undef IDX_TO_STR
#undef SUB_TRAVERSE

Status IValueRouter::getAllTensorNames(c10::IValue &ivalue, std::string prefix, std::vector<std::string> &names) {
    RETURN_ON_FAIL(traverse(ivalue, prefix, names));
    return TNN_OK;  
}

Status route_ivalue_to_tensor(c10::IValue &ivalue, at::TensorPtr &tensor, JitTypeMatcherPtr matcher) {
    TNN_CHECK(ivalue.type()->isSubtypeOf(matcher->type()), "IValue type %s != matcher type %s", 
                                                           ivalue.type()->annotation_str().c_str(), 
                                                           matcher->type()->annotation_str().c_str());
    switch (matcher->type()->kind()) {
        case c10::TypeKind::TupleType : 
            {
                TNN_CHECK(matcher->hasNext(), "Type(%s) has no next!", matcher->type()->annotation_str().c_str());
                auto tuple = ivalue.toTuple();
                int idx = matcher->intKey();

                TNN_CHECK(tuple->elements().size() > idx, "idx(%d) larger than Tuple size(%lu)!", idx, tuple->elements().size());

                auto element = tuple->elements()[idx];
                RETURN_ON_FAIL(route_ivalue_to_tensor(element, tensor, matcher->next()));
                break;
            }
        case c10::TypeKind::ListType: 
            {
                TNN_CHECK(matcher->hasNext(), "Type(%s) has no next!", matcher->type()->annotation_str().c_str());
                int idx = matcher->intKey();

                auto list = ivalue.toList();
                TNN_CHECK(list.size() > idx, "idx(%d) larger than List size(%lu)!", idx, list.size());

                c10::IValue element = list[idx];
                RETURN_ON_FAIL(route_ivalue_to_tensor(element, tensor, matcher->next()));
                break;
            }
        case c10::TypeKind::DictType: 
            {
                c10::IValue key;
                if (key_type(matcher->type()) == c10::TypeKind::StringType) {
                    key = c10::IValue(matcher->strKey());
                } else if (key_type(matcher->type()) == c10::TypeKind::IntType) {
                    key = c10::IValue(matcher->intKey());
                } else if (key_type(matcher->type()) == c10::TypeKind::FloatType) {
                    key = c10::IValue(matcher->floatKey());
                }

                TNN_CHECK(matcher->hasNext(), "Type(%s) has no next!", matcher->type()->annotation_str().c_str());

                auto dict = ivalue.toGenericDict();
                TNN_CHECK(dict.contains(key), "dict(%s) not contains specified key!", matcher->type()->annotation_str().c_str());

                c10::IValue element = dict.at(key);
                RETURN_ON_FAIL(route_ivalue_to_tensor(element, tensor, matcher->next()));
                break;
            }
        case c10::TypeKind::TensorType:
            {
                tensor = std::make_shared<at::Tensor>(ivalue.toTensor());
            }
            break;
        default:
            // printf("matcher type:%s kind:%d, str:%s\n", matcher->type()->annotation_str().c_str(), matcher->type()->kind(), typeKindToString(matcher->type()->kind()));
            return Status(TNNERR_PARAM_ERR, "Unsupported type in function route_ivalue_to_tensor");
    }

    return TNN_OK;
}

Status IValueRouter::route(c10::IValue &ivalue, at::TensorPtr &tensor) {
    TNN_CHECK(matcher_->valid(), "blob name \"%s\" not matched with the target type:\"%s\"", full_name_.c_str(), matcher_->type()->annotation_str().c_str());
    RETURN_ON_FAIL(route_ivalue_to_tensor(ivalue, tensor, matcher_));

    return TNN_OK;
}


}