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

#ifndef TNN_SOURCE_TNN_NETWORK_TORCH_TYPES_H_
#define TNN_SOURCE_TNN_NETWORK_TORCH_TYPES_H_

#include <string>
#include <map>
#include <memory>

#include "tnn/core/common.h"
#include "tnn/core/status.h"
#include "tnn/core/macro.h"

#include <torch/script.h>

namespace TNN_NS {

/*
    Tuple:  (int)
    Dict:   [key_type]
    List:   [5]

    eg: [Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]]
    output0(0)[xxx]
    output0(1)[1][xxx]
*/

struct TypeKindCloset
{
    TypeKindCloset(c10::TypeKind k, char s, char e): 
                kind(k), start(s), end(e) {}; 
    TypeKindCloset(){};
    c10::TypeKind kind;
    char start;
    char end; 
};

struct SubStr 
{
/* TODO :
    add checking macro  
    check if offset and len are in range in all functions
 */
public:
    SubStr(std::string str) {
        new (this) SubStr(str, 0, str.length());
    }

    SubStr(std::string str, int of, int l) {
        sptr_ = std::make_shared<std::string>(str);
        offset_ = of;
        len_ = l;
    }

    SubStr(const SubStr &str, int of, int l=-1) :
        sptr_(str.sptr_), offset_(of+str.offset_), len_(l) {
        if (len_ == -1) {
            len_ = str.len() - of;
        }
    }

    SubStr():sptr_(nullptr), offset_(0), len_(0) {}

    std::string str() {
        if (sptr_ &&
            offset_ >= 0 && 
            offset_ < sptr_->length() &&
            len_ >= 0 && 
            offset_ + len_ <= sptr_->length()) 
        {
            return sptr_->substr(offset_, len_);
        }
        return std::string();
    }

    int len() const {
        return len_;
    }

    int offset() const {
        return offset_;
    }

private:
    std::shared_ptr<std::string> sptr_;
    int offset_ = 0;
    int len_ = 0;
};

struct TypeKindMatcher
{
public:
    explicit TypeKindMatcher(c10::TypePtr type, SubStr type_str);

    SubStr prefix() const {return prefix_;}
    SubStr key() const {return key_;}
    SubStr suffix() const {return suffix_;}
    int offset() const {return offset_;}
    bool valid() const {return valid_;}

private:
    SubStr prefix_;
    SubStr suffix_;
    SubStr key_;
    int offset_ = -1;
    bool valid_ = false;
};

struct JitTypeMatcher;
using JitTypeMatcherPtr = std::shared_ptr<JitTypeMatcher>;

struct JitTypeMatcher: std::enable_shared_from_this<JitTypeMatcher>
{
public: 
    static JitTypeMatcherPtr create(c10::TypePtr type, std::string full_name) {
        return std::make_shared<JitTypeMatcher>(type, full_name);
    }

    explicit JitTypeMatcher(c10::TypePtr type, std::string full_name)
    {
        new (this) JitTypeMatcher(type, SubStr(full_name));
    }

    explicit JitTypeMatcher(c10::TypePtr type, SubStr full_name) : type_(type)
    {
        auto type_str = take_name(full_name);
        match(type_str);
    }

    explicit JitTypeMatcher(c10::TypePtr type, SubStr type_str, SubStr value_name, 
                            SubStr prefix) 
                            : type_(type), value_name_(value_name), prefix_(prefix)
    {
        match(type_str);
    }

    bool valid() {return valid_;}

    JitTypeMatcherPtr next(); 

    std::vector<c10::TypePtr> elements();

    static int id_from_name(std::string);

private:

    void match(SubStr);
    SubStr take_name(SubStr);

    c10::TypePtr type_;
    SubStr value_name_;
    SubStr prefix_;
    SubStr key_;
    SubStr suffix_;
    bool valid_;

    JitTypeMatcherPtr next_;
};



}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_NETWORK_TORCH_TYPES_H_