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

#ifndef TNN_SOURCE_TNN_EXTERN_WRAPPER_FOREIGN_TENSOR_H_
#define TNN_SOURCE_TNN_EXTERN_WRAPPER_FOREIGN_TENSOR_H_

#include <cstdint>
#include <map>
#include <string>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"

namespace TNN_NS {

//@brief Base Type of a ForeignTensor 
class ForeignTensor {
public:
    explicit ForeignTensor(){};

    // @brief virtual destructor
    virtual ~ForeignTensor(){};
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_EXTERN_WRAPPER_FOREIGN_TENSOR_H_
