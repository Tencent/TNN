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

// author: sanerzheng@tencent.com

#ifndef TNN_SOURCE_TNN_TRAIN_OPERATIONS_OP_TYPE_H
#define TNN_SOURCE_TNN_TRAIN_OPERATIONS_OP_TYPE_H

namespace TNN_NS {
namespace train {

// @brief all op types, used in train module, but may be used in predict modules;  
enum OpType {
    OP_ElEMENT = 1000,
}; 

enum ElementOpType {
    Unkown = 0,
    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
    Log = 5,
    Neg = 6,
    RSign = 7

}; 

} // namespace train
} // namespace TNN_NS
#endif  //TNN_SOURCE_TNN_TRAIN_OPERATIONS_OP_TYPE_H 