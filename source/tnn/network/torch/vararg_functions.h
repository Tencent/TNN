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

#ifndef TNN_SOURCE_NETWORK_TNNTORCH_VARARG_FUNCTIONS_H
#define TNN_SOURCE_NETWORK_TNNTORCH_VARARG_FUNCTIONS_H

#pragma once
#include <ATen/core/List.h>
#include <ATen/core/functional.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>

namespace torch {
namespace jit {

void tupleUnpack(Stack& stack);

void format(Stack& stack, size_t num_inputs);

void einsum(Stack& stack, size_t num_inputs);

void percentFormat(Stack& stack, size_t num_inputs);

void listUnpack(Stack& stack, size_t num_outputs);

void tupleConstruct(Stack& stack, size_t num_inputs);

void namedTupleConstruct(
    Stack& stack,
    at::TupleTypePtr type,
    size_t num_inputs);

void listConstruct(
    Stack& stack,
    const at::ListType& list_type,
    size_t num_inputs);

void dictConstruct(Stack& stack, const at::DictType& type, size_t num_inputs);

void createObject(Stack& stack, const at::ClassTypePtr& type);

void isinstance(Stack& stack, at::ArrayRef<at::TypePtr> types);

void tupleSlice(Stack& stack, size_t begin, size_t end);

void dequantize(Stack& stack);

} // namespace jit
} // namespace torch

#endif
