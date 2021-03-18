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

#include "tnn/network/tensorrt/dimension_expr.h"

#include <string.h>
#include <string>
#include <stdio.h>

#include "NvInfer.h"

#include "tnn/network/tensorrt/utils.h"
#include "tnn/core/macro.h"

namespace TNN_NS {

DimensionExpr::DimensionExpr(const nvinfer1::IDimensionExpr * idimexpr, nvinfer1::IExprBuilder &builder) :
                    expr_(idimexpr), builder_(builder)
{}

DimensionExpr::DimensionExpr(const int v, nvinfer1::IExprBuilder &builder) :
                    builder_(builder)
{
    expr_ = builder.constant(v);
}

// @brief virtual destructor
DimensionExpr::~DimensionExpr() {}

}
