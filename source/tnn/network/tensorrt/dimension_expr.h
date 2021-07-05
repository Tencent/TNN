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

#ifndef TNN_SOURCE_TNN_NETWORK_TENSORRT_DIMENSION_EXPR_H_
#define TNN_SOURCE_TNN_NETWORK_TENSORRT_DIMENSION_EXPR_H_

#include "NvInfer.h"

#include "tnn/core/status.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/extern_wrapper/foreign_tensor.h"

namespace TNN_NS {

// @brief Dimension builder helper class of tensorrt IDimensionExpr
class DimensionExpr {
public:
    explicit DimensionExpr(const nvinfer1::IDimensionExpr * idimexpr, nvinfer1::IExprBuilder &builder);

    explicit DimensionExpr(const int v, nvinfer1::IExprBuilder &builder);

    // @brief virtual destructor
    virtual ~DimensionExpr();

    const nvinfer1::IDimensionExpr* expr() const {
        return expr_;
    }

    // DimensionExpr with DimensionExpr
    DimensionExpr& operator=(const DimensionExpr& other)
    {
        expr_ = other.expr_;  
        builder_ = other.builder_;
        return *this;
    }

    // DimensionExpr with DimensionExpr
    friend DimensionExpr operator+(const DimensionExpr &lhs, const DimensionExpr& rhs)
    {
        nvinfer1::IExprBuilder& builder = lhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kSUM, *lhs.expr_, *rhs.expr_), builder);
    }

    friend DimensionExpr operator-(const DimensionExpr &lhs, const DimensionExpr& rhs)
    {
        nvinfer1::IExprBuilder& builder = lhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kSUB, *lhs.expr_, *rhs.expr_), builder);
    }

    friend DimensionExpr operator*(const DimensionExpr &lhs, const DimensionExpr& rhs)
    {
        nvinfer1::IExprBuilder& builder = lhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kPROD, *lhs.expr_, *rhs.expr_), builder);
    }

    friend DimensionExpr operator/(const DimensionExpr &lhs, const DimensionExpr& rhs)
    {
        nvinfer1::IExprBuilder& builder = lhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *lhs.expr_, *rhs.expr_), builder);
    }

    friend DimensionExpr ceil_div(const DimensionExpr &lhs, const DimensionExpr& rhs)
    {
        nvinfer1::IExprBuilder& builder = lhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kCEIL_DIV, *lhs.expr_, *rhs.expr_), builder);
    }

    // DimensionExpr with integer
    friend DimensionExpr operator+(const DimensionExpr &lhs, const int& rhs)
    {
        nvinfer1::IExprBuilder& builder = lhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kSUM, *lhs.expr_, *builder.constant(rhs)), builder);
    }

    friend DimensionExpr operator-(const DimensionExpr &lhs, const int& rhs)
    {
        nvinfer1::IExprBuilder& builder = lhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kSUB, *lhs.expr_, *builder.constant(rhs)), builder);
    }

    friend DimensionExpr operator*(const DimensionExpr &lhs, const int& rhs)
    {
        nvinfer1::IExprBuilder& builder = lhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kPROD, *lhs.expr_, *builder.constant(rhs)), builder);
    }

    friend DimensionExpr operator/(const DimensionExpr &lhs, const int& rhs)
    {
        nvinfer1::IExprBuilder& builder = lhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, *lhs.expr_, *builder.constant(rhs)), builder);
    }

    friend DimensionExpr ceil_div(const DimensionExpr &lhs, const int& rhs)
    {
        nvinfer1::IExprBuilder& builder = lhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kCEIL_DIV, *lhs.expr_, *builder.constant(rhs)), builder);
    }

    // integer with DimensionExpr
    friend DimensionExpr operator+( const int& lhs, const DimensionExpr &rhs)
    {
        nvinfer1::IExprBuilder& builder = rhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kSUM, *builder.constant(lhs), *rhs.expr_), builder);
    }

    friend DimensionExpr operator-( const int& lhs, const DimensionExpr &rhs)
    {
        nvinfer1::IExprBuilder& builder = rhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kSUB,  *builder.constant(lhs), *rhs.expr_), builder);
    }

    friend DimensionExpr operator*( const int& lhs, const DimensionExpr &rhs)
    {
        nvinfer1::IExprBuilder& builder = rhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kPROD,  *builder.constant(lhs), *rhs.expr_), builder);
    }

    friend DimensionExpr operator/( const int& lhs, const DimensionExpr &rhs)
    {
        nvinfer1::IExprBuilder& builder = rhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV,  *builder.constant(lhs), *rhs.expr_), builder);
    }

    friend DimensionExpr ceil_div( const int& lhs, const DimensionExpr &rhs)
    {
        nvinfer1::IExprBuilder& builder = rhs.builder_;
        return DimensionExpr(builder.operation(nvinfer1::DimensionOperation::kCEIL_DIV,  *builder.constant(lhs), *rhs.expr_), builder);
    }


private:
    const nvinfer1::IDimensionExpr * expr_ = nullptr;
    nvinfer1::IExprBuilder& builder_;
};

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TENSORRT_DIMENSION_EXPR_H_
