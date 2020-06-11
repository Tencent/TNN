/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file tensor.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_TENSOR_H
#define GE_TENSOR_H

#include <memory>
#include <vector>
#include <atomic>
#include "types.h"
#include "debug/ge_error_codes.h"
#include "./detail/attributes_holder.h"
#include "buffer.h"

namespace ge {

class TensorDescImpl;

using TensorDescImplPtr = std::shared_ptr<TensorDescImpl>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Shape {
public:
    Shape();
    ~Shape() = default;
    explicit Shape(std::vector<int64_t> s);

    size_t GetDimNum() const;
    //if the idx is invalid, return 0
    int64_t GetDim(size_t idx) const;
    graphStatus SetDim(size_t idx, int64_t value);
    std::vector<int64_t> GetDims() const;

    int64_t GetShapeSize() const;

    Shape(const Shape& other);
    Shape(Shape&& other);
    Shape& operator=(const Shape& other);
    Shape& operator=(Shape&& other);
private:
    GeIrProtoHelper<proto::ShapeDef> shapeDef_;
    friend class TensorDesc;
    // create from proto obj
    Shape(const ProtoMsgOwner& protoOnwer, proto::ShapeDef* protoMsg);

    void RefTo(Shape& shape){
        shapeDef_ = shape.shapeDef_;
    }
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TensorDesc : public  AttrHolder{
    friend class TensorUtils;
    friend  class AttrValue;
    friend class ModelSerialize;
public:

    TensorDesc();
    explicit TensorDesc(Shape shape, Format format = FORMAT_NCHW, DataType dt = DT_FLOAT);
    TensorDesc(const TensorDesc& desc);
    TensorDesc(TensorDesc&& desc);

    ~TensorDesc() = default;

    void Update(Shape shape, Format format = FORMAT_NCHW, DataType dt = DT_FLOAT);

    Shape GetShape() const;
    Shape& MutableShape();
    void SetShape(Shape shape);

    Format GetFormat() const;
    void SetFormat(Format format);

    DataType GetDataType() const;
    void SetDataType(DataType dt);

    TensorDesc Clone() const;
    TensorDesc& operator= (const TensorDesc& desc);
    TensorDesc& operator= (TensorDesc&& desc);

    graphStatus IsValid();


private:
    using AttrHolder::GetAllAttrs;
    using AttrHolder::GetAttr;
    using AttrHolder::SetAttr;
    using AttrHolder::DelAttr;
    using AttrHolder::HasAttr;
private:
    void Init();

    ProtoAttrMapHelper MutableAttrMap() override;
    ConstProtoAttrMapHelper GetAttrMap() const override;

    // create from proto obj
    TensorDesc(const ProtoMsgOwner& protoOnwer, proto::TensorDescriptor* protoMsg);
    friend class Tensor;
    friend class AttrValueImp;
    friend class ModelSerializeImp;

    GeIrProtoHelper<proto::TensorDescriptor> tensorDescriptor_;
    mutable Shape __shape_; // reference from tensorDescriptor_, do not direct use

    void RefTo(TensorDesc& tensorDesc){
        tensorDescriptor_ = tensorDesc.tensorDescriptor_;
    }
    Shape& ShapeReference() const;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Tensor {
public:
    Tensor();
    explicit Tensor(const TensorDesc &tensorDesc);
    explicit Tensor(const TensorDesc &tensorDesc, const std::vector<uint8_t> &data);
    explicit Tensor(const TensorDesc &tensorDesc, const Buffer &data);
    explicit Tensor(const TensorDesc &tensorDesc, const uint8_t *data, size_t size);
    explicit Tensor(TensorDesc &&tensorDesc, std::vector<uint8_t> &&data);
    ~Tensor() = default;

    TensorDesc GetTensorDesc() const;
    TensorDesc& MutableTensorDesc();
    graphStatus SetTensorDesc(const TensorDesc &tensorDesc);

    const Buffer GetData() const;
    Buffer MutableData();
    graphStatus SetData(std::vector<uint8_t> &&data);
    graphStatus SetData(const std::vector<uint8_t> &data);
    graphStatus SetData(const Buffer &data);
    graphStatus SetData(const uint8_t *data, size_t size);

    Tensor Clone() const;

    Tensor(const Tensor &other); // share value
    Tensor& operator= (const Tensor &other); // share value
private:
    friend class AttrValueImp;

    // create from proto obj
    Tensor(const ProtoMsgOwner& protoOnwer, proto::TensorDef* protoMsg);
    GeIrProtoHelper<proto::TensorDef> tensorDef_;
    mutable TensorDesc __desc_; // reference from tensorDef_, do not direct use
    TensorDesc& DescReference() const;
};
}


#endif //GE_TENSOR_H
