/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file op_reg.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_OP_REG_H
#define GE_OP_REG_H

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include "types.h"
#include "operator.h"
#include "tensor.h"
#include "attr_value.h"

namespace ge {

using std::string;
using std::vector;
using std::function;
using TensorPtr=std::shared_ptr<Tensor>;

class OpDesc;

class OpReg {
public:
    OpReg& N()
    {
        return *this;
    }

    OpReg& ATTR()
    {
        return *this;
    }

    OpReg& REQUIRED_ATTR()
    {
        return *this;
    }

    OpReg& INPUT()
    {
        return *this;
    }

    OpReg& OPTIONAL_INPUT()
    {
        return *this;
    }


    OpReg& OUTPUT()
    {
        return *this;
    }

    OpReg& INFER_SHAPE_AND_TYPE()
    {
        return *this;
    }
};

template <typename T>
struct GetType{
    using type = T;
    void assign(type &left, const T& right){left = right;}
};

template <>
struct GetType<TensorPtr>{
using type = ConstTensorPtr;
    void assign(type &left, const TensorPtr& right){left = right;}
};

template <>
struct GetType<ComputeGraphPtr>{
    using type = ConstComputeGraphPtr;
    void assign(type &left, const ComputeGraphPtr& right){left = right;}
};

template <>
struct GetType<vector<TensorPtr>>{
using type = vector<ConstTensorPtr>;
    void assign(type &left, const vector< TensorPtr>& right){
        for(auto& item:right){
            left.push_back(item);
        }
     }
};

template <>
struct GetType<vector<ComputeGraphPtr>>{
    using type = vector<ConstComputeGraphPtr>;
    void assign(type &left, const vector<ComputeGraphPtr>& right){
        for(auto& item:right){
            left.push_back(item);
        }
    }
};

template <typename T>
using GetType_t = typename GetType<T>::type;

#define REG_OP(x) namespace op{                                                  \
    class x : public Operator {                                                     \
        typedef x _THIS_TYPE;                                                       \
    public:                                                                         \
        explicit x(const string &name):Operator(name, #x) {                         \
            __##x();                                                                \
        }                                                                           \
        explicit x(): Operator(#x) {                                                \
            __##x();                                                                \
        }                                                                           \
    private:                                                                        \
        void __##x() { OpReg()

#define ATTR(x, default_value) N();__attr_##x();}                                \
    public:                                                                         \
        static const string name_attr_##x() {                                       \
            return #x;                                                              \
        }                                                                           \
        GetType_t<decltype(default_value)> get_attr_##x() const {                   \
            GetType_t<decltype(default_value)> ret;                                 \
            GetType<decltype(default_value)>().assign(ret, default_value);          \
            AttrValue attr;                                                         \
            if(Operator::GetAttr(#x, attr) == GRAPH_FAILED) {                             \
                return ret;                                                         \
            }                                                                       \
            attr.GetValue<decltype(default_value)>(ret);                            \
            return ret;                                                             \
        }                                                                           \
        _THIS_TYPE& set_attr_##x(decltype(default_value) v) {                       \
            auto attr = AttrValue::CreateFrom<decltype(default_value)>(v);          \
            Operator::SetAttr(#x, std::move(attr));                                 \
            return *this;                                                           \
        }                                                                           \
        _THIS_TYPE& set_attr_##x(function<decltype(default_value)()> v) {           \
            return *this;                                                           \
        }                                                                           \
    private:                                                                        \
        void __attr_##x(){                                                          \
            auto defaultAttr = AttrValue::CreateFrom<decltype(default_value)>(default_value);\
            Operator::AttrRegister(#x, std::move(defaultAttr));               \
            string attr_name(#x); \
            OpReg()

#define REQUIRED_ATTR(x, type)  N();__required_attr_##x();}                                \
    public:                                                                         \
        static const string name_attr_##x() {                                       \
            return #x;                                                              \
        }                                                                           \
        GetType_t<type> get_attr_##x() const {                   \
            GetType_t<type> ret;                                 \
            AttrValue attr;                                                         \
            if(Operator::GetAttr(#x, attr) == GRAPH_FAILED) {                             \
                return ret;                                                         \
            }                                                                       \
            attr.GetValue<type>(ret);                            \
            return ret;                                                             \
        }                                                                           \
        _THIS_TYPE& set_attr_##x(type v) {                       \
            auto attr = AttrValue::CreateFrom<type>(v);          \
            Operator::SetAttr(#x, std::move(attr));                                 \
            return *this;                                                           \
        }                                                                           \
        _THIS_TYPE& set_attr_##x(function<type()> v) {           \
            return *this;                                                           \
        }                                                                           \
    private:                                                                        \
        void __required_attr_##x(){                                                          \
            Operator::RequiredAttrRegister(#x);               \
            string attr_name(#x); \
            OpReg()


#define INPUT(x, t) N();__input_##x();}                                          \
    public:                                                                         \
        static const string name_in_##x() {                                         \
            return #x;                                                              \
        }                                                                           \
        _THIS_TYPE& set_input_##x(Operator &v, const string &srcName) {             \
            Operator::SetInput(#x, v, srcName);                                     \
            return *this;                                                           \
        }                                                                           \
        _THIS_TYPE& set_input_##x(Operator &v) {                                    \
            Operator::SetInput(#x, v);                                               \
            return *this;                                                           \
        }                                                                           \
        TensorDesc get_input_desc_##x() {                                           \
            return Operator::GetInputDesc(#x);                                      \
        }                                                                           \
        graphStatus update_input_desc_##x(const TensorDesc& tensorDesc) {           \
            return Operator::UpdateInputDesc(#x, tensorDesc);                       \
        }                                                                           \
    private:                                                                        \
        void __input_##x(){ Operator::InputRegister(#x);OpReg()

#define OPTIONAL_INPUT(x, t) N();__optional_input_##x();}                        \
    public:                                                                         \
        static const string name_in_##x() {                                         \
            return #x;                                                              \
        }                                                                           \
        _THIS_TYPE& set_input_##x(Operator &v) {                                    \
            Operator::OptionalInputRegister(#x);                                    \
            Operator::SetInput(#x, v);                                              \
            return *this;                                                           \
        }                                                                           \
        _THIS_TYPE& set_input_##x(Operator &v, const string &srcName) {             \
            Operator::OptionalInputRegister(#x);                                    \
            Operator::SetInput(#x, v, srcName);                                     \
            return *this;                                                           \
        }                                                                           \
        TensorDesc get_input_desc_##x() {                                           \
            return Operator::GetInputDesc(#x);                                      \
        }                                                                           \
        graphStatus update_input_desc_##x(const TensorDesc& tensorDesc) {                \
            return Operator::UpdateInputDesc(#x, tensorDesc);                       \
        }                                                                           \
    private:                                                                        \
        void __optional_input_##x(){ OpReg()

#define INFER_SHAPE_AND_TYPE(x) N();__infer_desc_();}                            \
    private:                                                                        \
         void __infer_desc_(){InferFuncRegister([&](Operator &v){return x((_THIS_TYPE&)v);}); OpReg()


#define ATTR_ALL_VERIFY(x) N();__verify_attr_();}                                \
    private:                                                                        \
         void __verify_attr_(){VerifierFuncRegister([&](Operator &v){return x((_THIS_TYPE&)v);}); OpReg()

#define OUTPUT(x, t) N();__out_##x();}                                           \
    public:                                                                         \
        static const string name_out_##x() {                                        \
            return #x;                                                              \
        }                                                                           \
        TensorDesc get_output_desc_##x() {                                          \
            return Operator::GetOutputDesc(#x);                                     \
        }                                                                           \
        graphStatus update_output_desc_##x(const TensorDesc& tensorDesc) {          \
            return Operator::UpdateOutputDesc(#x, tensorDesc);                      \
        }                                                                           \
    private:                                                                        \
        void __out_##x(){Operator::OutputRegister(#x);OpReg()

#define DYNAMIC_INPUT(x, t) N();__dy_input_##x();}                               \
    public:                                                                         \
        _THIS_TYPE& create_dynamic_input_##x(unsigned int num) {                    \
            Operator::DynamicInputRegister(#x, num);                                \
            return *this;                                                           \
        }                                                                           \
        TensorDesc get_dynamic_input_desc_##x(unsigned int index) {                 \
            return Operator::GetDynamicInputDesc(#x, index);                        \
        }                                                                           \
        graphStatus update_dynamic_input_desc_##x(unsigned int index, const TensorDesc& tensorDesc) { \
            return Operator::UpdateDynamicInputDesc(#x, index, tensorDesc);         \
        }                                                                           \
      _THIS_TYPE& set_dynamic_input_##x(unsigned int dstIndex, Operator &v) {         \
            Operator::SetInput(#x, dstIndex, v);                                              \
            return *this;                                                           \
        }                                                                           \
        _THIS_TYPE& set_dynamic_input_##x(unsigned int dstIndex, Operator &v, const string &srcName){ \
            Operator::SetInput(#x, dstIndex, v, srcName);                           \
            return *this;                                                           \
        }                                                                           \
    private:                                                                        \
        void __dy_input_##x(){ OpReg()

#define DYNAMIC_OUTPUT(x, t) N();__dy_output_##x();}                             \
    public:                                                                         \
        _THIS_TYPE& create_dynamic_output_##x(unsigned int num) {                   \
            Operator::DynamicOutputRegister(#x, num);                               \
            return *this;                                                           \
        }                                                                           \
        TensorDesc get_dynamic_output_desc_##x(unsigned int index) {                \
            return Operator::GetDynamicOutputDesc(#x, index);                       \
        }                                                                           \
        graphStatus update_dynamic_output_desc_##x(unsigned int index, const TensorDesc& tensorDesc) { \
            return Operator::UpdateDynamicOutputDesc(#x, index, tensorDesc);        \
        }                                                                           \
    private:                                                                        \
        void __dy_output_##x(){ OpReg()

#define OP_END() N();}};}

/* specialized shape inferencer macro */

#define DECLARE_INFERFUNC(op_name, func_name) \
    namespace op { \
    class op_name; \
    } \
    static graphStatus func_name(op::op_name& op);

#define IMPLEMT_INFERFUNC(op_name, func_name) \
    static graphStatus func_name(op::op_name& op)

/* specialized verifier macro */

#define DECLARE_VERIFIER(op_name, func_name) \
    namespace op { \
    class op_name; \
    } \
    static graphStatus func_name(op::op_name op);

#define IMPLEMT_VERIFIER(op_name, func_name) \
    static graphStatus func_name(op::op_name op)

/* utilty macros */

#define GET_INPUT_SHAPE(op, name) \
    op.GetInputDesc(name).GetShape().GetDims()

#define GET_DYNAMIC_INPUT_SHAPE(op, name, index) \
    op.GetDynamicInputDesc(name, index).GetShape().GetDims()

#define SET_OUTPUT_SHAPE(op, name, shape) \
    { \
        TensorDesc td = op.GetOutputDesc(name); \
        td.SetShape(Shape(shape)); \
        op.UpdateOutputDesc(name, td); \
    }

#define SET_DYNAMIC_OUTPUT_SHAPE(op, name, index, shape) \
    { \
        TensorDesc td = op.GetDynamicOutputDesc(name, index); \
        td.SetShape(Shape(shape)); \
        op.UpdateDynamicOutputDesc(name, index, td); \
    }

#define GET_ATTR(op, name, type, val) \
    { \
        AttrValue attr; \
        op.GetAttr(name, attr); \
        attr.GetValue<type>(val); \
    }
}
#endif //GE_OP_REG_H
