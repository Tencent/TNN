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
 * @file operator.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_OPERATOR_H
#define GE_OPERATOR_H

#include <map>
#include <memory>
#include <vector>
#include <functional>
#include "tensor.h"
#include "debug/ge_error_codes.h"

namespace ge {

class OperatorImpl;

using OperatorImplPtr = std::shared_ptr<OperatorImpl>;

class OpIO;

using OutHandler = std::shared_ptr<OpIO>;
using InHandler = std::shared_ptr<OpIO>;

class AttrValue;

using std::string;
using std::shared_ptr;
using std::function;

class ComputeGraph;
class OpDesc;

using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;
using OpDescPtr = std::shared_ptr<OpDesc>;

/*lint -e148*/
class GE_FUNC_HOST_VISIBILITY Operator {
public:
    friend class OperatorImpl;

    friend class GraphBuilderImpl;

public:
    Operator() {};

    explicit Operator(const string& type);

    explicit Operator(const string& name, const string& type); //lint !e148

    virtual ~Operator() = default;

    string GetName() const;

    Operator& SetInput(const string& dstName, const Operator& srcOprt); //only has one output index = 0

    Operator& SetInput(const string& dstName, const Operator& srcOprt, const string &name); //lint !e148

    TensorDesc GetInputDesc(const string& name) const;

    TensorDesc GetInputDesc(uint32_t index) const;

    bool TryGetInputDesc(const string& name, TensorDesc& tensorDesc) const;

    graphStatus UpdateInputDesc(const string& name, const TensorDesc& tensorDesc);

    TensorDesc GetOutputDesc(const string& name) const;

    TensorDesc GetOutputDesc(uint32_t index) const;

    graphStatus UpdateOutputDesc(const string& name, const TensorDesc& tensorDesc); //lint !e148

    TensorDesc GetDynamicInputDesc(const string& name, const unsigned int index) const;

    graphStatus UpdateDynamicInputDesc(const string& name, const unsigned int index,  const TensorDesc& tensorDesc); //lint !e148

    TensorDesc GetDynamicOutputDesc(const string& name, const unsigned int index) const;

    graphStatus UpdateDynamicOutputDesc(const string& name, const unsigned int index,  const TensorDesc& tensorDesc); //lint !e148

    Operator& SetAttr(const string& name, AttrValue&& attrValue);

    graphStatus GetAttr(const string& name, AttrValue& attrValue) const; //lint !e148

    graphStatus InferShapeAndType(); //lint !e148

    graphStatus VerifyAllAttr(bool disableCommonVerifier = false); //lint !e148

protected:
    explicit Operator(OperatorImplPtr&& opImpl);

    void InputRegister(const string& name);

    void OptionalInputRegister(const string& name);

    void InferFuncRegister(std::function<graphStatus (Operator&)> func);

    void VerifierFuncRegister(std::function<graphStatus (Operator&)> func);

    void OutputRegister(const string& name);

    void DynamicInputRegister(const string& name, const unsigned int num);

    void DynamicOutputRegister(const string& name, const unsigned int num);

    void AttrRegister(const string& name, AttrValue&& attrValue);

    void RequiredAttrRegister(const string& name);

//    void RegAttrVerfier(function<bool(const AttrValue&)> verfier){}

    graphStatus VerifyAll(); //lint !e148
    Operator& SetInput(const string& dstName, uint32_t dstIndex, const Operator& srcOprt); //only has one output index = 0
    Operator& SetInput(const string& dstName, uint32_t dstIndex, const Operator& srcOprt, const string &name); //lint !e148
private:
    Operator& SetInput(const string& dstName, OutHandler outHandler); //lint !e148

    OutHandler GetOutput(const string& name) const;

    OperatorImplPtr GetOperatorImplPtr() const;

    string GetOpType() const;

    std::vector<bool> GetOpIsInputConst() const;

    void SetOpIsInputConst(bool inputConst);

private:
    OperatorImplPtr operatorImpl_{nullptr};
};
/*lint +e148*/

}


#endif //GE_OPERATOR_H
