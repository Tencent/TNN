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
* @file attr_value.h
*
* @brief
*
* @version 1.0
*
*/
#ifndef GE_ATTR_VALUE_H
#define GE_ATTR_VALUE_H

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include "tensor.h"
#include "debug/ge_error_codes.h"
#include "buffer.h"
#include "detail/attributes_holder.h"

using std::vector;
using std::string;
using std::map;

namespace ge {

class Tensor;

using TensorPtr = std::shared_ptr<Tensor>;
using ConstTensorPtr = std::shared_ptr<const Tensor>;

class ComputeGraph;
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;
using ConstComputeGraphPtr = std::shared_ptr<const ComputeGraph>;

class TensorDesc;

class AttrValueImp;

/*lint -e148*/
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AttrValue {
public:

    class NamedAttrs : public AttrHolder{
    public:
        NamedAttrs();
        virtual ~NamedAttrs() = default;
        void SetName(const std::string& name);
        string GetName() const;
        AttrValue GetItem(const string& key) const;
    private:
        // create namedAttrs from protobuf obj
        NamedAttrs(const ProtoMsgOwner& owner, proto::NamedAttrs* protoMsg);
        GeIrProtoHelper<proto::NamedAttrs> namedAttrs_;
        ProtoAttrMapHelper MutableAttrMap() override;
        ConstProtoAttrMapHelper GetAttrMap() const override;
        friend class AttrValueImp;
    };

public:
    using INT = int64_t;
    using FLOAT = float;
    using BOOL = bool;
    using STR = std::string;
    using TENSOR = TensorPtr;
    using TENSOR_DESC = TensorDesc;
    using GRAPH = ComputeGraphPtr;
    using BYTES = Buffer;
    using NAMED_ATTRS = NamedAttrs;

    using LIST_INT = vector<INT>;
    using LIST_FLOAT = vector<FLOAT>;
    using LIST_BOOL = vector<BOOL>;
    using LIST_STR = vector<STR>;
    using LIST_TENSOR = vector<TENSOR>;
    using LIST_TENSOR_DESC = vector<TENSOR_DESC>;
    using LIST_GRAPH = vector<GRAPH>;
    using LIST_BYTES = vector<BYTES>;
    using LIST_NAMED_ATTRS = vector<NAMED_ATTRS>;

    enum ValueType {
        VT_NONE = 0,
        VT_STRING,
        VT_FLOAT,
        VT_BOOL,
        VT_INT,
        VT_TENSOR_DESC,
        VT_TENSOR,
        VT_BYTES,
        VT_GRAPH,
        VT_NAMED_ATTRS,

        VT_LIST_BASE = 1000,
        VT_LIST_STRING = VT_LIST_BASE + VT_STRING,
        VT_LIST_FLOAT = VT_LIST_BASE + VT_FLOAT,
        VT_LIST_BOOL = VT_LIST_BASE + VT_BOOL,
        VT_LIST_INT = VT_LIST_BASE + VT_INT,
        VT_LIST_TENSOR_DESC = VT_LIST_BASE + VT_TENSOR_DESC,
        VT_LIST_TENSOR = VT_LIST_BASE + VT_TENSOR,
        VT_LIST_BYTES = VT_LIST_BASE + VT_BYTES,
        VT_LIST_GRAPH = VT_LIST_BASE + VT_GRAPH,
        VT_LIST_NAMED_ATTRS = VT_LIST_BASE + VT_NAMED_ATTRS,
    };
public:
    template<class T>
    struct IsAttrTypeEnable {
        using DT = typename std::remove_cv<T>::type;

        static bool const VALUE =
                std::is_same<INT, DT>::value || std::is_same<FLOAT, DT>::value || std::is_same<BOOL, DT>::value ||
                std::is_same<STR, DT>::value || std::is_same<GRAPH, DT>::value || std::is_same<TENSOR, DT>::value ||
                std::is_same<TENSOR_DESC, DT>::value || std::is_same<BYTES, DT>::value ||
                std::is_same<NAMED_ATTRS, DT>::value;

        // not has list type of NamedAttrs
        static bool const LIST_VALUE = std::is_same<LIST_INT, DT>::value || std::is_same<LIST_FLOAT, DT>::value ||
               std::is_same<LIST_BOOL, DT>::value || std::is_same<LIST_STR, DT>::value ||
               std::is_same<LIST_GRAPH, DT>::value || std::is_same<LIST_TENSOR, DT>::value ||
               std::is_same<LIST_TENSOR_DESC, DT>::value ||
               std::is_same<LIST_BYTES, DT>::value || std::is_same<LIST_NAMED_ATTRS, DT>::value;

    };

    template<typename vector_type> using enable_if_vector_type_valid_t =
            typename std::enable_if<IsAttrTypeEnable<vector_type>::LIST_VALUE, int>::type;//to  cols

    template<typename one_type> using enable_if_one_type_valid_t =
            typename std::enable_if<IsAttrTypeEnable<one_type>::VALUE, int>::type;

    template<typename val_type> using enable_if_type_valid_t = typename std::enable_if<
            IsAttrTypeEnable<val_type>::VALUE || IsAttrTypeEnable<val_type>::LIST_VALUE, int>::type;

    template<typename seriliable_type> using enable_if_seriliable_type_valid_t = typename seriliable_type::__ge_serializable;
public:
    AttrValue();
    ~AttrValue() = default;
    // SetValue, Set initializer_list
    template<typename T, typename DT, enable_if_vector_type_valid_t<T> = 0>
    graphStatus SetValue(std::initializer_list<DT>&& val)
    {
        T vectorVal;
        for (auto& item : val) {
            vectorVal.push_back(item);
        }
        return SetValue(vectorVal);
    }

    // SetValue, Set vector
    template<typename T, typename DT, enable_if_vector_type_valid_t<T> = 0>
    graphStatus SetValue(const std::vector<DT>& val)
    {
        T vectorVal;
        for (auto item : val) {
            vectorVal.push_back(item);
        }
        return SetValue(vectorVal);
    }

    // SetValue, not list type
    template<typename T, typename DT, enable_if_one_type_valid_t<T> = 0>
    graphStatus SetValue(DT&& val)
    {
        return SetValue(T(std::forward<DT>(val)));
    }

    // GE_SERIALIZABLE
    template<typename T, enable_if_seriliable_type_valid_t<T> = 0>
    graphStatus SetValue(const T& t)
    {
        return t.Save(*this);
    }

    template<typename T, enable_if_seriliable_type_valid_t<T> = 0>
    graphStatus SetValue(const vector<T>& t)
    {
        vector<NamedAttrs> attrs;
        for (auto& item : t) {
            AttrValue val;
            item.Save(val);
            NamedAttrs attrsItem;
            val.GetValue<NamedAttrs>(attrsItem);
            attrs.push_back(attrsItem);
        }
        return SetValue(attrs);
    }

    // GetValue, list value
    template<typename T, typename DT, enable_if_vector_type_valid_t<T> = 0,
            typename std::enable_if<!std::is_same<DT, TensorPtr>::value, int>::type = 0>
    graphStatus GetValue(std::vector<DT>& val) const
    {
        T valGet;
        val.clear();
        auto status = GetValue(valGet);
        if (status != GRAPH_SUCCESS) {
            return status;
        }
        for (auto item : valGet) {
            val.push_back(item);
        }
        return GRAPH_SUCCESS;
    }

    // GetValue, not list type
    template<typename T, typename DT, enable_if_one_type_valid_t<T> = 0,
            typename std::enable_if<!std::is_same<DT, TensorPtr>::value, int>::type = 0>
    graphStatus GetValue(DT& val) const
    {
        T valGet;
        auto status = GetValue(valGet);
        if (status != GRAPH_SUCCESS) {
            return status;
        }
        val = DT(valGet);
        return GRAPH_SUCCESS;
    }

    // GE_SERIALIZABLE
    template<typename T, enable_if_seriliable_type_valid_t<T> = 0>
    graphStatus GetValue(T& t)
    {
        return t.Load(*this);
    }

    template<typename T, enable_if_seriliable_type_valid_t<T> = 0>
    graphStatus GetValue(vector<T>& t)
    {
        graphStatus status;
        t.clear();
        vector<NamedAttrs> attrs;
        status = this->GetValue(attrs);
        if (status != GRAPH_SUCCESS) {
            return status;
        }
        for (auto& attr : attrs) {
            T item;
            AttrValue val;
            val.SetValue(attr);
            status = item.Load(val);
            if (status != GRAPH_SUCCESS) {
                return status;
            }
            t.push_back(item);
        }
        return GRAPH_SUCCESS;
    }

    template<typename T, typename DT, enable_if_type_valid_t<T> = 0>
    static AttrValue CreateFrom(DT&& val)
    {
        AttrValue valRet;
        valRet.SetValue<T>(std::forward<DT>(val));
        return valRet;
    }

    template<typename T, typename DT, enable_if_vector_type_valid_t<T> = 0>
    static AttrValue CreateFrom(std::initializer_list<DT>&& val)
    {
        AttrValue valRet;
        valRet.SetValue<T>(std::move(val));
        return valRet;
    }

    template<typename T, enable_if_seriliable_type_valid_t<T> = 0>
    static AttrValue CreateFrom(const T& val)
    {
        AttrValue valRet;
        valRet.SetValue(val);
        return valRet;
    }

    template<typename T, enable_if_seriliable_type_valid_t<T> = 0>
    static AttrValue CreateFrom(const vector<T>& val)
    {
        AttrValue valRet;
        valRet.SetValue(val);
        return valRet;
    }

    ValueType GetValueType() const;

    bool IsEmpty() const;

    AttrValue Copy() const;

    // for map key
    bool operator ==(const AttrValue& other) const{return value_ == other.value_;}

    graphStatus MutableTensor(TensorPtr& tensor);
    graphStatus MutableListTensor(vector<TensorPtr>& list_tensor);

private:

#define VALUE_SET_GET_DEC(DT) \
    graphStatus SetValue(const DT& val); \
    graphStatus GetValue(DT& val) const;
    VALUE_SET_GET_DEC(AttrValue::STR)
    VALUE_SET_GET_DEC(AttrValue::INT)
    VALUE_SET_GET_DEC(AttrValue::FLOAT)
    VALUE_SET_GET_DEC(AttrValue::BOOL)
    VALUE_SET_GET_DEC(TensorDesc)
    VALUE_SET_GET_DEC(AttrValue::TENSOR)
    VALUE_SET_GET_DEC(AttrValue::GRAPH)
    VALUE_SET_GET_DEC(BYTES)
    VALUE_SET_GET_DEC(NamedAttrs)
    VALUE_SET_GET_DEC(vector<AttrValue::STR>)
    VALUE_SET_GET_DEC(vector<AttrValue::INT>)
    VALUE_SET_GET_DEC(vector<AttrValue::FLOAT>)
    VALUE_SET_GET_DEC(vector<AttrValue::BOOL>)
    VALUE_SET_GET_DEC(vector<TensorDesc>)
    VALUE_SET_GET_DEC(vector<AttrValue::TENSOR>)
    VALUE_SET_GET_DEC(vector<AttrValue::GRAPH>)
    VALUE_SET_GET_DEC(vector<AttrValue::BYTES>)
    VALUE_SET_GET_DEC(vector<NamedAttrs>)
#undef VALUE_SET_GET_DEC

    GeIrProtoHelper<proto::AttrDef> value_;
    AttrValue(const ProtoMsgOwner& protoOnwer, ge::proto::AttrDef* val);

    friend class AttrHolder;
    friend class ModelSerializeImp;
};
/*lint +e148*/

} // namespace ge

#endif //GE_ATTR_VALUE_H
