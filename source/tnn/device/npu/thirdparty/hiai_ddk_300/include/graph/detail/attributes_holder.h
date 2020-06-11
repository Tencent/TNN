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
 * @file attributes_holder.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_ATTRIBUTES_HOLDER_H
#define GE_ATTRIBUTES_HOLDER_H

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_set>
#include "../debug/ge_error_codes.h"

namespace google{
namespace protobuf{
class Message;
template <typename Key, typename T>
class Map;
}
}

namespace ge {

using std::string;
class AttrValue;

namespace proto{
class AttrDef;
class TensorDef;
class TensorDescriptor;
class ShapeDef;
class NamedAttrs;
class ModelDef;
class OpDef;
class GraphDef;
}

using ProtoAttrMap = ::google::protobuf::Map<::std::string, ::ge::proto::AttrDef>; //lint !e1073
using ProtoMsgOwner = std::shared_ptr<::google::protobuf::Message>;


template <class ProtoType>
class GeIrProtoHelper{
public:
    GeIrProtoHelper(const ProtoMsgOwner& protoOwner, ProtoType* protoMsg)
            :protoOwner_(protoOwner), protoMsg_(protoMsg){}

    GeIrProtoHelper(){
        protoOwner_ = std::shared_ptr<::google::protobuf::Message>(nullptr);
        protoMsg_ = nullptr;
    }
    virtual ~GeIrProtoHelper() = default;

    template <typename T>
    GeIrProtoHelper(const GeIrProtoHelper<T>& other){
        protoOwner_ = other.protoOwner_;
        protoMsg_ = other.protoMsg_;
    }
    template <typename T>
    GeIrProtoHelper& operator=(const GeIrProtoHelper<T>& other){
        protoOwner_ = other.protoOnwer_;
        protoMsg_ = other.protoMsg_;
        return *this;
    }
    void InitDefault();
    template <typename T>
    bool operator==(const GeIrProtoHelper<T>& other) const{
        return protoOwner_ == other.protoOwner_ && protoMsg_ == other.protoMsg_;
    }

    inline const ProtoMsgOwner& GetProtoOwner() const{
        return protoOwner_;
    }
    inline ProtoType* GetProtoMsg() const{
        return protoMsg_;
    }
    void CopyValueFrom(const GeIrProtoHelper<ProtoType>& other){
        if(other.protoMsg_ != nullptr && protoMsg_ != nullptr){
            *protoMsg_ = *other.protoMsg_;
        }
    }
    void MoveValueFrom(GeIrProtoHelper<ProtoType>&& other){
        if(other.protoMsg_ != nullptr && protoMsg_ != nullptr){
            *protoMsg_ = std::move(*other.protoMsg_);
        }
    }
public:
    ProtoMsgOwner protoOwner_ = nullptr;
    ProtoType* protoMsg_ = nullptr;
    friend class GeIrProtoHelper<typename std::conditional<std::is_const<ProtoType>::value,
            typename std::remove_const<ProtoType>::type, const ProtoType>::type>;
};

using ProtoAttrMapHelper = GeIrProtoHelper<ProtoAttrMap>;
using ConstProtoAttrMapHelper = GeIrProtoHelper<const ProtoAttrMap>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY  AttrHolder {
public:
    AttrHolder() = default;
    virtual ~AttrHolder() = default;

    graphStatus SetAttr(const string& name, const AttrValue& value);

    graphStatus GetAttr(const string& name, AttrValue& value) const;

    bool HasAttr(const string& name) const;

    graphStatus DelAttr(const string& name);

protected:
    graphStatus AddRequiredAttr(const std::string& name);
    const std::unordered_set<string> GetAllAttrNames() const;
    const std::map<string, AttrValue> GetAllAttrs() const; //lint !e1073
protected:
    virtual ProtoAttrMapHelper MutableAttrMap() = 0;
    virtual ConstProtoAttrMapHelper GetAttrMap() const = 0;

    friend class ModelSerializeImp;
    friend class AttrUtils;
    friend class AttrUtilsHelper;

    std::vector<string> requiredAttrs_;
};


}


#endif //GE_ATTRIBUTES_HOLDER_H
