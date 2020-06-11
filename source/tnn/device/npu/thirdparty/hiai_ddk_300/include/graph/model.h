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
 * @file model.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_MODEL_H
#define GE_MODEL_H

#include <string>
#include <vector>
#include <map>
#include "graph.h"
#include "attr_value.h"
#include "detail/attributes_holder.h"

namespace ge {
using std::map;
using std::vector;
using std::string;

/*lint -e148*/

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Model : public AttrHolder{
public:
    Model();

    ~Model() = default;

    explicit Model(const string& name, const string& customVersion);

    string GetName() const;
    void SetName(const string& name);

    uint32_t GetVersion() const;

    void SetVersion(uint32_t version) { version_ = version; }

    std::string GetPlatformVersion() const;

    void SetPlatformVersion(string version) { platform_version_ = version; }

    Graph GetGraph() const;

    void SetGraph(const Graph& graph);

    using AttrHolder::SetAttr;
    using AttrHolder::GetAttr;
    using AttrHolder::HasAttr;
    using AttrHolder::GetAllAttrs;
    using AttrHolder::GetAllAttrNames;

    graphStatus Save(Buffer& buffer) const;

    // model will be rewrite
    static graphStatus Load(const uint8_t* data, size_t len, Model& model);

    bool IsValid() const;


private:
    void Init();
private:
    ProtoAttrMapHelper MutableAttrMap() override;
    ConstProtoAttrMapHelper GetAttrMap() const override;

    ProtoAttrMapHelper attrs_;

    friend  class ModelSerializeImp;
    friend  class GraphDebugImp;
    string name_;
    uint32_t version_;
    std::string platform_version_{""};
    Graph graph_;
};

/*lint +e148*/

}


#endif //GE_MODEL_H
