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
 * @file graph.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_GRAPH_H
#define GE_GRAPH_H

#include <memory>
#include "operator.h"

namespace ge {
class GraphImpl;

using GraphImplPtr = std::shared_ptr<GraphImpl>;

class GE_FUNC_HOST_VISIBILITY Graph {
    friend class GraphUtils;

public:
    explicit Graph(const std::string& name);

    Graph() = default;

    ~Graph() = default;

    Graph& SetInputs(std::vector<Operator>& inputs);

    Graph& SetOutputs(std::vector<Operator>& outputs);

    bool IsValid() const;

    graphStatus AddOp(ge::Operator& op);

    ge::Operator FindOpByName(const string& name) const;

    graphStatus CheckOpByName(const string& name) const;

    graphStatus GetAllOpName(std::vector<string>& opName) const ;
private:
    GraphImplPtr impl_{nullptr};
};
}

#endif //GE_MODEL_H
