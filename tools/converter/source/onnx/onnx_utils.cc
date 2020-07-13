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

#include "onnx_utils.h"

onnx::ValueInfoProto MakeValueInfo(const std::string& name, const std::vector<int>& data_shape,
                                   const onnx::TensorProto::DataType& data_type) {
    onnx::ValueInfoProto value_info_proto;

    value_info_proto.set_name(name);
    onnx::TypeProto_Tensor* tensor_type = value_info_proto.mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(data_type);
    auto* shape = tensor_type->mutable_shape();
    for (int i = 0; i < data_shape.size(); ++i) {
        shape->add_dim()->set_dim_value(data_shape.at(i));
    }

    return value_info_proto;
}

template <class T>
onnx::TensorProto MakeTensor(const std::string& name, const std::vector<T>& v, onnx::TensorProto_DataType& data_type) {
    // TODO
}

onnx::AttributeProto MakeAttribute(const std::string& name, const std::vector<int>& vals) {
    onnx::AttributeProto attr;
    attr.set_name(name);
    for (const auto v : vals) {
        attr.add_ints(v);
    }
    attr.set_type(onnx::AttributeProto::INTS);
    return attr;
}
onnx::AttributeProto MakeAttribute(const std::string& name, const std::vector<float>& vals) {
    onnx::AttributeProto attr;
    attr.set_name(name);
    for (const auto v : vals) {
        attr.add_floats(v);
    }
    attr.set_type(onnx::AttributeProto::FLOATS);
    return attr;
}
onnx::AttributeProto MakeAttribute(const std::string& name, int64_t val) {
    onnx::AttributeProto attr;
    attr.set_name(name);
    attr.set_i(val);
    attr.set_type(onnx::AttributeProto::INT);
    return attr;
}
onnx::AttributeProto MakeAttribute(const std::string& name, const std::string& val) {
    onnx::AttributeProto attr;
    attr.set_name(name);
    attr.set_s(val);
    attr.set_type(onnx::AttributeProto::STRING);
    return attr;
}
onnx::AttributeProto MakeAttribute(const std::string& name, onnx::TensorProto& val) {
    onnx::AttributeProto attr;
    attr.set_name(name);
    attr.mutable_t()->CopyFrom(val);
    attr.set_type(onnx::AttributeProto::TENSOR);
    return attr;
}

onnx::NodeProto MakeNode(const std::string& type, const std::vector<std::string>& inputs,
                         const std::vector<std::string>& outputs, const std::vector<onnx::AttributeProto>& attributes,
                         const std::string& name) {
    onnx::NodeProto node;
    if (!name.empty()) {
        node.set_name(name);
    }
    node.set_op_type(type);
    for (const auto& input : inputs) {
        node.add_input(input);
    }
    for (const auto& output : outputs) {
        node.add_output(output);
    }
    for (const auto& attr : attributes) {
        node.add_attribute()->CopyFrom(attr);
    }
    return node;
}
