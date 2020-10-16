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

#include <algorithm>

namespace TNN_CONVERTER {

TNN_NS::DimsVector ConvertTensorShapeProtoToDimsVector(onnx::TensorShapeProto tensor_shape_proto) {
    TNN_NS::DimsVector dims_vector;
    int dim_size = tensor_shape_proto.dim_size();
    for (int i = 0; i < dim_size; ++i) {
        int dim = std::max(1, (int)tensor_shape_proto.dim(i).dim_value());
        dims_vector.push_back(dim);
    }
    return dims_vector;
}

onnx::AttributeProto_AttributeType GetAttributeType(const char *type_name) {
    if (type_name == typeid(int64_t).name()) {
        return onnx::AttributeProto_AttributeType_INT;
    } else if (type_name == typeid(int64_t[]).name()) {
        return onnx::AttributeProto_AttributeType_INTS;
    } else if (type_name == typeid(float).name()) {
        return onnx::AttributeProto_AttributeType_FLOAT;
    } else if (type_name == typeid(float[]).name()) {
        return onnx::AttributeProto_AttributeType_FLOATS;
    } else if (type_name == typeid(std::string).name()) {
        return onnx::AttributeProto_AttributeType_STRING;
    } else if (type_name == typeid(std::string[]).name()) {
        return onnx::AttributeProto_AttributeType_STRINGS;
    } else if (type_name == typeid(onnx::TensorProto).name()) {
        return onnx::AttributeProto_AttributeType_TENSOR;
    } else if (type_name == typeid(onnx::TensorProto[]).name()) {
        return onnx::AttributeProto_AttributeType_TENSORS;
    } else if (type_name == typeid(onnx::GraphProto).name()) {
        return onnx::AttributeProto_AttributeType_GRAPH;
    } else if (type_name == typeid(onnx::GraphProto[]).name()) {
        return onnx::AttributeProto_AttributeType_GRAPHS;
    } else {
        return onnx::AttributeProto_AttributeType_UNDEFINED;
    }
}
int GetAttributeInt(const onnx::NodeProto &node, const std::string &name, int default_value) {
    for (const auto &iter : node.attribute()) {
        if (iter.name() != name) {
            continue;
        }
        assert(iter.type() == onnx::AttributeProto_AttributeType_INT);
        return iter.i();
    }
    return default_value;
}

std::vector<int32_t> GetAttributeIntVector(const onnx::NodeProto &node, const std::string &name) {
    std::vector<int32_t> attributes;
    for (const auto &iter : node.attribute()) {
        if (iter.name() != name) {
            continue;
        }
        assert(iter.type() == onnx::AttributeProto_AttributeType_INTS);
        for (const auto &value : iter.ints()) {
            attributes.push_back(value);
        }
    }
    return attributes;
}

float GetAttributeFloat(const onnx::NodeProto &node, const std::string &name, float default_value) {
    for (const auto &iter : node.attribute()) {
        if (iter.name() != name) {
            continue;
        }
        assert(iter.type() == onnx::AttributeProto_AttributeType_FLOAT);
        return iter.f();
    }
    return default_value;
}

std::string GetAttributeString(const onnx::NodeProto &node, const std::string &name, std::string def) {
    for (const auto &iter : node.attribute()) {
        if (iter.name() == name) {
            assert(iter.type() == onnx::AttributeProto_AttributeType_STRING);
            return iter.s();
        }
    }
    return def;
}

std::vector<std::string> GetAttributeStringVector(const onnx::NodeProto &node, const std::string &name) {
    std::vector<std::string> attributes;
    for (const auto &iter : node.attribute()) {
        if (iter.name() != name) {
            continue;
        }
        assert(iter.type() == onnx::AttributeProto_AttributeType_STRINGS);
        for (const auto &value : iter.strings()) {
            attributes.push_back(value);
        }
    }
    return attributes;
}

std::vector<std::string> SplitString(std::string &s, const std::string &c) {
    std::vector<std::string> res;
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2) {
        res.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length()) {
        res.push_back(s.substr(pos1));
    }
    return res;
}

std::vector<uint8_t> GetAttributeUInt8Vector(const onnx::NodeProto &node, const std::string &name) {
    std::vector<uint8_t> attribute;
    for (const auto &iter : node.attribute()) {
        if (iter.name() == name) {
            assert(iter.type() == onnx::AttributeProto_AttributeType_STRING);
            const auto &raw_data = iter.s();
            int size             = raw_data.size();
            for (int i = 0; i < size; ++i) {
                attribute.push_back(*((uint8_t *)raw_data.data() + i));
            }
        }
    }
    return attribute;
}

std::vector<int8_t> Asymmetric2Symmetric(std::vector<uint8_t> &raw_value, uint8_t zero_point) {
    std::vector<int8_t> res;
    for (const auto &value : raw_value) {
        res.push_back(value - zero_point);
    }
    return res;
}

onnx::TensorProto GetAttributeTensor(const onnx::NodeProto &node, const char *key) {
    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto &attr = node.attribute(i);
        if (attr.name() == key) {
            return attr.t();
        }
    }

    return onnx::TensorProto();
}

const float *GetTensorProtoData(const onnx::TensorProto &tp) {
    if (tp.has_raw_data()) {
        return (const float *)tp.raw_data().data();
    } else if (tp.data_type() == 1) {
        return tp.float_data().data();
    } else if (tp.data_type() == 6) {
        return (const float *)tp.int32_data().data();
    } else if (tp.data_type() == 7) {
        return (const float *)tp.int64_data().data();
    } else if (tp.data_type() == 11) {
        return (const float *)tp.double_data().data();
    } else {
        assert(0);
        return nullptr;
    }
}

int GetTensorProtoDataSize(const onnx::TensorProto &tp) {
    if (tp.has_raw_data()) {
        const std::string &raw_data = tp.raw_data();
        if (tp.data_type() == 1) {
            return (int)raw_data.size() / 4;
        } else if (tp.data_type() == 2) {
            return (int)raw_data.size() / 1;
        } else if (tp.data_type() == 3) {
            return (int)raw_data.size() / 1;
        } else if (tp.data_type() == 4) {
            return (int)raw_data.size() / 2;
        } else if (tp.data_type() == 5) {
            return (int)raw_data.size() / 2;
        } else if (tp.data_type() == 6) {
            return (int)raw_data.size() / 4;
        } else if (tp.data_type() == 7) {
            return (int)raw_data.size() / 8;
        } else if (tp.data_type() == 11) {
            return (int)raw_data.size() / 8;
        } else {
            LOGD("unsupport data type: %d\n", tp.data_type());
            assert(0);
        }
    } else {
        if (tp.data_type() == 1) {
            return tp.float_data_size();
        } else if (tp.data_type() == 6) {
            return tp.int32_data_size();
        } else if (tp.data_type() == 7) {
            return tp.int64_data_size();
        } else if (tp.data_type() == 11) {
            return tp.double_data_size();
        } else {
            LOGD("unsupport data type: %d\n", tp.data_type());
            assert(0);
        }
    }
    return 0;
}

}  // namespace TNN_CONVERTER
