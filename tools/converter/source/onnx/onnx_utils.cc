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

#include "tnn/utils/dims_vector_utils.h"

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

std::vector<int64_t> GetAttributeInt64Vector(const onnx::NodeProto &node, const std::string &name) {
    std::vector<int64_t> attributes;
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

std::vector<int64_t> GetAttributeInt64Vector(const onnx::NodeProto &node, const std::string &name,
                                             std::map<std::string, const onnx::TensorProto *> &proxy_initializers_map,
                                             int location) {
    auto attributes = GetAttributeInt64Vector(node, name);
    if (attributes.empty()) {
        const int node_input_size = node.input_size();
        if (location < 0) {
            location += node_input_size;
        }
        assert(location >= 0);
        if (location > node_input_size - 1) {
            return attributes;
        }
        const auto &attributes_name = node.input(location);
        if (proxy_initializers_map.find(attributes_name) == proxy_initializers_map.end()) {
            return attributes;
        }
        const auto attributes_data = GetTensorProtoDataVector<int64_t>(*proxy_initializers_map[attributes_name]);
        for (const auto &item : attributes_data) {
            attributes.push_back(item);
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

float GetAttributeFloat(const onnx::NodeProto &node, const std::string &name, int location, float default_value,
                        std::map<std::string, const onnx::TensorProto *> proxy_initializers_map) {
    auto value = GetAttributeFloat(node, name, default_value);
    if (std::fabs(value - default_value) > 1e-6) {
        return value;
    }

    const auto node_input_size = node.input_size();
    assert(location >= 0 && location < node_input_size);

    const auto &target_name = node.input(location);
    assert(proxy_initializers_map.find(target_name) != proxy_initializers_map.end());
    const auto &tensor      = proxy_initializers_map[target_name];
    const int tensor_size   = GetTensorProtoDataSize(*tensor);
    const auto *tensor_data = GetTensorProtoData(*tensor);
    if (tensor_size > 0) {
        return tensor_data[0];
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
    int size = 0;
    if (tp.has_raw_data()) {
        const std::string &raw_data = tp.raw_data();
        switch (tp.data_type()) {
            case onnx::TensorProto_DataType_FLOAT: {
                size = int(raw_data.size() / sizeof(float));
                break;
            }
            case onnx::TensorProto_DataType_UINT8: {
                size = int(raw_data.size() / sizeof(uint8_t));
                break;
            }
            case onnx::TensorProto_DataType_INT8: {
                size = int(raw_data.size() / sizeof(int8_t));
                break;
            }
            case onnx::TensorProto_DataType_UINT16: {
                size = int(raw_data.size() / sizeof(uint16_t));
                break;
            }
            case onnx::TensorProto_DataType_INT16: {
                size = int(raw_data.size() / sizeof(int16_t));
                break;
            }
            case onnx::TensorProto_DataType_INT32: {
                size = int(raw_data.size() / sizeof(int32_t));
                break;
            }
            case onnx::TensorProto_DataType_INT64: {
                size = int(raw_data.size() / sizeof(int64_t));
                break;
            }
            case onnx::TensorProto_DataType_BOOL: {
                size = int(raw_data.size() / sizeof(bool));
                break;
            }
            case onnx::TensorProto_DataType_FLOAT16: {
                size = int(raw_data.size() / (sizeof(float) / 2));
                break;
            }
            case onnx::TensorProto_DataType_DOUBLE: {
                size = int(raw_data.size() / sizeof(double));
                break;
            }
            case onnx::TensorProto_DataType_UINT32: {
                size = int(raw_data.size() / sizeof(uint32_t));
                break;
            }
            case onnx::TensorProto_DataType_UINT64: {
                size = int(raw_data.size() / sizeof(uint64_t));
                break;
            }
            default: {
                LOGE("Onnx Converter: do not support tensor proto data type\n");
                size = -1;
            }
        }
    } else {
        switch (tp.data_type()) {
            case onnx::TensorProto_DataType_FLOAT: {
                size = tp.float_data_size();
                break;
            }
            case onnx::TensorProto_DataType_INT32: {
                size = tp.int32_data_size();
                break;
            }
            case onnx::TensorProto_DataType_INT64: {
                size = tp.int64_data_size();
                break;
            }
            case onnx::TensorProto_DataType_DOUBLE: {
                size = tp.double_data_size();
                break;
            }
            default: {
                LOGE("Onnx Converter: do not support tensor proto data type\n");
                size = -1;
            }
        }
    }
    return size;
}

void *GetDataFromTensor(const onnx::TensorProto &tensor, onnx::TensorProto_DataType data_type) {
    void *data_ptr = nullptr;
    if (tensor.data_type() == data_type) {
        if (tensor.has_raw_data()) {
            data_ptr = (void *)tensor.raw_data().data();
        } else if (data_type == onnx::TensorProto_DataType_FLOAT) {
            data_ptr = (void *)tensor.float_data().data();
        } else if (data_type == onnx::TensorProto_DataType_INT32) {
            data_ptr = (void *)tensor.int32_data().data();
        } else if (data_type == onnx::TensorProto_DataType_INT64) {
            data_ptr = (void *)tensor.int64_data().data();
        } else if (data_type == onnx::TensorProto_DataType_DOUBLE) {
            data_ptr = (void *)(tensor.double_data().data());
        } else {
            LOGE("Tensor(%s) do not have valid data\n", tensor.name().c_str());
        }
    } else {
        LOGE("Tensor(%s) data type does not match special data type\n", tensor.name().c_str());
    }
    return data_ptr;
}

const onnx::TensorProto *GetTensorFromConstantNode(const onnx::NodeProto &constant_node) {
    for (int i = 0; i < constant_node.attribute_size(); ++i) {
        const auto &attribute_proto = constant_node.attribute(i);
        const auto &attribute_name  = attribute_proto.name();
        if (attribute_name == "value") {
            return &attribute_proto.t();
        }
    }
    return nullptr;
}

TNN_NS::DimsVector CreateDimsVectorFromTensor(const onnx::TensorProto &tensor) {
    TNN_NS::DimsVector dims = {};
    const auto &tensor_dims = tensor.dims();
    if (tensor_dims.empty()) {
        return dims;
    }
    for (int i = 0; i < tensor_dims.size(); ++i) {
        dims.push_back((int)tensor.dims(i));
    }
    return dims;
}

void CreateRawBufferFromTensor(const onnx::TensorProto &tensor, TNN_NS::RawBuffer **raw_buffer) {
    int count               = GetTensorProtoDataSize(tensor);
    const auto &tensor_dims = CreateDimsVectorFromTensor(tensor);
    switch (tensor.data_type()) {
        case onnx::TensorProto_DataType_INT64: {
            int64_t *tensor_data_ptr = (int64_t *)(tensor.raw_data().data());
            // raw_buffer = std::make_shared<TNN_NS::RawBuffer>(data_count * sizeof(int32_t)).get();
            *raw_buffer = new TNN_NS::RawBuffer(count * sizeof(int32_t), tensor_dims);
            (*raw_buffer)->SetDataType(TNN_NS::DATA_TYPE_INT32);
            auto tmp = new int32_t[count]();
            for (int i = 0; i < count; ++i) {
                tmp[i] = (int)tensor_data_ptr[i];
            }
            memcpy((*raw_buffer)->force_to<void *>(), tmp, count * sizeof(int32_t));
            delete[] tmp;
            break;
        }
        case onnx::TensorProto_DataType_FLOAT: {
            auto raw_data_ptr = tensor.raw_data().data();
            *raw_buffer       = new TNN_NS::RawBuffer(count * sizeof(float), tensor_dims);
            (*raw_buffer)->SetDataType(TNN_NS::DATA_TYPE_FLOAT);
            memcpy((*raw_buffer)->force_to<void *>(), (void *)raw_data_ptr, count * sizeof(float));
            break;
        }
        default: {
            LOGE("Converter: do not support onnx tensor type\n");
        }
    }
}

void CreateRawBufferFromConstant(const onnx::NodeProto &constant_node, TNN_NS::RawBuffer **raw_buffer) {
    ASSERT(constant_node.op_type() == "Constant");
    onnx::TensorProto tensor;
    for (int i = 0; i < constant_node.attribute_size(); ++i) {
        const auto &attribute_proto = constant_node.attribute(i);
        const auto &attribute_name  = attribute_proto.name();
        if (attribute_name == "value") {
            tensor = attribute_proto.t();
            break;
        }
    }
    switch (tensor.data_type()) {
        case onnx::TensorProto_DataType_INT64: {
            auto data_count             = 1;
            const void *tensor_data_ptr = tensor.raw_data().data();
            // raw_buffer = std::make_shared<TNN_NS::RawBuffer>(data_count * sizeof(int32_t)).get();
            *raw_buffer = new TNN_NS::RawBuffer(data_count * sizeof(int32_t), TNN_NS::DimsVector({1}));
            (*raw_buffer)->SetDataType(TNN_NS::DATA_TYPE_INT32);
            int value = *((int64_t *)tensor_data_ptr);
            memcpy((*raw_buffer)->force_to<int32_t *>(), &value, data_count * sizeof(int32_t));
            break;
        }
        default: {
            LOGE("Converter: do not support onnx tensor type\n");
        }
    }
}

/**
 * onnx::TensorProto_DataType data_type
 * */
int TensorProtoDataType2TnnDataType(int data_type) {
    if (onnx::TensorProto_DataType_FLOAT == data_type) {
        return TNN_NS::DATA_TYPE_FLOAT;
    } else if (onnx::TensorProto_DataType_FLOAT16 == data_type) {
        return TNN_NS::DATA_TYPE_HALF;
    } else if (onnx::TensorProto_DataType_INT8 == data_type) {
        return TNN_NS::DATA_TYPE_INT8;
    } else if (onnx::TensorProto_DataType_INT32 == data_type || onnx::TensorProto_DataType_INT64 == data_type) {
        return TNN_NS::DATA_TYPE_INT32;
    } else {
        LOGE("Converter: TensorProtoDataType2TnnDataType do not support type\n");
        assert(0);
    }
}

TNN_NS::DataType GetTnnDataTypeFromOnnx(const onnx::TensorProto_DataType &onnx_type) {
    switch (onnx_type) {
        case onnx::TensorProto_DataType_FLOAT: {
            return TNN_NS::DATA_TYPE_FLOAT;
        }
        case onnx::TensorProto_DataType_FLOAT16: {
            return TNN_NS::DATA_TYPE_HALF;
        }
        case onnx::TensorProto_DataType_UINT8:
        case onnx::TensorProto_DataType_INT8: {
            return TNN_NS::DATA_TYPE_INT8;
        }
        case onnx::TensorProto_DataType_INT64:
        case onnx::TensorProto_DataType_INT32: {
            return TNN_NS::DATA_TYPE_INT32;
        }
        case onnx::TensorProto_DataType_BFLOAT16: {
            return TNN_NS::DATA_TYPE_BFP16;
        }
        default: {
            LOGE("Not support onnx TypeProto type: %d", onnx_type);
            assert(0);
        }
    }
}

template <class T>
std::vector<T> GetTensorProtoDataVector(const onnx::TensorProto &tp) {
    std::vector<T> data_vec;
    //    TensorProto_DataType_FLOAT = 1,
    int size  = GetTensorProtoDataSize(tp);
    T *data_T = nullptr;
    if (tp.has_raw_data()) {
        const std::string &raw_data = tp.raw_data();
        data_T                      = (T *)raw_data.data();
    } else if (tp.data_type() == 1) {
        data_T = (T *)tp.float_data().data();
    } else if (tp.data_type() == 6) {
        data_T = (T *)tp.int32_data().data();
    } else if (tp.data_type() == 7) {
        data_T = (T *)tp.int64_data().data();
    } else if (tp.data_type() == 11) {
        data_T = (T *)tp.double_data().data();
    } else {
        LOGE("name:%s data_type :%d\n", tp.name().c_str(), tp.data_type());
        assert(0);
        return data_vec;
    }

    for (int i = 0; i < size; i++) {
        data_vec.push_back(data_T[i]);
    }
    return data_vec;
}

TNN_NS::Status GetWeightInputIndexName(int &weight_input_index, std::string &weight_name, const onnx::NodeProto &node,
                                       std::map<std::string, const onnx::TensorProto *> proxy_initializers_map,
                                       std::map<std::string, std::shared_ptr<OnnxProxyNode>> &proxy_nodes) {
    weight_input_index        = -1;
    weight_name               = "";
    const int node_input_size = node.input_size();
    for (int i = 0; i < node_input_size; i++) {
        const auto &input_name = node.input(i);
        if (proxy_initializers_map.find(input_name) == proxy_initializers_map.end()) {
            continue;
        }
        if (weight_input_index != -1) {
            LOGE("Binary: Only support one weight input index\n");
            return TNN_NS::TNNERR_CONVERT_UNSUPPORT_LAYER;
        }
        weight_input_index = i;
        weight_name        = input_name;
    }

    return TNN_NS::TNN_CONVERT_OK;
}

}  // namespace TNN_CONVERTER
