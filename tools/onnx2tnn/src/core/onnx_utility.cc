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

#include "onnx_utility.h"
#include "onnx2tnn_prefix.h"
#include "onnx_op_converter.h"

std::vector<int64_t> get_tensor_proto_reshape_shape(
    const onnx::TensorProto& tp) {
    const int64_t* shape_data = 0;
    int size                  = 0;

    // int64
    if (tp.has_raw_data()) {
        shape_data = (const int64_t*)tp.raw_data().data();
        size       = tp.raw_data().size() / 8;
    } else if (tp.data_type() == 7) {
        shape_data = tp.int64_data().data();
        size       = tp.int64_data_size();
    }

    std::vector<int64_t> shape;
    for (int j = 0; j < size; j++) {
        shape.push_back(shape_data[j]);
    }

    return shape;
}

bool node_has_attr(const onnx::NodeProto& node, const char* key) {
    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key) {
            return true;
        }
    }

    return false;
}

onnx::AttributeProto* get_node_mutable_attr(onnx::NodeProto& node,
                                            const char* key) {
    for (int i = 0; i < node.attribute_size(); i++) {
        onnx::AttributeProto* attr = node.mutable_attribute(i);
        if (attr->name() == key) {
            return attr;
        }
    }
    return nullptr;
}

std::vector<int64_t> get_node_attr_ai(const onnx::NodeProto& node,
                                      const char* key) {
    std::vector<int64_t> v;

    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key) {
            v.resize(attr.ints_size());
            for (int j = 0; j < attr.ints_size(); j++) {
                v[j] = attr.ints(j);
            }

            break;
        }
    }
    return v;
}
/*
 * description：
 *  获取operator 中 attribute 值。
 *  首先通过 attribute 的 key 来获取 attribute 的 value。
 *  如果在 attribute 中查找不到，会继续在 inputs 中进行查找（兼容 onnx 1.6.0）。
 * return：
 *  返回 vector<int64>
 * note:
 *  inputs 中的数据类型包含 int32 以及 int64
 * */
std::vector<int64_t> get_node_attr_ai(const onnx::NodeProto& node,
                                      const char* key,
                                      const OnnxNetInfo& net_info, int number) {
    std::vector<int64_t> array_i;

    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key) {
            array_i.resize(attr.ints_size());
            for (int j = 0; j < attr.ints_size(); j++) {
                array_i[j] = attr.ints(j);
            }
            break;
        }
    }
    if (array_i.empty()) {
        // get params from inputs
        if (number < 0) {
            number += node.input_size();
        }
        assert(number >= 0);
        if (number > node.input_size() - 1) {
            LOGD("the number > node.input_size(), key:%s\n", key);
            return array_i;
        }
        const string& name = node.input(number);
        LOGD("name :%s\n", name.c_str());
        if (net_info.weights_map.find(name) == net_info.weights_map.end()) {
            LOGD("input %d name:%s is not weight\n", number, name.c_str());
            return array_i;
        }

        const onnx::TensorProto& tensorProto = net_info.weights_map.at(name);
        if (tensorProto.data_type() == onnx::TensorProto_DataType_INT32) {
            std::vector<int32_t> array_temp = get_tensor_proto_data_vector<int32_t>(tensorProto);
            array_i.clear();
            for (const auto item : array_temp) {
                array_i.emplace_back(item);
            }
        } else {
            array_i = get_tensor_proto_data_vector<int64_t>(tensorProto);
        }
    }
    return array_i;
}

/*
 * description：
 *  获取operator 中 attribute 值。
 *  首先通过 attribute 的 key 来获取 attribute 的 value。
 *  如果在 attribute 中查找不到，会继续在 inputs 中进行查找（兼容 onnx 1.6.0）。
 * return：
 *  返回 vector<int64>
 * note:
 *  inputs 中的数据类型包含 int32 以及 int64
 * */
std::vector<int64_t> get_node_attr_ai(const onnx::NodeProto& node,
                                      const char* key,
                                      const TensorProtoMap& weights_map,
                                      int number) {
    std::vector<int64_t> array_i;

    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key) {
            array_i.resize(attr.ints_size());
            for (int j = 0; j < attr.ints_size(); j++) {
                array_i[j] = attr.ints(j);
            }
            break;
        }
    }
    if (array_i.empty()) {
        // get params from inputs
        if (number < 0) {
            number += node.input_size();
        }
        assert(number >= 0);
        if (number > node.input_size() - 1) {
            LOGD("the number > node.input_size(), key:%s\n", key);
            return array_i;
        }
        const string& name = node.input(number);
        LOGD("name :%s\n", name.c_str());
        if (weights_map.find(name) == weights_map.end()) {
            LOGD("input %d name:%s is not weight\n", number, name.c_str());
            return array_i;
        }

        const onnx::TensorProto& tensorProto = weights_map.at(name);
        if (tensorProto.data_type() == onnx::TensorProto_DataType_INT32) {
            std::vector<int32_t> array_temp = get_tensor_proto_data_vector<int32_t>(tensorProto);
            array_i.clear();
            for (const auto item : array_temp) {
                array_i.emplace_back(item);
            }
        } else {
            array_i = get_tensor_proto_data_vector<int64_t>(tensorProto);
        }
    }
    return array_i;
}

bool set_node_attr_ai(onnx::NodeProto& node, const char* key, std::vector<int64_t> values){
    for (int i = 0; i < node.attribute_size(); ++i) {
        auto attr = node.mutable_attribute(i);
        if (attr->name() == key) {
            for (int j = 0; j < values.size(); j++) {
                attr->set_ints(j, values[j]);
            }
            return true;
        }
    }
    return false;
}

std::vector<float> get_node_attr_af(const onnx::NodeProto& node,
                                    const char* key) {
    std::vector<float> v;

    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key) {
            v.resize(attr.floats_size());
            for (int j = 0; j < attr.floats_size(); j++) {
                v[j] = attr.floats(j);
            }

            break;
        }
    }

    return v;
}

/*
 * description：
 *  获取operator 中 attribute 值。
 *  首先通过 attribute 的 key 来获取 attribute 的 value。
 *  如果在 attribute 中查找不到，会继续在 inputs 中进行查找（兼容 onnx 1.6.0）。
 * return：
 *  返回 vector<float>
 * note:
 *  inputs 中的数据类型包含 float
 * */
std::vector<float> get_node_attr_af(const onnx::NodeProto& node,
                                    const char* key,
                                    const OnnxNetInfo& net_info,
                                    const int number) {
    std::vector<float> v;

    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key) {
            v.resize(attr.floats_size());
            for (int j = 0; j < attr.floats_size(); j++) {
                v[j] = attr.floats(j);
            }
            break;
        }
    }

    if (v.size() > 0) {
        return v;
    }

    if (number < 0 || number >= node.input_size()) {
        LOGD("invalid number %d for input_size:%d\n", number,
             node.input_size());
        LOGD("node output %s key %s", node.output(0).c_str(), key);
        assert(0);
        return v;
    }

    // get attribute from inputs
    const string& name = node.input(number);
    if (net_info.weights_map.find(name) == net_info.weights_map.end()) {
        LOGD("invalid name for input: %s\n", name.c_str());
        return v;
    }
    const onnx::TensorProto& tensorProto = net_info.weights_map.at(name);

    const int size = get_tensor_proto_data_size(tensorProto);
    v              = get_tensor_proto_data_vector<float>(tensorProto);
    return v;
}
/*
 * description：
 *  获取op 中某些 attribute 的值。 先通过 attribute 的key 来获取 attribute 的
 * value。 如果在 attribute 中查找不到，会在 inputs 中进行查找。这是为了兼容
 * onnx 1.6.0 的标准.
 *
 * return：
 *  返回 vector<double>.
 *
 * note:
 *  将返回值设置为 double，是因为 onnx 1.6.0 中的 inputs 的类型为float 和 double
 * 类型
 * */
std::vector<double> get_node_attr_ad(const onnx::NodeProto& node,
                                     const char* key,
                                     const OnnxNetInfo& net_info,
                                     const int number) {
    std::vector<double> v;

    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key) {
            v.resize(attr.floats_size());
            for (int j = 0; j < attr.floats_size(); j++) {
                // float -> double
                v[j] = attr.floats(j);
            }
            break;
        }
    }
    if (v.empty()) {
        // get params from inputs
        assert(number >= 0);
        if (number > node.input_size() - 1) {
            LOGD("the number > node.input_size()\n");
            return v;
        }
        const string& name                   = node.input(number);
        const onnx::TensorProto& tensorProto = net_info.weights_map.at(name);
        if (tensorProto.data_type() == TensorProto_DataType_FLOAT) {
            const int size    = tensorProto.float_data_size();
            const float* data = tensorProto.float_data().data();
            v.resize(size);
            for (int i = 0; i < size; ++i) {
                // float -> double
                v[i] = data[i];
            }
        } else if (tensorProto.data_type() == TensorProto_DataType_DOUBLE) {
            const int size     = tensorProto.double_data_size();
            const double* data = tensorProto.double_data().data();
            v.resize(size);
            for (int i = 0; i < size; ++i) {
                v[i] = data[i];
            }
        } else {
            LOGD("not support the type");
        }
    }

    return v;
}

int64_t get_node_attr_i(const onnx::NodeProto& node, const char* key, int def) {
    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key) {
            return attr.i();
        }
    }
    return def;
}

float get_node_attr_f(const onnx::NodeProto& node, const char* key, float def) {
    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key) {
            return attr.f();
        }
    }

    return def;
}

float get_node_attr_f(const onnx::NodeProto& node, const char* key,
                      const OnnxNetInfo& net_info, const int number,
                      float def) {
    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key) {
            return attr.f();
        }
    }

    if (number < 0 || number >= node.input_size()) {
        LOGD("invalid number for input\n");
        return def;
    }

    // get attribute from inputs
    if (number > node.input_size()) {
        DLog("invalid number for input size: %s\n", node.name().c_str());
        return def;
    }
    const string& name = node.input(number);
    if (net_info.weights_map.find(name) == net_info.weights_map.end()) {
        LOGD("invalid name for input: %s\n", name.c_str());
        return def;
    }
    const onnx::TensorProto& tensorProto = net_info.weights_map.at(name);

    const int size          = get_tensor_proto_data_size(tensorProto);
    const float* tensorData = get_tensor_proto_data(tensorProto);
    if (size > 0) {
        return tensorData[0];
    } else {
        LOGD("TensorProto is invalid");
    }
    return def;
}

double get_node_attr_d(const onnx::NodeProto& node, const char* key,
                       const OnnxNetInfo& net_info, const int number,
                       double def) {
    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key) {
            // float -> double
            return attr.f();
        }
    }
    // get attribute from inputs

    if (number < 0 || number >= node.input_size()) {
        LOGD("invalid number for input\n");
        assert(0);
        return def;
    }

    const string& name = node.input(number);
    if (net_info.weights_map.find(name) == net_info.weights_map.end()) {
        LOGD("invalid name for input: %s\n", name.c_str());
        //        assert(0);
        return def;
    }

    const onnx::TensorProto& tensorProto = net_info.weights_map.at(name);

    const int size = get_tensor_proto_data_size(tensorProto);
    const double* tensorData =
        (const double*)get_tensor_proto_data(tensorProto);
    if (size > 0) {
        return tensorData[0];
    } else {
        LOGD("TensorProto is invalid");
    }
    return def;
}

std::string get_node_attr_s(const onnx::NodeProto& node, const char* key,
                            const std::string& def) {
    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key) {
            return attr.s();
        }
    }

    return def;
}

onnx::TensorProto get_node_attr_tensor(const onnx::NodeProto& node,
                                       const char* key) {
    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key) {
            return attr.t();
        }
    }

    return onnx::TensorProto();
}

onnx::TensorProto get_node_attr_tensor(const onnx::NodeProto& node,
                                       const char* key,
                                       const OnnxNetInfo& net_info,
                                       const int number) {
    for (int i = 0; i < node.attribute_size(); i++) {
        const onnx::AttributeProto& attr = node.attribute(i);
        if (attr.name() == key) {
            return attr.t();
        }
    }

    if (number < 0 || number >= node.input_size()) {
        LOGD("invalid number for input\n");
        assert(0);
        return onnx::TensorProto();
    }

    const string& name = node.input(number);
    if (net_info.weights_map.find(name) == net_info.weights_map.end()) {
        LOGD("invalid name for input: %s\n", name.c_str());
        return onnx::TensorProto();
    }

    auto tensorProto = net_info.weights_map.at(name);
    return tensorProto;
}

int get_tensor_proto_data_size(const onnx::TensorProto& tp) {
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
                DLog("Onnx Converter: do not support tensor proto data type\n");
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
                DLog("Onnx Converter: do not support tensor proto data type\n");
                size = -1;
            }
        }
    }
    return size;
}

const float* get_tensor_proto_data(const onnx::TensorProto& tp) {
    //    TensorProto_DataType_UNDEFINED = 0,
    //    TensorProto_DataType_FLOAT = 1,
    //    TensorProto_DataType_UINT8 = 2,
    //    TensorProto_DataType_INT8 = 3,
    //    TensorProto_DataType_UINT16 = 4,
    //    TensorProto_DataType_INT16 = 5,
    //    TensorProto_DataType_INT32 = 6,
    //    TensorProto_DataType_INT64 = 7,
    //    TensorProto_DataType_STRING = 8,
    //    TensorProto_DataType_BOOL = 9,
    //    TensorProto_DataType_FLOAT16 = 10,
    //    TensorProto_DataType_DOUBLE = 11,
    //    TensorProto_DataType_UINT32 = 12,
    //    TensorProto_DataType_UINT64 = 13,
    //    TensorProto_DataType_COMPLEX64 = 14,
    //    TensorProto_DataType_COMPLEX128 = 15,
    //    TensorProto_DataType_BFLOAT16 = 16
    if (tp.has_raw_data()) {
        return (const float*)tp.raw_data().data();
    } else if (tp.data_type() == 1) {
        return tp.float_data().data();
    } else if (tp.data_type() == 6) {
        return (const float*)tp.int32_data().data();
    } else if (tp.data_type() == 7) {
        return (const float*)tp.int64_data().data();
    } else if (tp.data_type() == 11) {
        return (const float*)tp.double_data().data();
    } else {
        printf("name:%s data_type :%d\n", tp.name().c_str(), tp.data_type());
        assert(0);
        return nullptr;
    }
}

float* get_tensor_proto_mutable_data(onnx::TensorProto& tp) {
    if (tp.has_raw_data()) {
        return (float*)tp.mutable_raw_data()->data();
    } else if (tp.data_type() == 1) {
        return tp.mutable_float_data()->mutable_data();
    } else if (tp.data_type() == 6) {
        return (float*)tp.mutable_int32_data()->mutable_data();
    } else if (tp.data_type() == 7) {
        return (float*)tp.mutable_int64_data()->mutable_data();
    } else if (tp.data_type() == 11) {
        return (float*)tp.mutable_double_data()->mutable_data();
    } else {
        assert(0);
        return nullptr;
    }
}

int read_proto_from_binary(const char* filepath,
                           google::protobuf::Message* message) {
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open()) {
        fprintf(stderr, "open failed %s\n", filepath);
        return -1;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

#if GOOGLE_PROTOBUF_VERSION >= 3002000
    codedstr.SetTotalBytesLimit(INT_MAX);
#else
    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
#endif

    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success ? 0 : -1;
}

std::string replace_node_name(std::string str, char older_value,
                              char new_value) {
    if (str.find(older_value) != std::string::npos) {
        replace(str.begin(), str.end(), older_value, new_value);
        return str;
    }
    return str;
}

std::vector<int> GetDimsFromTensor(const onnx::TensorProto& tensor) {
    std::vector<int> dims = {};
    for (const auto dim: tensor.dims()) {
        dims.push_back(int(dim));
    }
    return dims;
}

std::vector<int> GetDimsFromTensorShape(const onnx::TensorShapeProto& shape) {
    std::vector<int> dims = {};
    for (const auto &item : shape.dim()) {
        dims.push_back((int)item.dim_value());
    }
    return dims;
}

DataType GetTnnDataTypeFromOnnx(const onnx::TypeProto& onnx_type) {
    return GetTnnDataTypeFromOnnx(onnx_type.tensor_type().elem_type());
}

DataType GetTnnDataTypeFromOnnx(long long int onnx_data_type) {
    //keep the same as cast op
    switch (onnx_data_type) {
        case onnx::TensorProto_DataType_FLOAT:
        case onnx::TensorProto_DataType_DOUBLE:{
            return DATA_TYPE_FLOAT;
        }
        case onnx::TensorProto_DataType_FLOAT16: {
            return DATA_TYPE_HALF;
        }
        case onnx::TensorProto_DataType_BOOL: //INT8 BOOL(sizeof(bool) == sizeof(char))
        case onnx::TensorProto_DataType_UINT8:
        case onnx::TensorProto_DataType_INT8: {
            return DATA_TYPE_INT8;
        }
        case onnx::TensorProto_DataType_INT64:
        case onnx::TensorProto_DataType_INT32:
        case onnx::TensorProto_DataType_UINT32:
        case onnx::TensorProto_DataType_UINT64: {
            return DATA_TYPE_INT32;
        }
        case onnx::TensorProto_DataType_BFLOAT16: {
            return DATA_TYPE_BFP16;
        }
        default:{
            DLog("Not support onnx TypeProto type: %d",(int) onnx_data_type);
            assert(0);
        }
    }
    return DATA_TYPE_AUTO;
}

std::vector<int> CreateDimsVectorFromTensor(const onnx::TensorProto& tensor) {
    std::vector<int> dims = {};
    const auto& tensor_dims = tensor.dims();
    if (tensor_dims.empty()) {
        return dims;
    }
    for (int i = 0; i < tensor_dims.size(); i++) {
        dims.push_back((int)tensor.dims(i));
    }
    return dims;
}
