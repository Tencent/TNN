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

#ifndef onnx_utility_hpp
#define onnx_utility_hpp
#include <float.h>
#include <limits.h>
#include <stdio.h>

#include <iostream>

#include <algorithm>
#include <fstream>
#include <memory>
#include <limits>
#include <set>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "onnx.pb.h"
#include "onnx_op_converter.h"

std::vector<int64_t> get_tensor_proto_reshape_shape(const onnx::TensorProto& tp);

bool node_has_attr(const onnx::NodeProto &node, const char *key);

onnx::AttributeProto *get_node_mutable_attr(onnx::NodeProto &node,
                                            const char *key);
std::vector<int64_t> get_node_attr_ai(const onnx::NodeProto& node, const char* key);

std::vector<int64_t> get_node_attr_ai(const onnx::NodeProto &node, const char *key,
                                      const OnnxNetInfo& net_info, int number);

std::vector<int64_t> get_node_attr_ai(const onnx::NodeProto &node, const char *key,
                                      const TensorProtoMap &weights, int number);

std::vector<float> get_node_attr_af(const onnx::NodeProto &node, const char *key);

std::vector<float> get_node_attr_af(const onnx::NodeProto& node, const char* key,
                                    const OnnxNetInfo &net_info, const int number);

int64_t get_node_attr_i(const onnx::NodeProto &node, const char *key,
                        int def = 0);
float get_node_attr_f(const onnx::NodeProto &node, const char *key,
                      float def = 0.f);
float get_node_attr_f(const onnx::NodeProto &node, const char *key,
                      const OnnxNetInfo &net_info, const int number,
                      float def = 0.f);
double get_node_attr_d(const onnx::NodeProto &node, const char *key,
                       const OnnxNetInfo &net_info, const int number,
                       double def = 0.f);
std::string get_node_attr_s(const onnx::NodeProto &node, const char *key,
                            const std::string &def = std::string());
onnx::TensorProto get_node_attr_tensor(const onnx::NodeProto &node,
                                       const char *key);
onnx::TensorProto get_node_attr_tensor(const onnx::NodeProto &node, const char *key,
                                       const OnnxNetInfo &net_info, const int number);

int get_tensor_proto_data_size(const onnx::TensorProto &tp);
const float *get_tensor_proto_data(const onnx::TensorProto &tp);
float *get_tensor_proto_mutable_data(onnx::TensorProto &tp);

std::string replace_node_name(std::string str, char older_value, char new_value);

template <class T>
std::vector<T> get_tensor_proto_data_vector(const onnx::TensorProto &tp) {
    std::vector<T> data_vec;
   //    TensorProto_DataType_FLOAT = 1,
    int size  = get_tensor_proto_data_size(tp);
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
        printf("name:%s data_type :%d\n", tp.name().c_str(), tp.data_type());
        assert(0);
        return data_vec;
    }

    for (int i = 0; i < size; i++) {
        data_vec.push_back(data_T[i]);
    }
    return data_vec;
}

int read_proto_from_binary(const char *filepath,
                           google::protobuf::Message *message);

bool set_node_attr_ai(onnx::NodeProto& node, const char* key, std::vector<int64_t> values);

std::vector<int> GetDimsFromTensor(const onnx::TensorProto& tensor);
std::vector<int> GetDimsFromTensorShape(const onnx::TensorShapeProto& tensor);

DataType GetTnnDataTypeFromOnnx(const onnx::TypeProto& onnx_type);
DataType GetTnnDataTypeFromOnnx(long long int onnx_data_type);

std::vector<int> CreateDimsVectorFromTensor(const onnx::TensorProto& tensor);

#endif /* onnx_utility_hpp */
