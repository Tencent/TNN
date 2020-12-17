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

#include <fstream>
#include <iostream>
#include <sstream>
#include "onnx_op_converter.h"
#include "onnx_utility.h"

#include "half_utils.h"

template <class T>
onnx::TensorProto MakeTensor(const std::string &name, const std::vector<T> &v,
                             const std::vector<int> &shape, onnx::TensorProto_DataType data_type)
{
    onnx::TensorProto tensor;

    tensor.set_name(name);
    for (auto dim : shape)
    {
        tensor.add_dims(dim);
    }
    tensor.set_data_type(data_type);
    tensor.mutable_raw_data()->assign(
            reinterpret_cast<const char *>(v.data()), v.size() * sizeof(T));

    return tensor;
}

DECLARE_OP_CONVERTER(Gemm);

string OnnxOpConverterGemm::TNNOpType(NodeProto& node,
                                      OnnxNetInfo &net_info) {
    float alpha = get_node_attr_f(node, "alpha", 1.f);
    float beta  = get_node_attr_f(node, "beta", 1.f);
    int transA  = (int)get_node_attr_i(node, "transA", 0);
    int transB  = (int)get_node_attr_i(node, "transB", 0);
    if (std::abs(alpha - 1.f) <= 1e-6) {
        // InnerProduct-like A * B + C
        if (transA == 0) {
            return "InnerProduct";
        }
    }
    return "";
}

string OnnxOpConverterGemm::TNNLayerParam(NodeProto& node,
                                          OnnxNetInfo& net_info) {
    ostringstream layer_param;

    const std::string& onnx_op = node.op_type();

    float alpha   = get_node_attr_f(node, "alpha", 1.f);
    float beta    = get_node_attr_f(node, "beta", 1.f);
    int broadcast = (int)get_node_attr_i(node, "broadcast", 0);
    int transA    = (int)get_node_attr_i(node, "transA", 0);
    int transB    = (int)get_node_attr_i(node, "transB", 0);

    if (!(beta == 1 || beta == 0)) {
        DLog("error::Gemm convert failed: beta should be 0 or 1\n");
        assert(0);
    }

    if (alpha == 1.f) {
        // InnerProduct-like A * B + C
        if (transA == 0) {
            int axis = 1;  // Fix TODO
            const onnx::TensorProto& weights =
                    net_info.weights_map[node.input(1)];
            int channel_output =
                    transB ? (int)weights.dims(0) : (int)weights.dims(1);
            int has_bias = 1;
            layer_param << channel_output << " " << has_bias << " "
                        << (int)0 << " " << axis << " ";
        }
    } else {
        DLog("error::Gemm convert failed:transA(%d) transB(%d)\n", transA,
             transB);
        assert(0);
    }

    return layer_param.str();
}

int OnnxOpConverterGemm::WriteTNNModel(serializer* net_writer,
                                       NodeProto& node,
                                       OnnxNetInfo& net_info) {
    const std::string& onnx_op = node.op_type();
    std::string name = !node.name().empty() ? node.name() : node.output(0);
    const std::string& tnn_layer_type = TNNOpType(node, net_info);

    float alpha = get_node_attr_f(node, "alpha", 1.f);
    float beta  = get_node_attr_f(node, "beta", 1.f);
    int transA  = (int)get_node_attr_i(node, "transA", 0);
    int transB  = (int)get_node_attr_i(node, "transB", 0);
    if (alpha == 1.f) {
        // InnerProduct-like A * B + C
        if (transA == 0) {
            int axis = 1;  // Fix TODO

            //写头信息
            net_writer->put_int(0);  //触发type from string
            net_writer->put_string(tnn_layer_type);
            net_writer->put_string(name);

            //写数据
            //对应innerproduct_data的反序列化
            net_writer->put_string(name);
            auto B = get_node_attr_tensor(node, "B", net_info, 1);
            if (transB == 1) {
                WriteTensorData(B, net_writer, net_info.data_type);
            } else {
                auto const h = B.dims(0);
                auto const w = B.dims(1);

                float* permuted_data = new float[h * w];
                auto bptr = get_tensor_proto_data(B);

                float* permuted_data_ptr = permuted_data;
                for (int j = 0; j < w; j++) {
                    for (int k = 0; k < h; k++) {
                        float vb = bptr[k * w + j];
                        *permuted_data_ptr = vb;
                        permuted_data_ptr++;
                    }
                }

                WriteRawData(permuted_data, (int)(h * w), net_writer, net_info.data_type);
                delete [] permuted_data;
            }

            int num_bias = B.dims(1);
            if (transB == 1) {
                num_bias = B.dims(0);
            }

            onnx::TensorProto bias;
            if (node.input_size() == 3) {
                bias = get_node_attr_tensor(node, "C", net_info, 2);
            } else {
                std::vector<int> bias_shape = {num_bias};
                std::vector<float> bias_data(num_bias, 0.0f);
                bias = MakeTensor("C", bias_data, bias_shape, onnx::TensorProto::FLOAT);
            }
            WriteTensorData(bias, net_writer, net_info.data_type);
        }
    }

    //有权值写入的返回1， 没有的返回0
    return 1;
}

REGISTER_OP_CONVERTER(Gemm, Gemm);
