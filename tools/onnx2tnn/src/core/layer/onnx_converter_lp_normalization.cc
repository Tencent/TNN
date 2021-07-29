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



DECLARE_OP_CONVERTER(LpNormalization);

string OnnxOpConverterLpNormalization::TNNOpType(NodeProto& node,
                                                OnnxNetInfo &net_info) {
    return "Normalize";
}

string OnnxOpConverterLpNormalization::TNNLayerParam(NodeProto& node,
                                                    OnnxNetInfo& net_info) {


    int across_spatial_dims = 0;
    float epsilon = get_node_attr_f(node, "epsilon", 1e-10); // default esp from pytorch.
    int channel_shared = 0;
    int dim = (int)get_node_attr_i(node, "axis", 1);
    int p = (int)get_node_attr_i(node, "p", 2);

    ostringstream layer_param;
    layer_param <<across_spatial_dims<<" "<<epsilon<<" "<<channel_shared<<" "<<dim<<" "<<p<<" ";
    
    return layer_param.str();
}

bool OnnxOpConverterLpNormalization::HasLayerResource(NodeProto &node, OnnxNetInfo &net_info) {
    bool convert_for_old_model = false;
    if (convert_for_old_model) {
        return true;
    } else {
        return false;
    }
}

int OnnxOpConverterLpNormalization::WriteTNNModel(Serializer* net_writer,
                                                 NodeProto& node,
                                                 OnnxNetInfo& net_info) {
    bool convert_for_old_model = false;
    if (convert_for_old_model) {
        const onnx::TensorProto& scale = net_info.weights_map[node.input(1)];

        const std::string& onnx_op = node.op_type();
        std::string name = !node.name().empty() ? node.name() : node.output(0);
        const std::string& tnn_layer_type = TNNOpType(node, net_info);

        //写头信息
        net_writer->PutInt(0);  //触发type from string
        net_writer->PutString(tnn_layer_type);
        net_writer->PutString(name);

        //写数据
        // TNN 默认需要scale 作为weights, 长度是channel数量。
        // 暂时未知通道数量，临时按1024来写weights

        const float scale_data = 1.0f;
        int channels = 1024;

        float* k = new float[channels];

        // TNN 需要scale参数，onnx 模型中未提供，所以默认初始化为1.0f
        for(int i=0;i<channels;i++) {
            k[i] = 1.0f;
        }

        WriteRawData(k, channels, net_writer, net_info.data_type, {channels});

        delete [] k;


        return 1;
    } else {
        return 0;
    }
}

REGISTER_OP_CONVERTER(LpNormalization, LpNormalization);
