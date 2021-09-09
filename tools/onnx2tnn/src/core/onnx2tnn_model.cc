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

#include <math.h>

#include "onnx2tnn.h"
#ifdef __fp16
typedef __fp16 float16;
#else
typedef uint16_t float16;
#endif

int Onnx2TNN::TNNWriteModel() {
    int ret = 0;

    std::ofstream file_model;
    file_model.open(tnn_model_path_, std::ios::binary);
    file_model.write((char *)(&g_version_magic_number_v2), sizeof(g_version_magic_number_v2));
    int model_pos = sizeof(g_version_magic_number_v2);

    do {
        if (!file_model || !file_model.is_open() || !file_model.good()) {
            file_model.close();
            DLog("model file open failed\n");
            assert(0);
            break;
        }

        Serializer net_writer(file_model);

        const onnx::GraphProto& graph = onnx_model_->graph();

        //统计含有权值的层数
        int weight_layer_count = 0;
        //层数 含有权值的层数 先占位后更正
        net_writer.PutInt(weight_layer_count);
        //        file_model.write(reinterpret_cast<char*>(&weight_layer_count),
        //        sizeof(int));

        //写入每层的权值
        for (int i = 0; i < graph.node_size(); i++) {
            onnx::NodeProto& node      = (onnx::NodeProto&)graph.node(i);
            const std::string& onnx_op = node.op_type();
            const auto& used_const_node = this->onnx_net_info_.used_const_node;
            if (onnx_op == k_tnn_noop_type || ( onnx_op == "Constant" && used_const_node.find(node.output(0)) == used_const_node.end() )) {
                continue;
            }

            auto op_converter =
                OnnxOpConverterManager::Shared()->GetOnnxOpConverter(onnx_op);
            if (op_converter == nullptr) {
                fprintf(stderr, "get op convert for %s failed\n",onnx_op.c_str());
                assert(0);
            }
            weight_layer_count += op_converter->WriteTNNModel(
                &net_writer, node, onnx_net_info_);
        }

        //更正含有权值的层数
        file_model.seekp(model_pos, ios::beg);
        net_writer.PutInt(weight_layer_count);
        file_model.seekp(0, ios::end);
        
        //写入constant_map
        {
            std::set<std::string> const_id_set;
            
            //写入每层的constant输入（除了已经写入layerresource的，如conv）
            for (int i = 0; i < graph.node_size(); i++) {
                onnx::NodeProto& node      = (onnx::NodeProto&)graph.node(i);
                const std::string& onnx_op = node.op_type();
                const auto& used_const_node = this->onnx_net_info_.used_const_node;
                if (onnx_op == k_tnn_noop_type || ( onnx_op == "Constant" && used_const_node.find(node.output(0)) == used_const_node.end() )) {
                    continue;
                }

                auto op_converter =
                    OnnxOpConverterManager::Shared()->GetOnnxOpConverter(onnx_op);
                if (op_converter == nullptr) {
                    fprintf(stderr, "get op convert for %s failed\n",onnx_op.c_str());
                    assert(0);
                }
                
                for (int j = 0; j < (int)node.input_size(); j++) {
                    const std::string &input_name = node.input(j);
                    //some op like ConstantOfShape, its input(0) may be const but it is not in layer resource
                    if ( (j==0 || !op_converter->HasLayerResource(node, onnx_net_info_)) &&
                        onnx_net_info_.weights_map.find(input_name) != onnx_net_info_.weights_map.end() ) {
                        const_id_set.insert(input_name);
                    }
                }
            }
            
            if (const_id_set.size() < 0) {
                break;
            }
            
            //write version number
            net_writer.PutInt(g_version_magic_number_v2);
            //write const count
            net_writer.PutInt((int)const_id_set.size());
            for (auto id : const_id_set) {
                auto const_tensor = onnx_net_info_.weights_map[id];
                net_writer.PutString(id);
                if (const_tensor.data_type() == TensorProto_DataType_FLOAT ||
                    const_tensor.data_type() == TensorProto_DataType_DOUBLE) {
                    OnnxOpConverter::WriteTensorData(const_tensor, &net_writer, onnx_net_info_.data_type);
                } else {
                    OnnxOpConverter::WriteTensorData(const_tensor, &net_writer, DATA_TYPE_AUTO);
                }
            }
        }
    } while (0);
    
    file_model.close();

    return ret;
}
