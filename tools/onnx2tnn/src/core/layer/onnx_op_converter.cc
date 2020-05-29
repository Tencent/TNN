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

#include "onnx_op_converter.h"
#include <mutex>
#include "half_utils.h"
#include "onnx_utility.h"

string OnnxOpConverter::TNNLayerProto(NodeProto &node,
                                           OnnxNetInfo &net_info) {
    ostringstream proto_layer;

    string tnn_layer_type = TNNOpType(node, net_info);
    proto_layer << "\"" << tnn_layer_type << " ";

    std::string name = node.name();
    if (name.empty()) {
        name = node.output(0);
    }
    proto_layer << name << " ";

    int input_size  = node.input_size();
    int output_size = node.output_size();

    for (int j = 0; j < (int)node.input_size(); j++) {
        const std::string &input_name = node.input(j);
        if (net_info.weights_map.find(input_name) !=
            net_info.weights_map.end()) {
            input_size--;
        }
    }
    proto_layer << input_size << " " << output_size << " ";

    for (int j = 0; j < node.input_size(); j++) {
        std::string input_name = node.input(j);

        // check weight
        if (net_info.weights_map.find(input_name) !=
            net_info.weights_map.end()) {
            continue;
        }

        proto_layer << input_name << " ";
    }

    for (int j = 0; j < output_size; j++) {
        const std::string &output_name = node.output(j);
        proto_layer << output_name << " ";
    }

    string param = TNNLayerParam(node, net_info);
    proto_layer << param << ",\"";
    return proto_layer.str();
}

int OnnxOpConverter::WriteTensorData(const onnx::TensorProto &tensor,
                                     serializer *writer, DataType dataType) {
    int ret = 0;
    do {
        int item_size = get_tensor_proto_data_size(tensor);
        if (item_size == 0) {
            DLog("invalid size\n");
            assert(0);
            break;
        }

        if (tensor.has_raw_data()) {
            const std::string &raw_data = tensor.raw_data();
            WriteRawData((float *)raw_data.data(), item_size, writer, dataType);
        } else if (tensor.data_type() == 1) {
            WriteRawData((float *)tensor.float_data().data(), item_size, writer,
                         dataType);
        } else if (tensor.data_type() == 6) {
            int32_t *raw_data = (int32_t *)tensor.int32_data().data();
            float *temp = new float[item_size];
            for (int i=0; i<item_size; i++) {
                temp[i] = raw_data[i];
            }
            WriteRawData(temp, item_size, writer, dataType);
        } else if (tensor.data_type() == 7) {
            int64_t *raw_data = (int64_t *)tensor.int64_data().data();
            float *temp = new float[item_size];
            for (int i=0; i<item_size; i++) {
                temp[i] = raw_data[i];
            }
            WriteRawData(temp, item_size, writer, dataType);
            delete [] temp;
        } else {
            DLog("invalid tensor type\n");
            assert(0);
            break;
        }
    } while (0);
    return ret;
}

int OnnxOpConverter::WriteRawData(const float *raw_data, int data_count,
                                  serializer *writer, DataType dataType) {
    int ret = 0;
    do {
        if (data_count == 0 || !raw_data) {
            DLog("invalid data or size\n");
            assert(0);
            break;
        }

        if (dataType == DATA_TYPE_FLOAT) {
            writer->put_raw(data_count * sizeof(float), (char *)raw_data);
        } else if (dataType == DATA_TYPE_HALF) {
            float16 *half_data = new float16[data_count];
            ret = TNN_NS::ConvertFromFloatToHalf((float *)raw_data, (void *)half_data, data_count);
            writer->put_raw(data_count * sizeof(float16), (char *)half_data, DATA_TYPE_HALF);
            delete[] half_data;
        }
    } while (0);
    return ret;
}

OnnxOpConverterManager::OnnxOpConverterManager() {}

OnnxOpConverterManager::~OnnxOpConverterManager() {}

std::shared_ptr<OnnxOpConverterManager> &OnnxOpConverterManager::Shared() {
    static std::once_flag once;
    static std::shared_ptr<OnnxOpConverterManager>
        g_global_onnx_op_converter_manager;
    std::call_once(once, []() {
        g_global_onnx_op_converter_manager =
            std::make_shared<OnnxOpConverterManager>();
    });
    return g_global_onnx_op_converter_manager;
}

std::shared_ptr<OnnxOpConverter> OnnxOpConverterManager::GetOnnxOpConverter(
    string onnx_type) {
    auto iter = converter_map_.find(onnx_type);
    if (iter != converter_map_.end()) {
        return iter->second;
    }
    return nullptr;
}

int OnnxOpConverterManager::SetOnnxOpConverter(
    string onnx_type, std::shared_ptr<OnnxOpConverter> converter) {
    auto iter = converter_map_.find(onnx_type);
    if (iter != converter_map_.end()) {
        DLog("Error: onnx_type(%s) cannot be registered twice\n",
             onnx_type.c_str());
        assert(0);
        return 1;
    }
    converter_map_[onnx_type] = converter;
    return 0;
}
