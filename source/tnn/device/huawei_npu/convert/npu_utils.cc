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

#include "npu_utils.h"
#include <tnn/interpreter/layer_resource.h>
#include <tnn/utils/dims_vector_utils.h>
#include <sstream>
#include "tnn/core/macro.h"

namespace tnn {

Status NpuUtils::CreateAttrValue(shared_ptr<ge::op::Const>& attr_value, ge::Shape shape, RawBuffer &raw_buffer) {
    ge::TensorDesc desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::TensorPtr tensor_ptr = std::make_shared<ge::Tensor>();

    tensor_ptr->SetTensorDesc(desc);
    tensor_ptr->SetData(raw_buffer.force_to<uint8_t *>(), raw_buffer.GetBytesSize());

    attr_value->set_attr_value(tensor_ptr);
    return TNN_OK;
}

Status NpuUtils::CreateInputData(std::shared_ptr<ge::op::Data> &input_data, std::string &input_name,
                                 DimsVector dims_vector) {
    int n = dims_vector[0];
    int c = dims_vector[1];
    int h = dims_vector[2];
    int w = dims_vector[3];

    ge::Shape data_shape({n, c, h, w});
    ge::TensorDesc desc(data_shape, ge::FORMAT_NCHW, ge::DT_FLOAT);

    input_data = std::make_shared<ge::op::Data>(input_name);
    input_data->update_input_desc_x(desc);
    return TNN_OK;
}

Status NpuUtils::WriteModelFile(domi::ModelBufferData &model_buffer_data, std::string file_path) {
    int file_length = model_buffer_data.length;
    if (file_length == 0) {
        LOGE("ERROR: The file length equals to 0 build model fail\n");
        return Status(TNNERR_NPU_HIAI_API_ERROR, "ERROR: The file length equals to 0 build model fail");
    }
    std::ofstream file(file_path.c_str(), std::ios::binary);
    file.write(static_cast<char *>(model_buffer_data.data), model_buffer_data.length);
    file.close();
    return TNN_OK;
}

Status NpuUtils::CalculateBroadcastSize(vector<int> &weight, EltwiseLayerResource *layer_res, vector<int> &input) {
    int input_count = DimsVectorUtils::Count(input, 1);
    if (weight.size() < 4) {
        weight             = {1, 1, 1, 1};
        int layer_res_size = layer_res->element_handle.GetDataCount();
        if (layer_res_size == 1) {
            // single element
            weight[1] = layer_res_size;
        } else if (layer_res_size == input[1]) {
            // channel broadcast
            weight[1] = layer_res_size;
        } else if (layer_res_size == input_count) {
            // element broadcast
            weight[1] = input[1];
            weight[2] = input[2];
            weight[3] = input[3];
        } else if (layer_res_size == input[3]) {
            weight[3] = input[3];
        } else {
            return Status(TNNERR_LAYER_ERR, "Error: unsupported broadcast type");
        }
        layer_res->element_shape = weight;
    }
    return TNN_OK;
}

std::string NpuUtils::GetFileHash(ModelConfig &model_config) {
    std::string file_content = model_config.params[1] + model_config.params[0];
    int hash                 = 0;
    for (size_t i = 0; i < file_content.length(); ++i)
        hash = 65599 * hash + file_content.at(i);
    return std::to_string(hash ^ (hash >> 16));
}

bool NpuUtils::FileExits(std::string model_path) {
    std::ifstream infile(model_path);
    return infile.good();
}

Status NpuUtils::GetPadMode(int &pad_mode, int pad_type) {
    // huawei_npu pad mode
    if (pad_type == 0) {
        // pad_type : SAME_UPPER or SAME_LOWER
        // pad_mode : SAME
        pad_mode = 6;
    } else if (pad_type == 1) {
        // pad_type : VALID
        // pad_mode : VALID 5
        pad_mode = 5;
    } else if (pad_type == -1) {
        // pad_type : NOSET
        pad_mode = 0;
    } else {
        return Status(TNNERR_PARAM_ERR, "Error: ConvLayer dont support pad type");
    }
    return TNN_OK;
}

int NpuUtils::checkNpuVersion(const char *version) {
    // ddk version's format: xxx.xxx.xxx.xxx
    std::string version_s(version);
    size_t pos = std::string::npos;
    int count = 0, update_index = 1;
    while ((pos = version_s.find(".")) != std::string::npos) {
        std::string curr_update = version_s.substr(0, pos);
        if (count == update_index) {
            return std::stoi(curr_update.c_str());
        }
        version_s.erase(0, pos + 1);
        count++;
    }
    return 0;
}

std::string NpuUtils::modifyModelInputSize(InputShapesMap &inputs_shape, InputShapesMap &instance_input_shapes_map) {
    std::stringstream model_suffix_stream("");
    for (auto iter : inputs_shape) {
        if (instance_input_shapes_map.count(iter.first) > 0 && instance_input_shapes_map[iter.first] != iter.second) {
            instance_input_shapes_map[iter.first] = iter.second;
            model_suffix_stream << "_" << iter.first << "[";
            DimsVector value = iter.second;
            for (size_t i = 0; i < value.size(); ++i) {
                if (i != 0) {
                    model_suffix_stream << "x";
                }
                model_suffix_stream << value[i];
            }
            model_suffix_stream << "]";
        }
    }
    return model_suffix_stream.str();
}
}  // namespace tnn
