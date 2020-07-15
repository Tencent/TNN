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
#include "tnn/core/macro.h"

namespace tnn {

Status NpuUtils::CreateAttrValue(shared_ptr<ge::op::Const> attr_value, ge::Shape shape, RawBuffer &raw_buffer) {
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
        LOGE("The file length equals to 0 build model fail");
        return Status(TNNERR_HIAI_API_ERROR, " write model to om fail");
    }
    std::ofstream file(file_path.c_str(), std::ios::binary);
    file.write(static_cast<char *>(model_buffer_data.data), model_buffer_data.length);
    file.close();
    return TNN_OK;
}

Status NpuUtils::CreateAttrArray(std::shared_ptr<ge::op::Const> &attr_value, std::vector<int> calculate_shape,
                                 ge::TensorDesc input_desc) {
    ge::AttrValue::TENSOR input_size_tensor = std::make_shared<ge::Tensor>(input_desc);
    input_size_tensor->SetData((uint8_t *)calculate_shape.data(), sizeof(int) * 4);
    attr_value->set_attr_value(input_size_tensor);
    return Status();
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
        } else {
            LOGE("Error: unsupported broadcast type\n");
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

Status NpuUtils::GetPadMode(int &pad_mode, int pad_type, bool depthwise, bool depthwise_same) {
    // npu pad mode
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
        if (depthwise) {
            if (depthwise_same) {
                pad_mode = 6;
                return TNN_OK;
            }
            LOGE("Error: Npu ConvLayerDepthwise does not support current pad type, neither valid nor same\n");
            return Status(TNNERR_PARAM_ERR,
                          "Error: Npu ConvLayerDepthwise dont support current pad type, neither valid nor same");
        } else {
            pad_mode = 0;
        }
    } else {
        LOGE("Error: ConvLayer dont support pad type: %d\n", pad_type);
        return Status(TNNERR_PARAM_ERR, "Error: ConvLayer dont support pad type");
    }
    return TNN_OK;
}
}  // namespace tnn
