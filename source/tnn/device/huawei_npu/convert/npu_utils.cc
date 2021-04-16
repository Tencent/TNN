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
#include <stdlib.h>
#include <sstream>
#include "tnn/core/macro.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/split_utils.h"

namespace TNN_NS {

Status NpuUtils::CreateAttrValue(shared_ptr<ge::op::Const> &attr_value, ge::Shape shape, RawBuffer &raw_buffer) {
    ge::TensorPtr tensor_ptr = std::make_shared<ge::Tensor>();
    ge::Format format        = ge::FORMAT_RESERVED;
    if (shape.GetDims().size() == 4) {
        format = ge::FORMAT_NCHW;
    } else {
        format = ge::FORMAT_ND;
    }
    if (raw_buffer.GetDataType() == DATA_TYPE_FLOAT || raw_buffer.GetDataType() == DATA_TYPE_HALF) {
        ge::TensorDesc desc(shape, format, ge::DT_FLOAT);
        tensor_ptr->SetTensorDesc(desc);
    } else if (raw_buffer.GetDataType() == DATA_TYPE_INT32) {
        ge::TensorDesc desc(shape, format, ge::DT_INT32);
        tensor_ptr->SetTensorDesc(desc);
    } else {
        LOGE("raw buffer data type is not support in CreateAttrValue");
        return Status(TNNERR_INVALID_DATA, "raw buffer data type is not support in CreateAttrValue");
    }
    if (raw_buffer.GetDataType() == DATA_TYPE_HALF) {
        // if filter handle is half, need convert to float first.
        std::shared_ptr<float> float_data_ptr = GetFloatFromRawBuffer(raw_buffer);
        if (float_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        tensor_ptr->SetData((uint8_t *)float_data_ptr.get(), raw_buffer.GetDataCount() * sizeof(float));
    } else if (raw_buffer.GetDataType() == DATA_TYPE_FLOAT) {
        tensor_ptr->SetData(raw_buffer.force_to<uint8_t *>(), raw_buffer.GetBytesSize());
    } else if (raw_buffer.GetDataType() == DATA_TYPE_INT32) {
        tensor_ptr->SetData(raw_buffer.force_to<uint8_t *>(), raw_buffer.GetBytesSize());
    } else {
        tensor_ptr->SetData(raw_buffer.force_to<uint8_t *>(), raw_buffer.GetBytesSize());
    }
    attr_value->set_attr_value(tensor_ptr);
    return TNN_OK;
}

Status NpuUtils::CreateInputData(std::shared_ptr<ge::op::Data> &input_data, std::string &input_name,
                                 DimsVector dims_vector) {
    // TO-DO: support int32 input
    ge::TensorDesc desc(ge::Shape(NpuUtils::Int32VecToTVec<int64_t>(dims_vector)), ge::FORMAT_NCHW, ge::DT_FLOAT);

    input_data = std::make_shared<ge::op::Data>(input_name);
    input_data->update_input_desc_x(desc);
    return TNN_OK;
}

Status NpuUtils::CreateConstOpFromResource(std::shared_ptr<OperatorInfo> &const_op, std::string name,
                                           NetResource *net_resource) {
    if (net_resource->constant_map.count(name) > 0) {
        auto raw_buffer      = net_resource->constant_map[name];
        auto data_const      = std::make_shared<ge::op::Const>(name + "_const");
        auto raw_buffer_dims = raw_buffer->GetBufferDims();
        ge::Shape const_shape(NpuUtils::Int32VecToTVec<int64_t>(raw_buffer_dims));
        auto ret = NpuUtils::CreateAttrValue(data_const, const_shape, *raw_buffer);
        if (ret != TNN_OK) {
            LOGE("The input op of layer which is found in resource convert to const op failed\n");
            return ret;
        }
        const_op = std::make_shared<OperatorInfo>(data_const);
        const_op->SetShape(raw_buffer_dims);
    } else {
        LOGE("The input op of layer is not found in resource");
        return Status(TNNERR_LAYER_ERR, "The input op of layer is not found in resource");
    }

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

static bool IsNumberString(std::string num_str) {
    const char *num_char = num_str.c_str();

    for (int i = 0; i < num_str.length(); ++i) {
        if (!(num_char[i] >= '0' && num_char[i] <= '9')) {
            return false;
        }
    }
    return true;
}

bool NpuUtils::IsVersionValid(std::string version) {
    str_arr version_vec;
    auto ret = SplitUtils::SplitStr(version.c_str(), version_vec, ".");
    if (ret != TNN_OK) {
        LOGE("split npu version failed (str: %s)\n", version.c_str());
        return false;
    }

    if (version_vec.size() != 4) {
        return false;
    }

    for (auto val : version_vec) {
        if (!IsNumberString(val) && val != "xxx")
            return false;
    }

    return true;
}

bool NpuUtils::VersionCompare(std::string version, std::string cmp, VersionCompareType type) {
    if (!IsVersionValid(version) || !IsVersionValid(cmp)) {
        LOGE("invalid version(s1: %s  s2: %s)\n", version.c_str(), cmp.c_str());
        return false;
    }

    str_arr version_vec;
    str_arr cmp_vec;

    auto ret = SplitUtils::SplitStr(version.c_str(), version_vec, ".");
    if (ret != TNN_OK) {
        LOGE("split npu version failed (str: %s)\n", version.c_str());
        return false;
    }

    ret = SplitUtils::SplitStr(cmp.c_str(), cmp_vec, ".");
    if (ret != TNN_OK) {
        LOGE("split npu version failed (str: %s)\n", cmp.c_str());
        return false;
    }

    for (unsigned int i = 0; i < version_vec.size(); ++i) {
        int version_val = 0;
        int cmp_val     = 0;

        if (version_vec[i] == "xxx") {
            version_val = -1;
        }
        if (cmp_vec[i] == "xxx") {
            cmp_val = -1;
        }

        version_val = atoi(version_vec[i].c_str());
        cmp_val     = atoi(cmp_vec[i].c_str());

        if (VCT_SMALLER == type || VCT_SMALLEQUAL == type) {
            if (version_val < cmp_val) {
                return true;
            } else if (version_val > cmp_val) {
                return false;
            }
        } else if (VCT_BIGGER == type || VCT_BIGEQUAL == type) {
            if (version_val < cmp_val) {
                return false;
            } else if (version_val > cmp_val) {
                return true;
            }
        }
    }

    if (VCT_SMALLEQUAL == type || VCT_BIGEQUAL == type) {
        return true;
    } else {
        return false;
    }
}

void NpuUtils::SplitNetwork(const int cpu_count, NetStructure *net_structure, std::set<std::string> &visited,
                            std::map<std::string, shared_ptr<OperatorInfo>> &global_operator_map) {
    std::vector<shared_ptr<LayerInfo>> layers;
    // only view input
    InputShapesMap sub_input_shapes_map;
    for (int i = cpu_count; i < net_structure->layers.size(); i++) {
        std::shared_ptr<LayerInfo> &layer_info = net_structure->layers[i];
        for (std::string &input : layer_info->inputs) {
            // if the subnetwork input exists in visited
            if (visited.count(input) > 0) {
                // if the input has not defined in new inputShapeMap yet
                if (sub_input_shapes_map.count(input) == 0) {
                    sub_input_shapes_map[input] = global_operator_map[input]->GetShape();
                }
            }
        }
        layers.push_back(layer_info);
    }
    net_structure->layers           = layers;
    net_structure->inputs_shape_map = sub_input_shapes_map;

    // remove outputs which has visited
    std::set<std::string> outputs;
    for (auto output : net_structure->outputs) {
        if (visited.count(output) == 0) {
            outputs.insert(output);
        }
    }
    net_structure->outputs = outputs;
}

ge::DataType NpuUtils::ConvertToHiaiDataType(TNN_NS::DataType tnn_dtype) {
    if (DATA_TYPE_FLOAT == tnn_dtype) {
        return ge::DT_FLOAT;
    } else if (DATA_TYPE_HALF == tnn_dtype) {
        return ge::DT_FLOAT16;
    } else if (DATA_TYPE_INT8 == tnn_dtype) {
        return ge::DT_INT8;
    } else if (DATA_TYPE_INT32 == tnn_dtype) {
        return ge::DT_INT32;
    } else if (DATA_TYPE_BFP16 == tnn_dtype) {
        return ge::DT_UNDEFINED;
    }
    return ge::DT_UNDEFINED;
}

}  // namespace TNN_NS
