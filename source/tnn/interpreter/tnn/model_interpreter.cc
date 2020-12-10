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

#include "tnn/interpreter/tnn/model_interpreter.h"
#include <stdlib.h>
#include <sstream>

#include "tnn/core/common.h"
#include "tnn/interpreter/tnn/layer_interpreter/abstract_layer_interpreter.h"
#include "tnn/interpreter/tnn/objseri.h"

namespace TNN_NS {

TypeModelInterpreterRegister<TypeModelInterpreterCreator<ModelInterpreter>> g_tnn_model_interpreter_register(
    MODEL_TYPE_TNN);

std::string ModelInterpreter::Transfer(std::string content) {
    return content;
}

// Check if the magic number is valid.
bool ModelInterpreter::IsValidVersionNumber(uint32_t number) {
    return number == g_version_magic_number;
}

std::shared_ptr<Deserializer> ModelInterpreter::GetDeserializer(std::istream &is) {
    return std::make_shared<Deserializer>(is);
}

// Interpret the proto and model.
Status ModelInterpreter::Interpret(std::vector<std::string> &params) {
    std::string empty_content = "";

    auto &proto_content = params.size() > 0 ? params[0] : empty_content;
    Status status       = InterpretProto(proto_content);
    if (status != TNN_OK) {
        return status;
    }

    auto &model_content = params.size() > 1 ? params[1] : empty_content;
    status              = InterpretModel(model_content);
    return status;
}

Status ModelInterpreter::InterpretProto(std::string &content) {
    Status ret              = TNN_OK;
    NetStructure *structure = GetNetStructure();
    // NOTE??????
    structure->source_model_type = MODEL_TYPE_TNN;

    /*
     * each line of tnn proto File is in this format :
     *  "xxxxxxxxx,"
     * Here we remove the leading and tailing " and \n
     */
    int size           = static_cast<int>(content.size());
    char *proto_buffer = new char[size + 1];
    size_t fill        = 0;
    for (size_t i = 0; i < size; ++i) {
        if (content[i] != '\"' && content[i] != '\n') {
            proto_buffer[fill++] = content[i];
        }
    }
    proto_buffer[fill] = '\0';

    str_arr cfg_arr;
    if (fill == 0) {
        delete[] proto_buffer;
        return Status(TNNERR_INVALID_NETCFG, "proto content is empty");
    }

    ret = SplitUtils::SplitStr(proto_buffer, cfg_arr, ",", true, false);
    delete[] proto_buffer;
    if (ret != TNN_OK) {
        return Status(TNNERR_INVALID_NETCFG, "split proto error");
    }
    if (cfg_arr.empty() || cfg_arr.size() <= 5) {
        return Status(TNNERR_INVALID_NETCFG, "content line <= 5");
    }

    {  // get magic number
        str_arr cfg_line0;
        ret = SplitUtils::SplitStr(cfg_arr[0].c_str(), cfg_line0, " ", true, false);
        if (ret != TNN_OK) {
            return ret;
        }
        if (cfg_line0.size() >= 4) {
            this->version_magic_number = atoll(cfg_line0[3].c_str());
        }
    }

    std::string inputs_content = cfg_arr[1];
    ret                        = InterpretInput(inputs_content);
    if (ret != TNN_OK) {
        return ret;
    }
    std::string outputs_content = cfg_arr[3];
    ret                         = InterpretOutput(outputs_content);
    if (ret != TNN_OK) {
        return ret;
    }

    for (int i = layer_cfg_start_id; i < cfg_arr.size(); i++) {
        std::string layer_str = cfg_arr.at(i);
        if (layer_str.empty()) {
            continue;
        }
        ret = InterpretLayer(layer_str);
        if (ret != TNN_OK) {
            return ret;
        }
    }

    return TNN_OK;
}

Status ModelInterpreter::InterpretInput(const std::string &inputs_content) {
    NetStructure *structure = GetNetStructure();
    str_arr inputs_cfg_vec;
    /*
     * input list is separated by : symbol
     * eg:
     *  input0 1 3 384 128 : input1 1 3 64 64
     */
    Status ret = SplitUtils::SplitStr(inputs_content.c_str(), inputs_cfg_vec, ":", true, false);
    if (ret != TNN_OK) {
        return Status(TNNERR_INVALID_NETCFG, "split input line error");
    }
    for (int i = 0; i < inputs_cfg_vec.size(); i++) {
        str_arr input_cfg_vec;
        ret = SplitUtils::SplitStr(inputs_cfg_vec[i].c_str(), input_cfg_vec, " ", true, false);
        if (ret != TNN_OK || input_cfg_vec.size() < input_layer_cfg_count) {
            return Status(TNNERR_INVALID_NETCFG, "split input line error");
        }
        DimsVector &input_shape = structure->inputs_shape_map[input_cfg_vec[0]];
        // input_shape.set_name(input_cfg_vec[0]);
        input_shape.push_back(atoi(input_cfg_vec[1].c_str()));
        input_shape.push_back(atoi(input_cfg_vec[2].c_str()));
        input_shape.push_back(atoi(input_cfg_vec[3].c_str()));
        input_shape.push_back(atoi(input_cfg_vec[4].c_str()));
        if (input_cfg_vec.size() > 5) {
            input_shape.push_back(atoi(input_cfg_vec[5].c_str()));
        }
    }
    return TNN_OK;
}

Status ModelInterpreter::InterpretOutput(const std::string &outputs_content) {
    NetStructure *structure = GetNetStructure();
    str_arr output_cfg_vec;
    Status ret = SplitUtils::SplitStr(outputs_content.c_str(), output_cfg_vec, " ", true, false);
    if (ret != TNN_OK || output_cfg_vec.size() <= 0) {
        return Status(TNNERR_INVALID_NETCFG, "split output line error");
    }
    for (auto iter : output_cfg_vec) {
        structure->outputs.insert(iter);
    }
    return TNN_OK;
}

Status ModelInterpreter::InterpretLayer(const std::string &layer_str) {
    NetStructure *structure     = GetNetStructure();
    auto &layer_interpreter_map = GetLayerInterpreterMap();
    str_arr layer_cfg_arr;
    Status ret = SplitUtils::SplitStr(layer_str.c_str(), layer_cfg_arr, " ", true, true);
    if (ret != TNN_OK || layer_cfg_arr.empty()) {
        return Status(TNNERR_INVALID_NETCFG, "split layer info error");
    }

    auto cur_layer = std::make_shared<LayerInfo>();
    // 0.LayerType;1.layer_name;2.input_count;3.output_count
    std::string type_str = layer_cfg_arr[0];
    type_str             = Transfer(type_str);
    LayerType type       = GlobalConvertLayerType(type_str);
    if (type == LAYER_NOT_SUPPORT) {
        LOGE("Error: layer type %s is not supported.\n", layer_cfg_arr[0].c_str());
        return Status(TNNERR_PARAM_ERR, "layer type is not supported");
    }
    cur_layer->type     = type;
    cur_layer->type_str = type_str;
    cur_layer->name     = Transfer(layer_cfg_arr[1]);

    int in_count = atoi(layer_cfg_arr[2].c_str());
    cur_layer->inputs.clear();
    int out_count = atoi(layer_cfg_arr[3].c_str());
    cur_layer->outputs.clear();
    int in_id  = layer_param_start_id;
    int in_end = in_id + in_count;

    cur_layer->inputs.reserve(std::max(in_end - in_id, 1));
    for (; in_id < in_end; in_id++) {
        auto blob_name = Transfer(layer_cfg_arr[in_id]);
        cur_layer->inputs.push_back(blob_name);
        structure->blobs.insert(blob_name);
    }

    int out_id  = in_end;
    int out_end = out_id + out_count;

    cur_layer->outputs.reserve(std::max(out_end - out_id, 1));
    for (; out_id < out_end; out_id++) {
        auto blob_name = Transfer(layer_cfg_arr[out_id]);
        cur_layer->outputs.push_back(blob_name);
        structure->blobs.insert(blob_name);
    }

    LayerParam *param      = NULL;
    auto layer_interpreter = layer_interpreter_map[type];
    if (layer_interpreter != NULL) {
        layer_interpreter->InterpretProto(layer_cfg_arr, out_end, &param);
    }

    if (!param) {
        param = new LayerParam();
    }
    // is quantized
    if (type_str.compare(0, 9, "Quantized") == 0) {
        param->quantized = true;
    }

    // type
    if (param && layer_cfg_arr.size() >= 1) {
        param->type = cur_layer->type_str;
    }

    // name
    if (param && layer_cfg_arr.size() >= 2) {
        param->name = cur_layer->name;
    }

    cur_layer->param = shared_ptr<LayerParam>(param);

    if (ret != TNN_OK) {
        return TNNERR_INVALID_NETCFG;
    }

    structure->layers.push_back(cur_layer);
    return TNN_OK;
}

Status ModelInterpreter::InterpretModel(std::string &model_content) {
    NetResource *net_resource = GetNetResource();

    const auto model_length = model_content.length();
    if (model_length <= 0) {
#ifdef GENERATE_RESOURCE
        LOGD("model content is empty, will generate random data\n");
        return TNN_OK;
#else
        return Status(TNNERR_LOAD_MODEL, "model content is invalid");
#endif
    }

    std::istringstream content_stream;
    content_stream.str(model_content);

    uint32_t magic_version_number = 0;
    content_stream.read(reinterpret_cast<char *>(&magic_version_number), sizeof(g_version_magic_number));
    if (!IsValidVersionNumber(magic_version_number)) {
        content_stream.seekg(0, std::ios::beg);
    }

    res_header header;
    auto deserializer = GetDeserializer(content_stream);
    header.deserialize(*deserializer);
    if (header.layer_cnt_ <= 0 || header.layer_cnt_ >= 10000) {
        return Status(TNNERR_INVALID_MODEL, "Error: model is illegal");
    }

    auto &layer_interpreter_map = GetLayerInterpreterMap();
    for (int index = 0; index < header.layer_cnt_; ++index) {
        layer_header ly_head;
        ly_head.deserialize(*deserializer);

        LayerResource *layer_resource = NULL;
        auto layer_interpreter        = layer_interpreter_map[ly_head.type_];
        // refactor later, layer_interpreter NULL return error_code.
        if (layer_interpreter != NULL) {
            Status result = layer_interpreter->InterpretResource(*deserializer, &layer_resource);
            if (result != TNN_OK) {
                return result;
            }
            net_resource->resource_map[ly_head.name_] = std::shared_ptr<LayerResource>(layer_resource);
        } else {
            LOGE(
                "Error: layer_interpreter nil name:%s type_from_str:%s "
                "type:%d\n",
                ly_head.name_.c_str(), ly_head.type_str_.c_str(), ly_head.type_);
            return Status(TNNERR_LOAD_MODEL, "Error: layer_interpreter is nil");
        }
    }

    return TNN_OK;
}

Status ModelInterpreter::RegisterLayerInterpreter(LayerType type, AbstractLayerInterpreter *interpreter) {
    std::map<LayerType, std::shared_ptr<AbstractLayerInterpreter>> &layer_interpreter_map = GetLayerInterpreterMap();
    layer_interpreter_map[type] = std::shared_ptr<AbstractLayerInterpreter>(interpreter);
    return TNN_OK;
}

std::map<LayerType, std::shared_ptr<AbstractLayerInterpreter>> &ModelInterpreter::GetLayerInterpreterMap() {
    static std::map<LayerType, std::shared_ptr<AbstractLayerInterpreter>> layer_interpreter_map;
    return layer_interpreter_map;
}

}  // namespace TNN_NS
