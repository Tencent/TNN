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

#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/ncnn/layer_interpreter/abstract_layer_interpreter.h"
#include "tnn/interpreter/ncnn/ncnn_layer_type.h"
#include "tnn/interpreter/ncnn/ncnn_model_interpreter.h"
#include "tnn/interpreter/ncnn/ncnn_param_utils.h"
#include "tnn/interpreter/ncnn/optimizer/ncnn_optimizer_manager.h"

namespace TNN_NS {

namespace ncnn {

#define RETURN_ON_ERROR(status) RETURN_ON_NEQ(status, RPD_OK)

    TypeModelInterpreterRegister<TypeModelInterpreterCreator<NCNNModelInterpreter>> g_ncnn_model_interpreter_register(
        MODEL_TYPE_NCNN);

    Status NCNNModelInterpreter::Interpret(std::vector<std::string> params) {
        std::string proto_content = params.size() > 0 ? params[0] : "";
        RETURN_ON_ERROR(InterpretProto(proto_content));
        std::string model_content = params.size() > 1 ? params[1] : "";
        RETURN_ON_ERROR(InterpretModel(model_content));
        RETURN_ON_ERROR(NCNNOptimizerManager::Optimize(GetNetStructure(), GetNetResource()));
        RETURN_ON_ERROR(FindOutputs());

        return RPD_OK;
    }

    Status NCNNModelInterpreter::FindOutputs() {
        NetStructure *structure = GetNetStructure();
        auto layers             = structure->layers;
        auto blobs              = structure->blobs;
        std::set<std::string> out_blobs;

        for (auto layer : layers) {
            for (auto out_blob : layer->outputs) {
                out_blobs.insert(out_blob);
            }

            for (auto in_blob : layer->inputs) {
                if (out_blobs.find(in_blob) != out_blobs.end()) {
                    out_blobs.erase(in_blob);
                }
            }
        }

        structure->outputs = out_blobs;

        return RPD_OK;
    }

    Status NCNNModelInterpreter::Convert(shared_ptr<LayerInfo> cur_layer,
                                         std::vector<std::shared_ptr<LayerInfo>> output_layers) {
        return RPD_OK;
    }

    Status NCNNModelInterpreter::InterpretProto(std::string content) {
        Status ret                   = RPD_OK;
        NetStructure *structure      = GetNetStructure();
        structure->source_model_type = MODEL_TYPE_NCNN;

        int size = static_cast<int>(content.size());

        char *proto_buffer = new char[size + 1];
        size_t fill        = 0;
        for (size_t i = 0; i < size; ++i) {
            proto_buffer[fill++] = content[i];
        }
        proto_buffer[fill] = '\0';

        // 0. Split lines
        str_arr cfg_arr;
        ret = SplitUtils::SplitStr(proto_buffer, cfg_arr, "\n", true, false);
        delete[] proto_buffer;
        if (ret != RPD_OK) {
            return Status(RPDERR_INVALID_NETCFG, "split proto error");
        }
        if (cfg_arr.empty() || cfg_arr.size() <= 3) {
            return Status(RPDERR_INVALID_NETCFG, "content line <= 3");
        }

        // 1. Parse ncnn magic number
        int magic_number = atoi(cfg_arr[0].c_str());
        if (magic_number != ncnn_magic_number) {
            return Status(RPDERR_INVALID_NETCFG, "invalid_magic_number");
        }

        // 2. Parse ncnn layer cnt and blob cnt
        std::string layer_cfg_content = cfg_arr[1];
        str_arr layer_cfg_vec;
        ret = SplitUtils::SplitStr(layer_cfg_content.c_str(), layer_cfg_vec, " ", true, false);
        if (ret != RPD_OK || layer_cfg_vec.size() != 2) {
            return Status(RPDERR_INVALID_NETCFG, "split layer cnt failed");
        }

        int layer_cnt = atoi(layer_cfg_vec[0].c_str());
        int blob_cnt  = atoi(layer_cfg_vec[1].c_str());

        // 3. Parse ncnn input layer
        std::string input_cfg_content = cfg_arr[input_layer_id];
        str_arr input_cfg_vec;
        ret = SplitUtils::SplitStr(input_cfg_content.c_str(), input_cfg_vec, " ", true, true);
        if (ret != RPD_OK) {
            return Status(RPDERR_INVALID_NETCFG, "split input layer failed");
        }

        auto input_name = input_cfg_vec[layer_param_start_id];

        DimsVector input_shape = DimsVector();

        if (input_cfg_vec.size() > 5) {
            str_arr hwc_param_str(input_cfg_vec.begin() + 5, input_cfg_vec.end());
            str_dict hwc_vec;
            ret = SplitUtils::SplitParamList(hwc_param_str, hwc_vec);
            if (ret != RPD_OK) {
                LOGE("%s\n", ret.description().c_str());
                return Status(RPDERR_INVALID_NETCFG, "split input layer failed");
            }

            // Default batch size 1
            input_shape.push_back(1);
            input_shape.push_back(GetInt(hwc_vec, 2, 0));  // c
            input_shape.push_back(GetInt(hwc_vec, 1, 0));  // h
            input_shape.push_back(GetInt(hwc_vec, 0, 0));  // w
        }

        structure->inputs_shape_map[input_name] = input_shape;

        auto &layer_interpreter_map = GetLayerInterpreterMap();

        for (int i = layer_cfg_start_id; i < cfg_arr.size(); i++) {
            str_arr layer_cfg_arr;
            std::string layer_str = cfg_arr.at(i);
            if (layer_str.empty()) {
                continue;
            }
            // 0. Split layer str
            ret = SplitUtils::SplitStr(layer_str.c_str(), layer_cfg_arr, " ", true, true);
            if (ret != RPD_OK || layer_cfg_arr.empty()) {
                return Status(RPDERR_INVALID_NETCFG, "split layer info error");
            }

            auto cur_layer = std::make_shared<LayerInfo>();
            // 0.LayerType;1.layer_name;2.input_count;3.output_count
            std::string type_str = layer_cfg_arr[0];

            cur_layer->type_str = type_str;
            cur_layer->type     = LAYER_NOT_SUPPORT;
            cur_layer->name     = layer_cfg_arr[1];

            // 1. Parse in out nodes
            int in_count = atoi(layer_cfg_arr[2].c_str());
            cur_layer->inputs.clear();
            int out_count = atoi(layer_cfg_arr[3].c_str());
            cur_layer->outputs.clear();
            int in_id  = layer_param_start_id;
            int in_end = in_id + in_count;

            cur_layer->inputs.reserve(std::max(in_end-in_id, 1));
            for (; in_id < in_end; in_id++) {
                cur_layer->inputs.push_back(layer_cfg_arr[in_id]);
                structure->blobs.insert(layer_cfg_arr[in_id]);
            }

            int out_id  = in_end;
            int out_end = out_id + out_count;

            cur_layer->outputs.reserve(std::max(out_end-out_id, 1));
            for (; out_id < out_end; out_id++) {
                cur_layer->outputs.push_back(layer_cfg_arr[out_id]);
                structure->blobs.insert(layer_cfg_arr[out_id]);
            }

            // 2. Split param dict
            str_arr param_arr(layer_cfg_arr.begin() + out_end, layer_cfg_arr.end());
            str_dict param_dict;
            ret = SplitUtils::SplitParamList(param_arr, param_dict);
            if (ret != RPD_OK) {
                LOGE("%s\n", ret.description().c_str());
                return Status(RPDERR_INVALID_NETCFG, "split layer param failed");
            }

            // 3. Create Layer interpreter
            auto layer_interpreter = layer_interpreter_map[type_str];
            if (layer_interpreter == NULL) {
                LOGET("layer %s not supported\n", "ncnn", type_str.c_str());
                return Status(RPDERR_INVALID_NETCFG, "nill interpreter");
            }

            // 4. Interpreter layer
            LayerParam *param = NULL;
            ret               = layer_interpreter->InterpretProto(type_str, param_dict, cur_layer->type, &param);
            if (ret != RPD_OK) {
                return ret;
            }

            // 5. check Type
            if (cur_layer->type == LAYER_NOT_SUPPORT) {
                LOGET("layer %s interprete failed\n", "ncnn", type_str.c_str());
                return Status(RPDERR_INVALID_NETCFG, "interpreter failed");
            }

            if (!param) {
                param = new LayerParam();
            }

            // name
            if (param && layer_cfg_arr.size() >= 2) {
                param->name = layer_cfg_arr[1];
            }

            cur_layer->param = shared_ptr<LayerParam>(param);

            structure->layers.push_back(cur_layer);
        }

        return RPD_OK;
    }

    Status NCNNModelInterpreter::InterpretModel(std::string model_content) {
        auto &layer_interpreter_map = GetLayerInterpreterMap();

        NetResource *net_resource = GetNetResource();
        NetStructure *structure   = GetNetStructure();

        const auto model_length = model_content.length();
        if (model_length <= 0) {
#ifdef BENCHMARK
            return RPD_OK;
#else
            return Status(RPDERR_LOAD_MODEL, "model content is invalid");
#endif
        }

        std::istringstream content_stream;
        content_stream.str(model_content);

        Deserializer deserializer(content_stream);

        for (auto layer : structure->layers) {
            auto type_str = layer->type_str;

            // 0. Create Layer interpreter
            auto layer_interpreter = layer_interpreter_map[type_str];
            if (layer_interpreter == NULL) {
                LOGET("layer %s not supported\n", "ncnn", type_str.c_str());
                return Status(RPDERR_INVALID_NETCFG, "nill interpreter");
            }

            // 1. Interpreter layer
            LayerResource *layer_resource = NULL;

            Status result = layer_interpreter->InterpretResource(deserializer, layer, &layer_resource);
            if (result != RPD_OK) {
                LOGDT("ncnn model %s interpreter failed\n", "ncnn", type_str.c_str());
                return result;
            }

            // 1. Interpreter layer
            net_resource->resource_map[layer->name] = std::shared_ptr<LayerResource>(layer_resource);
        }

        return RPD_OK;
    }

    Status NCNNModelInterpreter::RegisterLayerInterpreter(std::string type_name,
                                                          AbstractLayerInterpreter *interpreter) {
        auto &layer_interpreter_map      = GetLayerInterpreterMap();
        layer_interpreter_map[type_name] = std::shared_ptr<AbstractLayerInterpreter>(interpreter);
        return RPD_OK;
    }

    std::map<std::string, std::shared_ptr<AbstractLayerInterpreter>> &NCNNModelInterpreter::GetLayerInterpreterMap() {
        static std::map<std::string, std::shared_ptr<AbstractLayerInterpreter>> layer_interpreter_map;
        return layer_interpreter_map;
    }

}  // namespace ncnn

}  // namespace TNN_NS
