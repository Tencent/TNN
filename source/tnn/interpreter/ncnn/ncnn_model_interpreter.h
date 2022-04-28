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

#ifndef TNN_SOURCE_TNN_INTERPRETER_NCNN_NCNN_MODEL_INTERPRETER_H_
#define TNN_SOURCE_TNN_INTERPRETER_NCNN_NCNN_MODEL_INTERPRETER_H_

#include <memory>
#include <vector>
#include <algorithm>
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/utils/safe_map.h"

namespace TNN_NS {

namespace ncnn {

    class AbstractLayerInterpreter;

    static const int layer_cfg_start_id   = 2;
    static const int layer_param_start_id = 4;
    static const int ncnn_magic_number    = 7767517;

    // @brief NCNNModelInterpreter used to interpreter ncnn model
    class NCNNModelInterpreter : public DefaultModelInterpreter {
    public:
        // @brief ncnn model interpreter load params is param content and bin
        virtual Status Interpret(std::vector<std::string> &params);

        static Status RegisterLayerInterpreter(std::string type_name, AbstractLayerInterpreter* creator);

        // @brief get layer interpreter by layer type
        static const safe_map<std::string, std::shared_ptr<AbstractLayerInterpreter>>& GetLayerInterpreterMap();

    private:
        Status InterpretProto(std::string &content);
        Status InterpretModel(std::string &model_content);
        Status InterpretInput();
        Status AppendCommonLayer(
            str_arr& layer_cfg_arr, NetStructure *structure,
            const safe_map<std::string, std::shared_ptr<AbstractLayerInterpreter>> &layer_interpreter_map);

        Status FindOutputs();
        Status Convert(shared_ptr<LayerInfo> cur_layer, std::vector<std::shared_ptr<LayerInfo>> output_layers);

        // @brief get layer interpreter by layer type
        static safe_map<std::string, std::shared_ptr<AbstractLayerInterpreter>>& LayerInterpreterMap();
    };

}  // namespace ncnn

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_NCNN_NCNN_MODEL_INTERPRETER_H_
