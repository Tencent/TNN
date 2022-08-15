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

#ifndef TNN_SOURCE_TNN_INTERPRETER_TNN_TNN_MODEL_INTERPRETER_H_
#define TNN_SOURCE_TNN_INTERPRETER_TNN_TNN_MODEL_INTERPRETER_H_

#include <algorithm>
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/interpreter/tnn/objseri.h"
#include "tnn/utils/safe_map.h"

using namespace TNN_NS;
namespace TNN_NS {

class AbstractLayerInterpreter;

static const int layer_cfg_start_id    = 5;
static const int layer_param_start_id  = 4;
static const int input_layer_cfg_count = 2;

// refactor later
struct res_header : public Serializable {
    int layer_cnt_;

    res_header() : layer_cnt_(0) {}

public:
    virtual void serialize(Serializer& out) {
        out.PutInt(layer_cnt_);
    }

    virtual void deserialize(Deserializer& in) {
        layer_cnt_ = in.GetInt();
        layer_cnt_ = layer_cnt_ & 0x1FFFFFFF;
    }
};

struct layer_header : public Serializable {
public:
    layer_header() {
        type_ = LAYER_NOT_SUPPORT;
    }

public:
    virtual void serialize(Serializer& out) {
        out.PutInt((LayerType)type_);
        out.PutString(type_str_);
        out.PutString(name_);
    }
    virtual void deserialize(Deserializer& in) {
        int type_v = in.GetInt();
        if (type_v < 0 || type_v > 10000) {
            return;
        }
        type_ = (LayerType)type_v;

        std::string type_str    = in.GetString();
        LayerType type_from_str = GlobalConvertLayerType(type_str);
        if (LAYER_NOT_SUPPORT != type_from_str) {
            type_     = type_from_str;
            name_     = in.GetString();
            type_str_ = type_str;
        } else {
            name_     = type_str;
            type_str_ = "";
        }
    }

public:
    LayerType type_;
    std::string type_str_;
    std::string name_;
};

// @brief ModelInterpreter used to interpreter raidnet v1 model
class ModelInterpreter : public DefaultModelInterpreter {
public:
    // @brief copy constructor
    ModelInterpreter();

    // @brief copy constructor
    ModelInterpreter(const ModelInterpreter& interp);

    // @brief assign constructor
    ModelInterpreter& operator=(ModelInterpreter interp);

    // @brief model interpreter load params is proto contents,
    // model contents.
    virtual Status Interpret(std::vector<std::string>& params);

    // @brief interpret extra config, such as conv winograd for specific conv layer
    virtual Status InterpretConfig(std::map<std::string, std::string>& config_map);

    static Status RegisterLayerInterpreter(LayerType type, AbstractLayerInterpreter* creator);

    // @brief get layer interpreter by layer type
    static const safe_map<LayerType, std::shared_ptr<AbstractLayerInterpreter>>& GetLayerInterpreterMap();

    // @brief copy interpreter
    virtual std::shared_ptr<AbstractModelInterpreter> Copy();

private:
    // @brief get layer interpreter by layer type
    static safe_map<LayerType, std::shared_ptr<AbstractLayerInterpreter>>& LayerInterpreterMap();

protected:
    virtual Status InterpretProto(std::string& content);
    virtual Status InterpretModel(std::string& model_content);
    virtual Status InterpretInput(const std::string& inputs_content);
    virtual Status InterpretOutput(const std::string& outputs_content);
    virtual Status InterpretLayer(const std::string& layer_str);

protected:
    virtual std::string Transfer(std::string content);
    virtual bool IsValidVersionNumber(uint32_t number);
    virtual std::shared_ptr<Deserializer> GetDeserializer(std::istream& is);
    ;

protected:
    uint32_t version_magic_number = 0;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_TNN_TNN_MODEL_INTERPRETER_H_
