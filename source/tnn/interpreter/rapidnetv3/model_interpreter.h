// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_TNN_INTERPRETER_RAPIDNETV3_RAPIDNETV3_MODEL_INTERPRETER_H_
#define TNN_SOURCE_TNN_INTERPRETER_RAPIDNETV3_RAPIDNETV3_MODEL_INTERPRETER_H_

#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/interpreter/tnn/model_interpreter.h"
#include "tnn/interpreter/rapidnetv3/objseri.h"


namespace rapidnetv3 {

    class AbstractLayerInterpreter;

    static const int layer_cfg_start_id    = 5;
    static const int layer_param_start_id  = 4;
    static const int input_layer_cfg_count = 5;

    // refactor later
    struct res_header : public TNN_NS::res_header {
    };

    struct layer_header : public TNN_NS::layer_header {
    };

    // @brief ModelInterpreter used to interpreter raidnet v1 model
    class ModelInterpreter : public TNN_NS::ModelInterpreter {
    public:
        // @brief rapidnet v1 model interpreter load params is proto contents,
        // model path.
        virtual Status Interpret(std::vector<std::string> &params);

        // @brief get model version
        ModelVersion GetModelVersion();

        static Status RegisterLayerInterpreter(
            LayerType type, TNN_NS::AbstractLayerInterpreter* creator);

        // @brief get layer interpreter by layer type
        static std::map<LayerType, std::shared_ptr<TNN_NS::AbstractLayerInterpreter>>&
        GetLayerInterpreterMap();

    protected:
        virtual Status InterpretProto(std::string &content);
        virtual Status InterpretModel(std::string &model_content);
        virtual Status InterpretInput(const std::string& inputs_content);
        virtual Status InterpretOutput(const std::string& outputs_content);
        virtual Status InterpretLayer(const std::string& layer_str);
        
    protected:
        virtual std::string Transfer(std::string content);
        virtual bool IsValidVersionNumber(uint32_t number);
        virtual std::shared_ptr<TNN_NS::Deserializer> GetDeserializer(std::istream &is);

    protected:
        ModelVersion model_version_;
    };

}  // namespace rapidnetv3



#endif  // TNN_SOURCE_TNN_INTERPRETER_RAPIDNETV3_RAPIDNETV3_MODEL_INTERPRETER_H_
