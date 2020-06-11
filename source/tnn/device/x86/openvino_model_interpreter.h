// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_X86_OPENVINO_MODEL_INTERPRETER_H_
#define TNN_SOURCE_DEVICE_X86_OPENVINO_MODEL_INTERPRETER_H_

#include <vector>
#include "tnn/core/status.h"
#include "tnn/interpreter/abstract_model_interpreter.h"

#include <inference_engine.hpp>

namespace TNN_NS {

// @brief OpenVINO model interpreter interpret openvino model
class OpenVINOModelInterpreter:public AbstractModelInterpreter {
public:

    OpenVINOModelInterpreter();

    // @brief virtual destructor
    virtual ~OpenVINOModelInterpreter();

    // @brief different interpreter has different order param
    virtual Status Interpret(std::vector<std::string> params);

    InferenceEngine::CNNNetwork GetCNNNetwork();

private:
    InferenceEngine::CNNNetwork network_;
    InferenceEngine::Core ie_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_X86_OPENVINO_MODEL_INTERPRETER_H_
