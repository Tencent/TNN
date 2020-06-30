// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ATLAS_MODEL_INTERPRETER_H_
#define TNN_SOURCE_DEVICE_ATLAS_ATLAS_MODEL_INTERPRETER_H_

#include <memory>
#include <vector>
#include "atlas_common_types.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

// @brief Atlas model interpreter interpret Atlas model
class AtlasModelInterpreter : public AbstractModelInterpreter {
public:
    AtlasModelInterpreter();

    // @brief virtual destructor
    virtual ~AtlasModelInterpreter();

    // @brief different interpreter has different order param
    virtual Status Interpret(std::vector<std::string> params);

    AtlasModelConfig& GetModelConfig();

private:
    AtlasModelConfig model_config_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ATLAS_MODEL_INTERPRETER_H_
