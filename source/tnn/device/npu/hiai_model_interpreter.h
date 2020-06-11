// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_NPU_HIAI_MODEL_INTERPRETER_H_
#define TNN_SOURCE_DEVICE_NPU_HIAI_MODEL_INTERPRETER_H_

#include <memory>
#include <vector>
#include "HIAIMixModel.h"
#include "core/status.h"
#include "interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

// @brief SNPE model interpreter interpret SNPE model
class HiaiModelInterpreter : public AbstractModelInterpreter {
public:
    HiaiModelInterpreter();

    // @brief virtual destructor
    virtual ~HiaiModelInterpreter();

    // @brief different interpreter has different order param
    virtual Status Interpret(std::vector<std::string> params);

private:
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_NPU_HIAI_MODEL_INTERPRETER_H_
