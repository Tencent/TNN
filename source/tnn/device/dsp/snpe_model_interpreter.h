// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_DSP_SNPE_MODEL_INTERPRETER_H_
#define TNN_SOURCE_DEVICE_DSP_SNPE_MODEL_INTERPRETER_H_

#include <memory>
#include <vector>
#include "DlContainer/IDlContainer.hpp"
#include "core/status.h"
#include "interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

// @brief SNPE model interpreter interpret SNPE model
class SnpeModelInterpreter : public AbstractModelInterpreter {
public:
    SnpeModelInterpreter();

    // @brief virtual destructor
    virtual ~SnpeModelInterpreter();

    // @brief different interpreter has different order param
    virtual Status Interpret(std::vector<std::string> params);

    std::unique_ptr<zdl::DlContainer::IDlContainer>& GetContainer();

private:
    std::unique_ptr<zdl::DlContainer::IDlContainer> container_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_DSP_SNPE_MODEL_INTERPRETER_H_
