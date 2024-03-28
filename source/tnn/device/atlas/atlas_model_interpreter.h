// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ATLAS_MODEL_INTERPRETER_H_
#define TNN_SOURCE_DEVICE_ATLAS_ATLAS_MODEL_INTERPRETER_H_

#include <memory>
#include <vector>
#include <map>
#include <mutex>
#include "atlas_common_types.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/abstract_model_interpreter.h"

namespace TNN_NS {

struct WeightPacket {
    void* weight_mem_ptr = nullptr;
    aclrtContext context = nullptr;
};

// @brief Atlas model interpreter interpret Atlas model
class AtlasModelInterpreter : public AbstractModelInterpreter {
public:
    AtlasModelInterpreter();

    // @brief virtual destructor
    virtual ~AtlasModelInterpreter();

    // @brief different interpreter has different order param
    virtual Status Interpret(std::vector<std::string> &params);

    // @brief get model config info
    AtlasModelConfig& GetModelConfig();

    // @brief get buffer ptr for model weights
    void* GetModelWeightsBufferPtr(int device_id);

    // @brief get buffer size for model weights
    size_t GetModelWeightsBufferSize();

private:
    AtlasModelConfig model_config_;
    std::map<int, WeightPacket> model_weight_map_;
    size_t model_weight_size_ = 0;
    std::mutex mutex_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ATLAS_MODEL_INTERPRETER_H_
