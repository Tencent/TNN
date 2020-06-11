// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ENGINES_OUTPUT_ENGINE_OUTPUT_ENGINE_H_
#define TNN_SOURCE_DEVICE_ATLAS_ENGINES_OUTPUT_ENGINE_OUTPUT_ENGINE_H_

#include <map>
#include <utility>
#include "atlas_common_types.h"
#include "hiaiengine/engine.h"
#include "tnn/core/macro.h"

#define DT_INPUT_SIZE 1
#define DT_OUTPUT_SIZE 1

namespace TNN_NS {

class OutputEngine : public hiai::Engine {
public:
    OutputEngine() {}

    HIAI_StatusT Init(const hiai::AIConfig& config,
                      const std::vector<hiai::AIModelDescription>& model_desc);

    HIAI_DEFINE_PROCESS(DT_INPUT_SIZE, DT_OUTPUT_SIZE)

private:
    int output_count_ = 0;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ENGINES_OUTPUT_ENGINE_OUTPUT_ENGINE_H_
