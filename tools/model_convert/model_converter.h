// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_TOOLS_MODEL_CONVERT_MODEL_CONVERTER_H_
#define TNN_TOOLS_MODEL_CONVERT_MODEL_CONVERTER_H_

#include <memory>
#include "tnn/core/blob.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/rapidnetv3/model_interpreter.h"
#include "tnn/interpreter/rapidnetv3/objseri.h"

namespace TNN_NS {

struct ModelInfo {
    rapidnetv3::ModelVersion model_version;
    DataType model_data_type;
};

class ModelConvertor {
public:
    // @brief ModelConvertor Constructor
    ModelConvertor();

    // @brief ModelConvertor virtual Destructor
    virtual ~ModelConvertor();

public:
    // @brief init with model config, net structure and net resource info
    // @param model_config model config info
    // shape in proto
    Status Init(ModelConfig& model_config);

    // @brief set model version to save
    // @param ver, model version
    void SetModelVersion(rapidnetv3::ModelVersion ver);

    // @brief dump model info
    void DumpModelInfo();

    // @brief int net with network config, net structure and net resource info
    // @param proto_path, file path to save the quantized proto.
    // @param model_path, file path to save the quantized model.
    Status Serialize(std::string proto_path, std::string model_path);

private:
    std::shared_ptr<rapidnetv3::ModelInterpreter> interpreter_;
    rapidnetv3::ModelVersion model_version_;
};

}  // namespace TNN_NS

#endif  // TNN_TOOLS_MODEL_CONVERT_MODEL_CONVERTER_H_
