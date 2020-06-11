// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_ATLAS_ENGINES_INFERENCE_ENGINE_INFERENCE_ENGINE_H_
#define TNN_SOURCE_DEVICE_ATLAS_ENGINES_INFERENCE_ENGINE_INFERENCE_ENGINE_H_

#include <utility>
#include "atlas_common_types.h"
#include "dvpp/idvppapi.h"
#include "hiaiengine/ai_model_manager.h"
#include "hiaiengine/engine.h"
#include "tnn/core/macro.h"

#define DT_INPUT_SIZE 1
#define DT_OUTPUT_SIZE 1

namespace TNN_NS {

class InferenceEngine : public hiai::Engine {
public:
    InferenceEngine()
        : use_dynamic_aipp_(false),
          aipp_swap_rb_(false),
          aipp_normalize_(false) {}

    HIAI_StatusT Init(const hiai::AIConfig& config,
                      const std::vector<hiai::AIModelDescription>& model_desc);

    HIAI_DEFINE_PROCESS(DT_INPUT_SIZE, DT_OUTPUT_SIZE)

private:
    int ParseConfig(const hiai::AIConfig& config,
                    hiai::AIModelDescription& model_desc);
    HIAI_StatusT SetDynamicAipp();
    int SendOutputData(std::shared_ptr<hiai::IAITensor>& output_tensor,
                       std::shared_ptr<TransferDataType>& input_trans_data);
    int SendQueryDimInfo(hiai::TensorDimension& dim_info, QueryType qt);
    int SendTransDataEnd(CommandType ct, QueryType qt);

    bool use_dynamic_aipp_;
    bool aipp_swap_rb_;
    bool aipp_normalize_;
    shared_ptr<hiai::AippDynamicParaTensor> aipp_params_;
    std::shared_ptr<hiai::AIModelManager> model_manager_;
    std::vector<hiai::TensorDimension> input_tensor_dims_;
    std::vector<hiai::TensorDimension> output_tensor_dims_;
    std::vector<std::shared_ptr<uint8_t>> input_data_buffer_;
    std::vector<std::shared_ptr<uint8_t>> output_data_buffer_;
    std::vector<std::shared_ptr<hiai::IAITensor>> input_tensor_vec_;
    std::vector<std::shared_ptr<hiai::IAITensor>> output_tensor_vec_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ENGINES_INFERENCE_ENGINE_INFERENCE_ENGINE_H_
