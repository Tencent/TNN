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

#ifndef TNN_TOOLS_QUANTIZATION_CALIBRATION_H_
#define TNN_TOOLS_QUANTIZATION_CALIBRATION_H_

#include <memory>
#include "tnn/core/blob.h"
#include "tnn/core/instance.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/default_model_interpreter.h"

#include "calibration_common.h"
#include "scale_calculator.h"

namespace TNN_NS {

class Calibration {
public:
    // @brief Calibration Constructor
    Calibration();

    // @brief Calibration virtual Destructor
    virtual ~Calibration();

public:
    // @brief int net with network config, net structure and net resource info
    // @param config network config info
    // @param inputs_shape_map modify input shape, if empty, it will use the
    // shape in proto
    Status Init(NetworkConfig& net_config, ModelConfig& model_config, InputShapesMap inputs_shape = InputShapesMap());

    // @brief set the quanztize method
    // @param params the params of calibration
    int SetCalibrationParams(CalibrationParam params);

    // @brief int net with network config, net structure and net resource info
    // @param dataset calibration inputs
    Status RunCalibration(DataSet& dataset);

    // @brief int net with network config, net structure and net resource info
    // @param proto_path, file path to save the quantized proto.
    // @param model_path, file path to save the quantized model.
    Status Serialize(std::string proto_path, std::string model_path);

private:
    int CalBlobScale(DataSet& dataset);
    int InitFeatureMap();
    int UpdateBlobRange(DataSet& dataset);
    int UpdateBlobDistribute(DataSet& dataset);
    IntScaleResource* CreateIntScale(std::vector<float> scale_vec);
    IntScaleResource* CreateIntScale(std::vector<float> scale_vec, std::vector<int8_t> zero_point_vec);

    int QuantizeParams();
    int QuantizeConvParams(ConvLayerResource* resource, ConvLayerParam* param, IntScaleResource* input_scale);
    int QuantizeFcParams(InnerProductLayerResource* resource, InnerProductLayerParam* param,
                         IntScaleResource* input_scale);
    // int CalQuantizedWeights(const float* weights, const int size, const int output_channel, bool merge_channel,
    //                         int8_t* quantized_weight, float* weight_scale);
    int CalQuantizedWeights(const float* weights, const int size, const int output_channel, bool merge_channel,
                            int8_t* quantized_weight, float* weight_scale,  int8_t* weight_zero_point);

    int MergeBlobScale();
    void MergeBlobScaleRecursion(LayerInfo* layer_info, NetStructure* net_struct, NetResource* net_resource);
    LayerInfo* GetLayerInfoFromOutpubBlobName(std::string blob_name, NetStructure* net_struct);

    std::shared_ptr<DefaultModelInterpreter> interpreter_;
    std::shared_ptr<Instance> instance_;
    std::map<Blob*, std::shared_ptr<ScaleCalculator>> feature_map_;
    CalibrationParam cali_params_;
};

}  // namespace TNN_NS

#endif  // TNN_TOOLS_QUANTIZATION_CALIBRATION_H_
