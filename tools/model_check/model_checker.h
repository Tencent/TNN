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

#ifndef TNN_TOOLS_MODEL_CHECK_MODEL_CHECKER_H_
#define TNN_TOOLS_MODEL_CHECK_MODEL_CHECKER_H_

#include <memory>
#include "file_reader.h"
#include "tnn/core/blob.h"
#include "tnn/core/instance.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/status.h"
#include "tnn/core/tnn.h"

namespace TNN_NS {

struct ModelCheckerParam {
    std::pair<std::string, FileFormat> input_file;
    std::vector<float> input_bias;
    std::vector<float> input_scale;
    bool dump_output;
    std::pair<std::string, FileFormat> ref_file;
};

class ModelChecker {
public:
    // @brief ModelChecker Constructor
    ModelChecker();

    // @brief ModelChecker virtual Destructor
    virtual ~ModelChecker();

public:
    // @brief int net with network config, net structure and net resource info
    // @param config network config info
    // @param inputs_shape_map modify input shape, if empty, it will use the
    // shape in proto
    Status Init(NetworkConfig& net_config, ModelConfig& model_config,
                InputShapesMap inputs_shape = InputShapesMap());

    // @brief set model checker param
    // @param params the params of model checker
    int SetModelCheckerParams(ModelCheckerParam params);

    // @brief run model checker
    Status RunModelChecker();

private:
    Status FeedInputData();
    Status GetOutputRefData();
    Status GetCpuBlobData();
    Status CompareDeviceAndCpu();
    bool CompareData(void* device_data, void* cpu_data, DimsVector blob_dims);
    void DumpBlobData(void* blob_data, DimsVector blob_dims,
                      std::string output_name);

    ModelCheckerParam model_checker_params_;
    std::shared_ptr<TNN> tnn_;
    std::shared_ptr<Instance> instance_cpu_;
    std::shared_ptr<Instance> instance_device_;
    std::map<std::string, std::shared_ptr<float>> output_ref_data_map_;
    std::map<std::string, std::shared_ptr<char>> cpu_blobdata_map;
    std::vector<std::pair<LayerInfo*, bool>> check_results;
};

}  // namespace TNN_NS

#endif  // TNN_TOOLS_MODEL_CHECK_MODEL_CHECKER_H_
