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
    bool only_check_output = false;
    bool check_batch       = false;
    std::pair<std::string, FileFormat> ref_file;
    std::string dump_dir_path;
    std::string dump_output_path;
    std::string dump_unaligned_layer_path;
};

enum CompareType { DEFAULT = 0, COSINE = 1 };

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
    Status Init(NetworkConfig& net_config, ModelConfig& model_config);

    // @brief set model checker param
    // @param params the params of model checker
    Status SetModelCheckerParams(ModelCheckerParam params);

    // @brief run model checker
    Status RunModelChecker();

private:
    // @brief change batch size to check multi batch
    Status ChangeBatchOfInputShapes(InputShapesMap& input_shapes);

    // @brief compare all blobs data between device and cpu
    Status CompareDeviceAndCpu();
    // @brief per channel compare
    Status RunModelCheckerPerLayer();

    // @brief just compare output
    Status RunModelCheckerOutput();

    // @brief per layer compare dump file
    Status RunModelCheckerFromDumpFile();

    // @brief judge whether src_dims can be extended to dst_dims
    bool IsDimsCanBeExtend(std::vector<int> src_dims, std::vector<int> dst_dims);
    // @brief extend mat map due to multi batch
    Status ExtendMatMap(const BlobMap& blobs_map, std::map<std::string, std::shared_ptr<Mat>>& mat_map);
    // @brief feed input data of instance
    Status FeedInputData();

    // @brief get output mat map form instance
    Status GetOutputData(Instance* instance, std::map<std::string, std::shared_ptr<Mat>>& output_map);
    // @brief get output mat map form file
    Status GetOutputRefData();
    // @brief convert blob data to nchw float data
    Status GetBlobData(Instance* instance, Blob* blob, std::map<std::string, std::shared_ptr<char>>& output_map);
    // @brief get all blobs data
    Status GetCpuBlobData();

    // @brief compare raw
    bool CompareData(void* device_data, void* cpu_data, DataType data_type, DimsVector blob_dims, CompareType type = DEFAULT);
    // @brief dump blob data
    void DumpBlobData(void* blob_data, DimsVector blob_dims, std::string output_name, DataType data_type);

    ModelCheckerParam model_checker_params_;
    std::shared_ptr<TNN> tnn_                  = nullptr;
    std::shared_ptr<Instance> instance_cpu_    = nullptr;
    std::shared_ptr<Instance> instance_device_ = nullptr;
    std::map<std::string, std::shared_ptr<Mat>> output_ref_mat_map_;
    std::map<std::string, std::shared_ptr<char>> cpu_blobdata_map;
    std::vector<std::pair<LayerInfo*, bool>> check_results;
};

}  // namespace TNN_NS

#endif  // TNN_TOOLS_MODEL_CHECK_MODEL_CHECKER_H_
