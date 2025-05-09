// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TNN_SOURCE_DEVICE_ATLAS_ATLAS_NETWORK_H_
#define TNN_SOURCE_DEVICE_ATLAS_ATLAS_NETWORK_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "acl/acl.h"
#include "tnn/core/default_network.h"
#include "tnn/core/macro.h"
#include "tnn/device/atlas/atlas_common_types.h"
#include "tnn/device/atlas/atlas_context.h"

namespace TNN_NS {

class AtlasNetwork : public DefaultNetwork {
public:
    // @brief virtual default destructor
    virtual ~AtlasNetwork();

    // @brief init network with net cfg and net res.
    // @param net_cfg
    // @param net_res
    virtual Status Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                        InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape, InputDataTypeMap inputs_data_type,
                        bool enable_const_folder = true);

    //  @brief return the amount of memory required for forward
    //  @param memory_size: the memory size used by tnn layers for
    //  forward
    //  @return error code: If successful, returns zero. Otherwise, returns
    //  an error code.
    virtual Status GetForwardMemorySize(size_t &memory_size);

    //  @brief: set memory used by the tnn instance without forward
    //  memory, the memory size must be at least that returned by
    //  GetForwardMemorySize(). releasing or otherwise using the memory for
    //  other purposes during the tnn network run will result in
    //  undefined behavior.
    //  @param memory: the memory used by tnn layers for forward
    //  @return error code: If successful, returns zero. Otherwise, returns
    //  an error code.
    //
    virtual Status SetForwardMemory(void *memory);

    // @brief reshape network
    virtual Status Reshape(const InputShapesMap &inputs);

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    virtual Status GetCommandQueue(void **command_queue);

    // @brief set tnn command queue
    virtual Status SetCommandQueue(void *command_queue);

    // @brief network infer, it will sync to wait result
    virtual Status Forward();

    // @brief tnn instance network infer, it will not wait
    virtual Status ForwardAsync(Callback call_back);

    // @brief get all input blobs
    // @param blobs input blobs name map
    virtual Status GetAllInputBlobs(BlobMap &blobs);

    // @brief get all output blobs
    // @param blobs output blobs name map
    virtual Status GetAllOutputBlobs(BlobMap &blobs);

    // @brief get OM info of ATLAS OM model
    std::shared_ptr<AtlasOMModelInfo> GetOMModelInfo();

private:
    // OM RELATED
    
    // @brief load model from om file
    Status LoadOMModelFromFile(const std::string &om_file);

    // @brief load model from memory
    Status LoadOMModelFromMemory(const std::string &om_content);

    // @brief deduce model dynamic input mode
    Status DeduceOMModelDynamicMode();
    
    // @brief deduce model AIPP input format
    Status DeduceOMModelAIPPInputFormat();

    // @brief internal init network for OM model
    Status InitOMModel(ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                       InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape,
                       InputDataTypeMap inputs_data_type, bool enable_const_folder);

    // @brief internal reshape network for OM model
    virtual Status ReshapeOMModel(const InputShapesMap &inputs);

    // @brief get input dims
    Status GetInputInfo(size_t index, std::vector<int> &input_dims, aclFormat &input_format,
                        aclDataType &input_data_type);
    
    // @brief set dynamic input dims for OM models converted with --input_shape_range
    Status SetRangeDynamicInputDim(std::string input_name, const DimsVector& target_input_shape);

    // @brief update dynamic output dims for OM models converted with --input_shape_range
    Status UpdateRangeDynamicOutputDims();

    // @brief set dynmaic batch size
    Status SetDynamicBatchSize(std::string blob_name, int batch_size);

    std::map<std::string, int> output_dim0_map_;
    void* om_model_memory_ptr_                        = nullptr;
    void* om_model_weight_ptr_                        = nullptr;
    std::shared_ptr<AtlasOMModelInfo> om_model_info_  = nullptr;



    // @brief add blob into map
    Status AddBlobToMap(const InputShapesMap &max_input_shapes_map, size_t index, void *data, bool is_input);

    // @brief allocate data set and create Blob
    Status AllocateDatasetCreateBlob(aclmdlDataset **data_set, const InputShapesMap &max_input_shapes_map,
                                     bool is_input);

    // @brief destory dataset
    void DestroyDataset(aclmdlDataset *&data_set);

    ModelType model_type_;
    BlobMap input_blob_map_;
    BlobMap output_blob_map_;

    bool network_init_called_                         = false;
    aclmdlDataset* aclmdl_input_dataset_              = nullptr;
    aclmdlDataset* aclmdl_output_dataset_             = nullptr;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_ATLAS_ATLAS_NETWORK_H_
