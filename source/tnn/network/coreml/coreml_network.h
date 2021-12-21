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

#ifndef TNN_SOURCE_TNN_DEVICE_COREML_COREML_NETWORK_H_
#define TNN_SOURCE_TNN_DEVICE_COREML_COREML_NETWORK_H_

#import <CoreML/CoreML.h>

#include "tnn/core/abstract_device.h"
#include "tnn/core/abstract_network.h"
#include "tnn/core/context.h"
#include "tnn/core/default_network.h"
#include "mlmodel/include/Model.pb-c.h"
#include "tnn/network/coreml/layer_convert/coreml_base_layer.h"
#import "coreml_model.h"

namespace TNN_NS {

class CoreMLNetwork : public AbstractNetwork {
public:
    // @brief CoreMLNetwork Constructor
    CoreMLNetwork();

    // @brief CoreMLNetwork virtual Destructor
    virtual ~CoreMLNetwork();

public:
    // @brief int net with network config, net structure and net resource info
    // @param config network config info
    // @param net_structure network structure info
    // @param net_resource network resource info
    // @param inputs_shape_map modify input shape, if empty, it will use the
    // shape in proto
    virtual Status Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
        InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape, bool enable_const_folder=true);

    // @brief reshape with input shape info
    // @inputs input shape info
    virtual Status Reshape(const InputShapesMap &inputs);

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    virtual Status GetCommandQueue(void **command_queue);

    // @brief network forward
    virtual Status Forward();

    // @brief tnn instance network infer, it will not wait
    virtual Status ForwardAsync(Callback call_back);

    // @brief network deinit to release init create resource
    virtual Status DeInit();

    // @brief get network forward for all blob memory size
    virtual Status GetForwardMemorySize(int &memory_size);

    // @brief set forward memory when share memory mode is set from external
    virtual Status SetForwardMemory(void *memory);

    // @brief get all input blobs
    virtual Status GetAllInputBlobs(BlobMap &blobs);

    // @brief get all output blobs
    virtual Status GetAllOutputBlobs(BlobMap &blobs);
            
    Status SetInput(CoreML__Specification__FeatureDescription** describe, std::string name, std::vector<int> shape, DataType type);
    
    Status SetOutput(CoreML__Specification__FeatureDescription** describe, std::string name, std::vector<int> shape, DataType type);
        
    Status InitCoreMLModel(NetStructure *net_structure, NetResource *net_resource);
    
    Status ConvertCoreMLModel(NetStructure *net_structure, NetResource *net_resource);
        
    Status CompileModel(CoreML__Specification__Model* model);
    
protected:
    ModelConfig model_config_;
    AbstractDevice *device_              = nullptr;
    Context *context_                       = nullptr;
    NetStructure *net_structure_      = nullptr;
    NetResource *net_resource_     = nullptr;
    
    BlobMap blob_input_map_;
    BlobMap blob_output_map_;
    
    std::vector<std::shared_ptr<Blob> > blob_input_ = {};
    std::vector<std::shared_ptr<Blob> > blob_output_ = {};
    
    Status CheckCoreMLStatus();
    
    std::unique_ptr<_CoreML__Specification__Model> coreml_model_;
    std::vector<std::shared_ptr<CoreMLBaseLayer> > coreml_layers_;
    std::vector<CoreML__Specification__NeuralNetworkLayer*> coreml_layer_ptrs_;
    
    std::shared_ptr<void> coreml_neural_network_;
    std::shared_ptr<void> coreml_model_description_;
    
    std::shared_ptr<CoreML__Specification__FeatureDescription *> coreml_input_arr_;
    std::vector<std::shared_ptr<CoreML__Specification__FeatureDescription> > coreml_input_feature_description_;
    std::vector<std::shared_ptr<CoreML__Specification__FeatureType> > coreml_input_feature_type_;
    std::vector<std::shared_ptr<CoreML__Specification__ArrayFeatureType> > coreml_input_array_feature_type_;
    std::vector<std::shared_ptr<int64_t> > coreml_input_shape_;
    std::vector<std::shared_ptr<char> > input_name_;
    
    std::shared_ptr<CoreML__Specification__FeatureDescription *> coreml_output_arr_;
    std::vector<std::shared_ptr<CoreML__Specification__FeatureDescription> > coreml_output_feature_description_;
    std::vector<std::shared_ptr<CoreML__Specification__FeatureType> > coreml_output_feature_type_;
    std::vector<std::shared_ptr<CoreML__Specification__ArrayFeatureType> > coreml_output_array_feature_type_;
    std::vector<std::shared_ptr<int64_t> >coreml_output_shape_;
    std::vector<std::shared_ptr<char> > output_name_;
    
    CoreMLModel* coreml_executor_ = nil;
    
};

} // namespace TNN_NS

#endif // TNN_SOURCE_TNN_DEVICE_COREML_COREML_NETWORK_H_
