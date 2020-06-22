// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_OPENVINO_OPENVINO_NETWORK_H_
#define TNN_SOURCE_DEVICE_OPENVINO_OPENVINO_NETWORK_H_

#include <vector>

#include <inference_engine.hpp>

#include "tnn/core/abstract_network.h"
#include "tnn/core/default_network.h"
#include "tnn/core/blob.h"
#include "tnn/core/blob_manager.h"
#include "tnn/core/common.h"
#include "tnn/core/context.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/net_resource.h"
#include "tnn/interpreter/net_structure.h"
#include "tnn/layer/base_layer.h"
#include "tnn/layer/base_layer.h"
#include "tnn/device/openvino/layer_builder/openvino_layer_builder.h"

namespace TNN_NS {

class OpenVINONetwork_:public DefaultNetwork {
public:
    // @brief virtual default destructor
    virtual ~OpenVINONetwork_();

    // @brief init network with net cfg and net res.
    // @param net_cfg
    // @param net_res
    virtual Status Init(NetworkConfig &net_config, ModelConfig &model_config,
                        AbstractModelInterpreter* interpreter,
                        InputShapesMap inputs_shape);

    // @brief deinit release init create resource
    virtual Status DeInit();

    //  @brief return the amount of memory required for forward
    //  @param memory_size: the memory size used by tnn layers for
    //  forward
    //  @return error code: If successful, returns zero. Otherwise, returns
    //  an error code.
    virtual Status GetForwardMemorySize(int &memory_size);

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

    // @brief network infer
    virtual Status Reshape(const InputShapesMap &inputs);

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    virtual Status GetCommandQueue(void **command_queue);

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

    //
    virtual Status SetNetInputNode(NetStructure *net_structure, NetResource *net_resource);

    virtual Status BuildNgraphNetwork(NetStructure *net_structure);

private:
    virtual Status InitLayers(NetStructure *net_structure, NetResource *net_resource);
  
    InferenceEngine::Core ie_;
    // InferenceEngine::CNNNetwork network_;
    std::shared_ptr<InferenceEngine::CNNNetwork> network_;
    InferenceEngine::ExecutableNetwork executable_network_;
    InferenceEngine::InferRequest infer_request_;
    BlobMap input_blob_map_;
    BlobMap output_blob_map_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_OPENVINO_OPENVINO_NETWORK_H_
