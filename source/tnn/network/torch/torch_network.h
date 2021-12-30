// Copyright 2021 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_NETWORK_TNNTORCH_TNNTORCH_NETWORK_H
#define TNN_SOURCE_NETWORK_TNNTORCH_TNNTORCH_NETWORK_H

#include <vector>

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

#include <c10/cuda/CUDAGuard.h>
#include <torch/script.h>

namespace TNN_NS {

class TNNTorchNetwork:public DefaultNetwork {
public:
    // @brief virtual default destructor
    virtual ~TNNTorchNetwork();

    // @brief init network with net cfg and net res.
    // @param net_cfg
    // @param net_res
    virtual Status Init(NetworkConfig &net_config, ModelConfig &model_config, AbstractModelInterpreter *interpreter,
                        InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape,
                        bool enable_const_folder = true);

    // @brief deinit release init create resource
    // virtual Status DeInit();

    // @brief network infer
    virtual Status Reshape(const InputShapesMap &inputs);

    // @brief get tnn command queue
    // @param command_queue device command queue for forward
    // virtual Status GetCommandQueue(void **command_queue);

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

    // @brief share torch network resource(module and graph) to another network
    // @param network to share resource
    virtual Status ShareNetResource(AbstractNetwork *network);

    // @brief get torch network module
    torch::jit::Module GetModule() { return module_; }

    // @brief get torch network graph
    std::shared_ptr<torch::jit::Graph> GetGraph() { return graph_; }
private:

    virtual Status CreateIOBinding(InputShapesMap  min_shape, InputShapesMap max_shape);
  
    virtual Status ClearOutputs();

    virtual Status ReleaseTorchOutputTensors();

    Status DumpAllOutputBlob();

    torch::jit::Module module_;
    std::shared_ptr<torch::jit::Graph> graph_;

    std::string forward_func_name_ = "forward";

    BlobMap input_blob_map_;
    BlobMap output_blob_map_;

    bool init_done_ = false;
    DataType precision_;

    InputShapesMap min_inputs_shape_;
    InputShapesMap max_inputs_shape_;

    std::vector<torch::IValue> in_ivalues_;
    c10::cuda::CUDAStream* cuda_stream_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_NETWORK_TNNTORCH_TNNTORCH_NETWORK_H
