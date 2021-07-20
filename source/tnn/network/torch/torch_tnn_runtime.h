#pragma once

#include "tnn/interpreter/abstract_model_interpreter.h"
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

#include "tnn/network/torch/SegmentedBlock.h"
#include "tnn/interpreter/default_model_interpreter.h"

#include <torch/script.h>
#include "c10/util/intrusive_ptr.h"
#include "torch/custom_class.h"
namespace TNN_NS{
namespace runtime {
    
struct TNNEngine : torch::CustomClassHolder {
    TNNEngine(NetworkConfig &network_config, ModelConfig &model_config);
    TNNEngine(std::shared_ptr<Instance> &instance);
    std::shared_ptr<Instance> instance_;

    TNNEngine& operator=(const TNNEngine& other); 
};

std::vector<at::Tensor> execute_engine(std::vector<at::Tensor> inputs, c10::intrusive_ptr<TNNEngine> compiled_engine);
}
}