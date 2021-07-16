#pragma once

#include <vector>

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

namespace TNN_NS {
namespace conversion {

class TorchConvertCtx {
public:
    TorchConvertCtx() {
        auto interpreter = CreateModelInterpreter(MODEL_TYPE_TNN);
        interpreter_ = std::shared_ptr<AbstractModelInterpreter>(interpreter);
    };
    void buildOp(const torch::jit::Node* node);
    bool dealPrime(const torch::jit::Node* node);
    int declareTensor(std::string name);
    int lookupTensor(std::string name);
    std::string lookupTensor(int idx) const;
    void declareVar(std::string name, const torch::jit::Node* var);
    const torch::jit::Node* lookupVar(std::string name) const;
    std::shared_ptr<AbstractModelInterpreter> get_interpreter() {
        return interpreter_;
    };
private:
    std::map<std::string, const torch::jit::Node*> varTable;
    std::map<std::string, int> tensorIdx;
    std::shared_ptr<AbstractModelInterpreter> interpreter_;
};

struct TNNEngine : public torch::CustomClassHolder {
    TNNEngine(NetworkConfig &network_config, ModelConfig &model_config) {
        instance_ = std::make_shared<Instance>(network_config, model_config);
    }
    std::shared_ptr<Instance> instance_;
};

c10::intrusive_ptr<TNNEngine> ConvertBlockToInstance(partitioning::SegmentedBlock &block, TorchConvertCtx *ctx); 

}
}