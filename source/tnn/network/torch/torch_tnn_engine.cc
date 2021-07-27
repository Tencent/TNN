#include "tnn/network/torch/torch_tnn_runtime.h"
#include "torch/custom_class.h"

namespace TNN_NS {
namespace runtime {

void TorchConvertCtx::buildOp(const torch::jit::Node *node) {

}

bool TorchConvertCtx::dealPrime(const torch::jit::Node *node) {
    std::string opType = node->kind().toUnqualString();
    switch (node->kind()) {
        case at::prim::Constant:
        case at::prim::ListConstruct:
        case at::prim::ListUnpack:
            for (const auto output : node->outputs()) {
                declareVar(output->debugName(), node);
            }
            return true;
        default:
            break;
    }
    if (opType == "If") {
        if (!node->outputs().empty()) {
            return false;
        }
        return true;
    }
    if (opType == "Loop") {
        return true;
    }
    return true;
}

int TorchConvertCtx::declareTensor(std::string name) {
    if (tensorIdx.count(name)) {
        return tensorIdx[name];
    }
    int idx = tensorIdx.size();
    tensorIdx[name] = idx;
    return idx;
}

int TorchConvertCtx::lookupTensor(std::string name) {
    const auto iter = tensorIdx.find(name);
    if (iter != tensorIdx.end()) {
        return iter->second;
    }
    const auto iterVar = varTable.find(name);
    if (iterVar != varTable.end()) {
        buildOp(iterVar->second);
        return lookupTensor(name);
    }
    return -1;
}

std::string TorchConvertCtx::lookupTensor(int idx) const {
    return "NaN";
}

void TorchConvertCtx::declareVar(std::string name, const torch::jit::Node* var) {
    if (varTable.count(name)) {
        return;
    }
    varTable[name] = var;
}

const torch::jit::Node* TorchConvertCtx::lookupVar(std::string name) const {
    const auto iter = varTable.find(name);
    if (iter != varTable.end()) {
        return iter->second;
    }
    return nullptr;
}

TNNEngine::TNNEngine(NetworkConfig& network_config, ModelConfig& model_config) {
    instance_ = std::make_shared<Instance>(network_config, model_config);
}
TNNEngine::TNNEngine(std::shared_ptr<Instance> &instance) {
    instance_ = instance;
}

TNNEngine& TNNEngine::operator=(const TNNEngine& other) {
    instance_ = other.instance_;
    return (*this);
}

}  // namespace runtime
}  // namespace TNN_NS