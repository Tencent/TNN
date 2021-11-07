#include "tnn/network/torch/torch_tnn_runtime.h"
#include "torch/custom_class.h"
#include "tnn/core/tnn.h"

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

TNNEngine::TNNEngine(std::vector<std::string> &serialize) {
    NetworkConfig network_config;
    ModelConfig model_config;
    Status status;

    model_config.params.emplace_back(serialize[0]);
    model_config.params.emplace_back(serialize[1]);

    input_names = Deserialize(serialize[2]);
    output_names = Deserialize(serialize[3]);

    auto get_input_shape = [&](const std::string shape_str) {
        InputShapesMap input_shape;
        std::vector<std::string> input_shapes_vec = Deserialize(serialize[4]);
        for (size_t i = 0; i < input_names.size(); i++) {
            DimsVector dims;
            auto input_shape_s = Deserialize(input_shapes_vec[i], TORCH_INT_DELIM);
            for (auto &iter : input_shape_s) {
                dims.emplace_back(std::stoi(iter));
            }
            input_shape[input_names[i]] = dims;
        }
        return input_shape;
    };

    InputShapesMap min_input_shape = get_input_shape(serialize[4]);
    InputShapesMap max_input_shape = get_input_shape(serialize[5]);

    auto config_vec = Deserialize(serialize[6]);
    network_config.device_type = static_cast<DeviceType>(std::stoi(config_vec[0]));
    network_config.device_id = std::stoi(config_vec[1]);
    network_config.precision = static_cast<Precision>(std::stoi(config_vec[2]));
    network_config.share_memory_mode = static_cast<ShareMemoryMode>(std::stoi(config_vec[3]));

    TNN net;
    net.Init(model_config);
    instance_ = net.CreateInst(network_config, status, min_input_shape, max_input_shape);

    is_init_ = true;

}

TNNEngine& TNNEngine::operator=(const TNNEngine& other) {
    instance_ = other.instance_;
    return (*this);
}

}  // namespace runtime
}  // namespace TNN_NS