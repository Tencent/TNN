#include "tnn/network/torch/torch_tnn_runtime.h"
#include "torch/custom_class.h"
#include "tnn/core/tnn.h"
#include "tnn/interpreter/tnn/model_interpreter.h"

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

    model_config.params.emplace_back(serialize[PROTO_IDX]);
    model_config.params.emplace_back(serialize[MODEL_IDX]);

    input_names = Deserialize(serialize[INPUT_NAME_IDX]);
    output_names = Deserialize(serialize[OUTPUT_NAME_IDX]);

    auto get_input_shape = [&](const std::string shape_str) {
        InputShapesMap input_shape;
        std::vector<std::string> input_shapes_vec = Deserialize(shape_str);
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

    InputShapesMap min_input_shape = get_input_shape(serialize[MIN_SHAPE_IDX]);
    InputShapesMap max_input_shape = get_input_shape(serialize[MAX_SHAPE_IDX]);

    auto config_vec = Deserialize(serialize[CONFIG_IDX]);
    network_config.device_type = static_cast<DeviceType>(std::stoi(config_vec[0]));
    network_config.device_id = std::stoi(config_vec[1]);
    network_config.precision = static_cast<Precision>(std::stoi(config_vec[2]));
    network_config.share_memory_mode = static_cast<ShareMemoryMode>(std::stoi(config_vec[3]));

    auto interpreter = CreateModelInterpreter(MODEL_TYPE_TNN);
    auto interpreter_ptr = std::shared_ptr<AbstractModelInterpreter>(interpreter);
    interpreter_ptr->Interpret(model_config.params);
    dynamic_cast<ModelInterpreter *>(interpreter_ptr.get())->SetCache(serialize[CACHE_IDX]);
    model_config.params.clear();

    instance_ = std::make_shared<Instance>(network_config, model_config);

    if (min_input_shape.size() == 0 && max_input_shape.size() == 0) {
        ctx_->set_interpreter(interpreter_ptr);
    } else {
        dynamic_cast<ModelInterpreter *>(interpreter)->GetNetStructure()->inputs_shape_map = max_input_shape;
        instance_->Init(interpreter_ptr, min_input_shape, max_input_shape);
        is_init_ = true;
    }
    network_config_ = network_config;

}

TNNEngine& TNNEngine::operator=(const TNNEngine& other) {
    instance_ = other.instance_;
    return (*this);
}

}  // namespace runtime
}  // namespace TNN_NS