#include "tnn/network/torch/torch_convert.h"
#include "tnn/network/torch/torch_op_converter.h"

namespace TNN_NS {
namespace conversion {

using namespace partitioning;

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

void ConvertNodeToLayer(const torch::jit::Node *node, LayerInfo *layer_info, LayerResource **layer_res) {
    const auto& op_type = node->kind().toUnqualString();

    auto& converter = GetGlobalTorchConvertMap()[op_type];
    converter->Convert(node, layer_info, layer_res);
    
    // Todo need register convert table

}

c10::intrusive_ptr<TNNEngine> ConvertBlockToInstance(partitioning::SegmentedBlock &block, TorchConvertCtx *ctx) {
    ModelConfig model_config;
    NetworkConfig network_config;

    network_config.device_type = DEVICE_X86;
    auto instance_ptr = c10::make_intrusive<TNNEngine>(network_config, model_config);

    auto interpreter = dynamic_cast<DefaultModelInterpreter *>(ctx->get_interpreter().get());
    auto net_structure = interpreter->GetNetStructure();
    auto net_resource  = interpreter->GetNetResource();

    auto g = block.g();

    // set input shape
    InputShapesMap inputs_shape_map;
    int input_idx = 0;
    for (auto &input : g->inputs()) {
        inputs_shape_map[input->debugName()] = block.in_shape()[input_idx++];
        net_structure->blobs.insert(input->debugName());
        std::cout << "[ConvertBlockToInstance:input ] " << input->debugName() << std::endl;
    }
    net_structure->inputs_shape_map = inputs_shape_map;

    for (const auto node : g->block()->nodes()) {
        auto kind = node->kind();
        if (kind.is_prim() && ctx->dealPrime(node)) {
            continue;
        }

        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        LayerResource *layer_res = nullptr;
        ConvertNodeToLayer(node, layer_info.get(), &layer_res);
        net_structure->layers.push_back(layer_info);
        if (layer_res) {
            net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);
        }

        for (auto input : layer_info->inputs) {
            net_structure->blobs.insert(input);
        }

        for (auto output: layer_info->outputs) {
            net_structure->blobs.insert(output);
        }

        auto inputs = node->inputs();
        for (auto input : inputs) std::cout << "[node] input " << input->debugName() << std::endl;
        auto node_name = node->outputs().size() > 0 ? node->output(0)->debugName() : "";
        std::cout << "[ConvertBlockToInstance:node  ] " << node->kind().toUnqualString() << " " << node_name << " " << std::endl;
    }
    
    for (auto &output : g->outputs()) {
        std::cout << "[ConvertBlockToInstance:output] " << output->debugName() << std::endl;
        net_structure->blobs.insert(output->debugName());
        net_structure->outputs.insert(output->debugName());
    }

    instance_ptr->instance_->Init(ctx->get_interpreter(), inputs_shape_map);
    // set output blob names

    return instance_ptr;
}

}
}