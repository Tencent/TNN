#include "tnn/network/torch/torch_convert.h"
#include "tnn/network/torch/torch_op_converter.h"

namespace TNN_NS {
namespace conversion {

using namespace partitioning;


void ConvertNodeToLayer(const torch::jit::Node *node, LayerInfo *layer_info, LayerResource **layer_res) {
    const auto& op_type = node->kind().toUnqualString();

    auto& converter = GetGlobalTorchConvertMap()[op_type];
    converter->Convert(node, layer_info, layer_res);
    
    // Todo need register convert table

}

c10::intrusive_ptr<runtime::TNNEngine> ConvertBlockToInstance(partitioning::SegmentedBlock &block, NetworkConfig &config) {
    auto ctx = std::make_shared<runtime::TorchConvertCtx>();

    ModelConfig model_config;
    NetworkConfig network_config;

    network_config.device_type = config.device_type;
    network_config.device_id = config.device_id;
    auto instance_ptr = c10::make_intrusive<runtime::TNNEngine>(network_config, model_config);

    auto interpreter = dynamic_cast<DefaultModelInterpreter *>(ctx->get_interpreter().get());
    auto net_structure = interpreter->GetNetStructure();
    auto net_resource  = interpreter->GetNetResource();

    auto g = block.g();

    // set input shape
    InputShapesMap inputs_shape_map;
    int input_idx = 0;
    for (auto &input : g->inputs()) {
        // inputs_shape_map[input->debugName()] = block.in_shape()[input_idx++];
        net_structure->blobs.insert(input->debugName());
        instance_ptr->input_names.push_back(input->debugName());
        // std::cout << "[ConvertBlockToInstance:input ] " << input->debugName() << std::endl;
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
    }
    
    for (auto &output : g->outputs()) {
        // std::cout << "[ConvertBlockToInstance:output] " << output->debugName() << std::endl;
        net_structure->blobs.insert(output->debugName());
        net_structure->outputs.insert(output->debugName());
        instance_ptr->output_names.push_back(output->debugName());
    }

    instance_ptr->ctx_ = ctx;
    // instance_ptr->instance_->Init(ctx->get_interpreter(), inputs_shape_map);
    // set output blob names

    return instance_ptr;
}

}
}