#include "tnn/network/torch/torch_convert.h"
#include "tnn/network/torch/torch_op_converter.h"
#include "tnn/interpreter/tnn/model_interpreter.h"
namespace TNN_NS {
namespace conversion {

using namespace partitioning;


void ConvertNodeToLayer(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
    const auto& op_type = node->kind().toQualString();

    auto& converter = GetGlobalTorchConvertMap()[op_type];
    converter->Convert(node, net_structure, net_resource);
    
    // Todo need register convert table

}

c10::intrusive_ptr<runtime::TNNEngine> ConvertBlockToInstance(partitioning::SegmentedBlock &block, NetworkConfig &config) {
    auto ctx = std::make_shared<runtime::TorchConvertCtx>();

    ModelConfig model_config;
    NetworkConfig network_config;

    network_config.device_type = config.device_type;
    network_config.device_id = config.device_id;
    network_config.precision = (config.device_type == DEVICE_CUDA && config.precision == PRECISION_AUTO)? PRECISION_LOW : config.precision;
    network_config.share_memory_mode = (config.device_type == DEVICE_CUDA)? SHARE_MEMORY_MODE_SHARE_ONE_THREAD : SHARE_MEMORY_MODE_DEFAULT;
    auto instance_ptr = c10::make_intrusive<runtime::TNNEngine>(network_config, model_config);

    auto interpreter = dynamic_cast<IRModelInterpreter *>(ctx->get_interpreter().get());
    auto net_structure = interpreter->GetNetStructure();
    auto net_resource  = interpreter->GetNetResource();

    auto g = block.g();
#ifdef SAVE_CACHE_FILE
    interpreter->InterpretMd5(g->toString(false));
#endif

    // set input shape
    InputShapesMap inputs_shape_map;
    int input_idx = 0;
    for (auto &input : g->inputs()) {
        // inputs_shape_map[input->debugName()] = block.in_shape()[input_idx++];
        net_structure->blobs.insert(input->debugName());
        instance_ptr->input_names.push_back(input->debugName());
        if (block.max_in_shape().size()) {
            instance_ptr->min_inputs_shape.push_back(block.min_in_shape()[input_idx]);
            instance_ptr->max_inputs_shape.push_back(block.max_in_shape()[input_idx]);
            input_idx++;
        }
        // std::cout << "[ConvertBlockToInstance:input ] " << input->debugName() << std::endl;
    }
    net_structure->inputs_shape_map = inputs_shape_map;

    for (const auto node : g->block()->nodes()) {
        auto kind = node->kind();
        if (kind.is_prim() && ctx->dealPrime(node)) {
            continue;
        }

        ConvertNodeToLayer(node, net_structure, net_resource);
    }
    
    for (auto &output : g->outputs()) {
        // std::cout << "[ConvertBlockToInstance:output] " << output->debugName() << std::endl;
        net_structure->blobs.insert(output->debugName());
        net_structure->outputs.insert(output->debugName());
        instance_ptr->output_names.push_back(output->debugName());
    }

    instance_ptr->ctx_ = ctx;
    instance_ptr->network_config_ = network_config;
    // instance_ptr->instance_->Init(ctx->get_interpreter(), inputs_shape_map);
    // set output blob names
    if (block.min_in_shape().size() != 0 && block.max_in_shape().size() != 0 && block.min_in_shape().size() == block.max_in_shape().size()) {
        InputShapesMap min_inputs_shape_map;
        InputShapesMap max_inputs_shape_map;
        for (int i = 0; i < instance_ptr->input_names.size(); i++) {
            min_inputs_shape_map[instance_ptr->input_names[i]] = block.min_in_shape()[i];
            max_inputs_shape_map[instance_ptr->input_names[i]] = block.max_in_shape()[i];
        }
        net_structure->inputs_shape_map = max_inputs_shape_map;
        instance_ptr->instance_->Init(ctx->get_interpreter(), min_inputs_shape_map, max_inputs_shape_map);
        instance_ptr->is_init_ = true;
    }

    return instance_ptr;
}

}
}