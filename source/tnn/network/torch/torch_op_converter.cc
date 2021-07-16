#include "tnn/network/torch/torch_op_converter.h"

namespace TNN_NS {
namespace conversion
{
std::map<std::string, std::shared_ptr<TorchOpConverter>>& GetGlobalTorchConvertMap() {
    static std::once_flag once;
    static std::shared_ptr<std::map<std::string, std::shared_ptr<TorchOpConverter>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<std::string, std::shared_ptr<TorchOpConverter>>); });
    return *creators;
}

class ConvTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, LayerInfo *layer_info, LayerResource **layer_resouce) {
        layer_info->type = LAYER_CONVOLUTION;
        layer_info->type_str = "Convolution";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        return TNN_OK;
    }
};

class PoolTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, LayerInfo *layer_info, LayerResource **layer_resouce) {
        layer_info->type = LAYER_RELU;
        layer_info->type_str = "Relu";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());
        
        return TNN_OK;
    }
};

class ReluTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, LayerInfo *layer_info, LayerResource **layer_resouce) {
        layer_info->type = LAYER_POOLING;
        layer_info->type_str = "Pooling";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());
        
        return TNN_OK;
    }
};

class AddTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, LayerInfo *layer_info, LayerResource **layer_resouce) {
        layer_info->type = LAYER_ADD;
        layer_info->type_str = "Add";
        layer_info->name = node->output(0)->debugName();
        for (auto &input : node->inputs()) {
            layer_info->inputs.push_back(input->debugName());
        }

        for (auto &output : node->outputs()) {
            layer_info->outputs.push_back(output->debugName());
        }        
        return TNN_OK;
    }
};

REGISTER_TORCH_OP_CONVERTER(Conv, conv2d)
REGISTER_TORCH_OP_CONVERTER(Relu, relu_)
REGISTER_TORCH_OP_CONVERTER(Pool, max_pool2d)
REGISTER_TORCH_OP_CONVERTER(Add, add_)

} // namespace conversion
}