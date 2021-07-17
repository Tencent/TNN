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

        const auto& inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<ConvLayerParam>();
        auto conv_res = new(ConvLayerResource);
        const auto weight = inputs[1];
        const auto bias = inputs[2];
        const auto stride = getValue<std::vector<int64_t>>(inputs[3]);
        const auto padding = getValue<std::vector<int64_t>>(inputs[4]);
        const auto dialation = getValue<std::vector<int64_t>>(inputs[5]);
        std::vector<int> shape;
        auto weight_vec = getValue<float>(weight, shape);

        // set param accroding to real value, just test here
        layer_param->name = "Convolution";
        layer_param->pad_type = 0;
        layer_param->output_channel = shape[0];
        layer_param->input_channel = shape[1];
        layer_param->kernels = {shape[2], shape[3]};
        layer_param->dialations = {(int)dialation[0], (int)dialation[1]};
        layer_param->strides = {(int)stride[0], (int)stride[1]};
        layer_param->pads = {(int)padding[0], (int)padding[0], (int)padding[1], (int)padding[1]};
        conv_res->filter_handle = RawBuffer(weight_vec.size() * sizeof(float), reinterpret_cast<char *>(weight_vec.data()));

        auto bias_vec = getValue<float>(bias, shape);
        if (bias_vec.size() != 0) {
            layer_param->bias = 1;
            conv_res->bias_handle = RawBuffer(bias_vec.size() * sizeof(float), reinterpret_cast<char *>(bias_vec.data()));
        }

        layer_info->param = layer_param;
        *layer_resouce = conv_res;


        return TNN_OK;
    }
};

class PoolTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, LayerInfo *layer_info, LayerResource **layer_resouce) {
        layer_info->type = LAYER_POOLING;
        layer_info->type_str = "Pooling";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto pool_param = std::make_shared<PoolingLayerParam>();
        pool_param->name = "Pooling";
        std::string op_type = node->kind().toUnqualString();

        if (op_type.find("adaptive") == std::string::npos) {
            const auto kernel_size = getValue<std::vector<int64_t>>(inputs[1]);
            const auto stride = getValue<std::vector<int64_t>>(inputs[2]);
            const auto padding = getValue<std::vector<int64_t>>(inputs[3]);
            
            pool_param->pad_type = 0;
            pool_param->kernels_params = {(int)kernel_size[0], (int)kernel_size[1]};
            pool_param->strides = {(int)stride[0], (int)stride[1]};
            pool_param->pads = {(int)padding[0], (int)padding[0], (int)padding[1], (int)padding[1]};
            pool_param->kernel_indexs = {-1, -1};
            pool_param->kernels = {-1, -1};
        } else {
            return TNNERR_LAYER_ERR;
        }

        layer_info->param = pool_param;
        *layer_resouce = nullptr;
        
        return TNN_OK;
    }
};

class ReluTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, LayerInfo *layer_info, LayerResource **layer_resouce) {
        layer_info->type = LAYER_RELU;
        layer_info->type_str = "Relu";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());
        
        layer_info->param = std::make_shared<LayerParam>();
        *layer_resouce = nullptr;
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
            if (layer_info->inputs.size() == 2) break;
        }

        for (auto &output : node->outputs()) {
            layer_info->outputs.push_back(output->debugName());
        }

        layer_info->param = std::make_shared<MultidirBroadcastLayerParam>();
        *layer_resouce = nullptr;

        return TNN_OK;
    }
};

REGISTER_TORCH_OP_CONVERTER(Conv, conv2d)
REGISTER_TORCH_OP_CONVERTER(Relu, relu_)
REGISTER_TORCH_OP_CONVERTER(Pool, max_pool2d)
REGISTER_TORCH_OP_CONVERTER(Add, add_)

} // namespace conversion
}