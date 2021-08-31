#include "tnn/network/torch/torch_op_converter.h"
// #include <ATen/native/quantized/cpu/conv_packed_params.h>

namespace TNN_NS {

// the function schema is defined in aten/src/ATen/native/native_functions.ymal

namespace conversion
{
std::map<std::string, std::shared_ptr<TorchOpConverter>>& GetGlobalTorchConvertMap() {
    static std::once_flag once;
    static std::shared_ptr<std::map<std::string, std::shared_ptr<TorchOpConverter>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<std::string, std::shared_ptr<TorchOpConverter>>); });
    return *creators;
}

#define ADD_INPUTS_AND_OUTPUTS                                                                                         \
    for (auto input : layer_info->inputs) {                                                                            \
        net_structure->blobs.insert(input);                                                                            \
    }                                                                                                                  \
    for (auto output : layer_info->outputs) {                                                                          \
        net_structure->blobs.insert(output);                                                                           \
    }

// func: conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor
class Conv2DTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
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
        const auto group = getValue<int64_t>(inputs[6]);
        auto weight_buf = getValue(weight);
        auto shape = weight_buf.GetBufferDims();

        // set param accroding to real value, just test here
        layer_param->name = layer_info->name;
        layer_param->pad_type = 3;
        layer_param->output_channel = shape[0];
        layer_param->input_channel = shape[1];
        layer_param->kernels = {shape[2], shape[3]};
        layer_param->dialations = {(int)dialation[0], (int)dialation[1]};
        layer_param->strides = {(int)stride[0], (int)stride[1]};
        layer_param->group = group;
        layer_param->pads = {(int)padding[0], (int)padding[0], (int)padding[1], (int)padding[1]};
        conv_res->name = layer_info->name;
        conv_res->filter_handle = weight_buf;

        auto bias_buf = getValue(bias);
        if (bias_buf.GetBytesSize() != 0) {
            layer_param->bias = 1;
            conv_res->bias_handle = bias_buf;
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(conv_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

// func: _convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, 
//                    int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor
class _ConvTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        const auto& inputs = node->inputs();
        const auto transposed = getValue<bool>(inputs[6]);
        return !transposed; 
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
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
        const auto group = getValue<int64_t>(inputs[8]);
        // const auto transposed = getValue<bool>(inputs[6]);

        // if (transposed) {
        //     layer_info->type_str = LAYER_DECONVOLUTION;
        //     std::cout << "deconv" << std::endl;
        // }

        auto weight_buf = getValue(weight);
        auto shape = weight_buf.GetBufferDims();

        // set param accroding to real value, just test here
        layer_param->name = layer_info->name;
        layer_param->pad_type = 3;
        layer_param->output_channel = shape[0];
        layer_param->input_channel = shape[1];
        layer_param->kernels = {shape[2], shape[3]};
        layer_param->dialations = {(int)dialation[0], (int)dialation[1]};
        layer_param->strides = {(int)stride[0], (int)stride[1]};
        layer_param->pads = {(int)padding[0], (int)padding[0], (int)padding[1], (int)padding[1]};
        layer_param->group = group;
        conv_res->name = layer_info->name;
        conv_res->filter_handle = weight_buf;

        auto bias_buf = getValue(bias);
        if (bias_buf.GetBytesSize() != 0) {
            layer_param->bias = 1;
            conv_res->bias_handle = bias_buf;
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(conv_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

// func: max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
// func: avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
// func: adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
class PoolTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_POOLING;
        layer_info->type_str = "Pooling";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto pool_param = std::make_shared<PoolingLayerParam>();
        pool_param->name = layer_info->name;
        std::string op_type = node->kind().toUnqualString();

        if (op_type.find("adaptive") == std::string::npos) {
            const auto kernel_size = getValue<std::vector<int64_t>>(inputs[1]);
            const auto stride = getValue<std::vector<int64_t>>(inputs[2]);
            const auto padding = getValue<std::vector<int64_t>>(inputs[3]);
            const auto dialation = getValue<std::vector<int64_t>>(inputs[4]);
            const auto ceil_mode = getValue<bool>(inputs[5]);
            
            pool_param->pad_type = 3;
            pool_param->kernels_params = {(int)kernel_size[0], (int)kernel_size[1]};
            pool_param->strides = {(int)stride[0], (int)stride[1]};
            pool_param->pads = {(int)padding[0], (int)padding[0], (int)padding[1], (int)padding[1]};
            pool_param->kernel_indexs = {-1, -1};
            pool_param->kernels = {-1, -1};
            pool_param->output_shape = {-1, -1};
            pool_param->ceil_mode = ceil_mode;
        } else {
            return TNNERR_LAYER_ERR;
        }

        layer_info->param = pool_param;

        net_structure->layers.push_back(layer_info);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

// func: relu_(Tensor(a!) self) -> Tensor(a!)
class ReluTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_RELU;
        layer_info->type_str = "Relu";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        layer_info->param = std::make_shared<LayerParam>();

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
class AddTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
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

        net_structure->layers.push_back(layer_info);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

// class QuantConv2DTorchConverter : public TorchOpConverter {
// public:
//     Status Convert(const torch::jit::Node *node, LayerInfo *layer_info, LayerResource **layer_resouce) {
//         const auto& inputs = node->inputs();
//         auto weight = toIValue(inputs[1]).value();
//         std::cout << weight.isTuple() << std::endl;
//         std::cout << weight.isTensor() << std::endl;
//         std::cout << weight.isObject() << std::endl;
//         auto object = weight.toObject().get();
//         auto slots = object->slots();
//         for (auto &slot : slots) {
//             std::cout << slot.isCapsule() << std::endl;
//             auto conv_param = reinterpret_cast<ConvPackedParamsBase<2> *>(slot.toCapsule().get());
//             // c10::intrusive_ptr<ConvPackedParamsBase<2>> conv_param = slot.toCapsule();
//             std::cout << "get" << std::endl;
//         }

//         return TNN_OK;
//     }
// };

REGISTER_TORCH_OP_CONVERTER(Conv2D, aten, conv2d)
REGISTER_TORCH_OP_CONVERTER(_Conv, aten, _convolution)
REGISTER_TORCH_OP_CONVERTER(Relu, aten, relu_)
REGISTER_TORCH_OP_CONVERTER(Pool, aten, max_pool2d)
REGISTER_TORCH_OP_CONVERTER(Add, aten, add_)


// REGISTER_TORCH_OP_CONVERTER(QuantConv2D, quantized, conv2d)

} // namespace conversion
}