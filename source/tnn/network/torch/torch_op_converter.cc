#include "tnn/network/torch/torch_op_converter.h"
// #include <ATen/native/quantized/cpu/conv_packed_params.h>

namespace TNN_NS {

// the function schema is defined in aten/src/ATen/native/native_functions.ymal
// Todo: tnn tensorrt plugin not fully support fp16, resource rawbuffer should be convert to fp32 to avoid init error

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
        auto layer_res = new(ConvLayerResource);
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
        layer_param->pad_type = -1;
        layer_param->output_channel = shape[0];
        layer_param->input_channel = shape[1];
        // order [w, h]
        layer_param->kernels = {shape[3], shape[2]};
        layer_param->dialations = {(int)dialation[1], (int)dialation[0]};
        layer_param->strides = {(int)stride[1], (int)stride[0]};
        layer_param->group = group;
        layer_param->pads = {(int)padding[1], (int)padding[1], (int)padding[0], (int)padding[0]};
        layer_res->name = layer_info->name;
        layer_res->filter_handle = ConvertHalfHandle(weight_buf);

        if (toIValue(bias)->isTensor()) {
            layer_param->bias      = 1;
            layer_res->bias_handle = getValue(bias);
            layer_res->bias_handle = ConvertHalfHandle(layer_res->bias_handle);
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

// func: _convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, 
//                    int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor
class _ConvTorchConverter : public TorchOpConverter {
public:
  /*
    bool IsSupported(const torch::jit::Node *node) {
        const auto& inputs = node->inputs();
        const auto transposed = getValue<bool>(inputs[6]);
        return !transposed; 
    }
  */

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        const auto& inputs = node->inputs();
        const auto transposed = getValue<bool>(inputs[6]);
        
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = transposed ? LAYER_DECONVOLUTION : LAYER_CONVOLUTION;;
        layer_info->type_str = transposed ? "Deconvolution" : "Convolution";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<ConvLayerParam>();
        auto layer_res = new(ConvLayerResource);
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
        layer_param->pad_type = -1;
        layer_param->output_channel = shape[0];
        layer_param->input_channel = shape[1];
        layer_param->kernels = {shape[3], shape[2]};
        layer_param->dialations = {(int)dialation[1], (int)dialation[0]};
        layer_param->strides = {(int)stride[1], (int)stride[0]};
        layer_param->pads = {(int)padding[1], (int)padding[1], (int)padding[0], (int)padding[0]};
        layer_param->group = group;
        layer_res->name = layer_info->name;
        layer_res->filter_handle = ConvertHalfHandle(weight_buf);

        auto bias_buf = getValue(bias);
        if (bias_buf.GetBytesSize() != 0) {
            layer_param->bias = 1;
            layer_res->bias_handle = ConvertHalfHandle(bias_buf);
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

// func: max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
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

        auto layer_param = std::make_shared<PoolingLayerParam>();
        layer_param->name = layer_info->name;
        std::string op_type = node->kind().toUnqualString();

        if (op_type.find("adaptive") == std::string::npos) {
            const auto kernel_size = getValue<std::vector<int64_t>>(inputs[1]);
            const auto stride = getValue<std::vector<int64_t>>(inputs[2]);
            const auto padding = getValue<std::vector<int64_t>>(inputs[3]);
            const auto dialation = getValue<std::vector<int64_t>>(inputs[4]);
            const auto ceil_mode = getValue<bool>(inputs[5]);
            
            layer_param->pad_type = -1;
            layer_param->kernels_params = {(int)kernel_size[1], (int)kernel_size[0]};
            layer_param->strides = {(int)stride[1], (int)stride[0]};
            layer_param->pads = {(int)padding[1], (int)padding[1], (int)padding[0], (int)padding[0]};
            layer_param->kernel_indexs = {-1, -1};
            layer_param->kernels = {-1, -1};
            layer_param->output_shape = {-1, -1};
            layer_param->ceil_mode = ceil_mode;
        } else {
            const auto output_shape = getValue<std::vector<int64_t>>(inputs[1]);
            layer_param->is_adaptive_pool = 1;
            layer_param->output_shape = {(int)output_shape[1], (int)output_shape[0]};
            layer_param->kernels_params = {-1, -1};
            layer_param->strides = {1, 1};
            layer_param->pads = {0, 0, 0, 0};
            layer_param->kernel_indexs = {-1, -1};
            layer_param->kernels = {-1, -1};
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

// func: avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
class AvgPoolTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_POOLING;
        layer_info->type_str                  = "Pooling";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param  = std::make_shared<PoolingLayerParam>();
        layer_param->name = layer_info->name;

        auto kernel_size = getValue<std::vector<int64_t>>(inputs[1]);
        auto stride      = getValue<std::vector<int64_t>>(inputs[2]);
        auto padding     = getValue<std::vector<int64_t>>(inputs[3]);
        auto ceil_mode   = getValue<bool>(inputs[4]);

        /*
         * When padding in AvgPool is not 0, the inference results of Pytorch and TNN are inconsistent.
         * Therefore, when converting, insert the Pad operator before AvgPool,
         * and set padding of AvgPool to 0 at the same time.

         * E.g.，
         * In AvgPool，kernel_size = 3, stride=1, padding=1
         * Input，
         * 1.0, 1.0, 1.0
         * 1.0, 1.0, 1.0
         * 1.0, 1.0, 1.0
         * the output of Pytorch，
         * 0.444, 0.667, 0.444
         * 0.667, 1.000, 0.667
         * 0.444, 0.667, 0.444
         * the output of TNN (Pad operator is not inserted)
         * 1.0, 1.0, 1.0
         * 1.0, 1.0, 1.0
         * 1.0, 1.0, 1.0
         */

        bool need_insert_pad = false;
        for (const auto &pad : padding) {
            need_insert_pad = (pad != 0);
        }

        if (need_insert_pad) {
            std::shared_ptr<LayerInfo> pad_layer_info = std::make_shared<LayerInfo>();
            pad_layer_info->type                      = LAYER_PAD;
            pad_layer_info->type_str                  = "Pad";
            pad_layer_info->name                      = layer_info->name + "_pad";

            pad_layer_info->inputs.push_back(layer_info->inputs[0]);
            pad_layer_info->outputs.push_back(pad_layer_info->name);
            layer_info->inputs[0] = pad_layer_info->outputs[0];

            auto pad_layer_param  = std::make_shared<PadLayerParam>();
            const int pad_h       = static_cast<int>(padding[0]);
            const int pad_w       = static_cast<int>(padding[1]);
            pad_layer_param->pads = {pad_w, pad_w, pad_h, pad_h, 0, 0, 0, 0};

            pad_layer_info->param = pad_layer_param;

            net_structure->layers.push_back(pad_layer_info);

            for (const auto &pad_input : pad_layer_info->inputs) {
                net_structure->blobs.insert(pad_input);
            }
            for (const auto &pad_output : pad_layer_info->outputs) {
                net_structure->blobs.insert(pad_output);
            }
        }

        layer_param->pool_type      = 1;
        layer_param->pad_type       = -1;
        layer_param->kernels_params = {(int)kernel_size[1], (int)kernel_size[0]};
        layer_param->strides        = {(int)stride[1], (int)stride[0]};
        layer_param->pads           = {0, 0, 0, 0};
        layer_param->kernel_indexs  = {-1, -1};
        layer_param->kernels        = {-1, -1};
        layer_param->output_shape   = {-1, -1};
        layer_param->ceil_mode      = ceil_mode;

        layer_info->param = layer_param;

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
class BinaryTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        switch (node->kind()) {
            case at::aten::add:
            case at::aten::add_:
                layer_info->type     = LAYER_ADD;
                layer_info->type_str = "Add";
                break;
            case at::aten::mul:
                layer_info->type     = LAYER_MUL;
                layer_info->type_str = "Mul";
                break;
            case at::aten::div:
                layer_info->type     = LAYER_DIV;
                layer_info->type_str = "Div";
                break;
            case at::aten::gt:
                layer_info->type     = LAYER_GREATER;
                layer_info->type_str = "Greater";
                break;
            default:
                LOGE("Unsupport layer type %s\n", node->kind().toUnqualString());
                ASSERT(0);
        }
        layer_info->name = node->output(0)->debugName();

        auto layer_param = std::make_shared<MultidirBroadcastLayerParam>();

        const auto &inputs     = node->inputs();
        const auto input0_kind = inputs[0]->node()->kind();
        const auto input1_kind = inputs[1]->node()->kind();
        if (input0_kind == at::prim::Constant || input1_kind == at::prim::Constant) {
            const int weight_input_index    = input0_kind == at::prim::Constant ? 0 : 1;
            const int input_index           = input0_kind == at::prim::Constant ? 1 : 0;
            layer_param->weight_input_index = weight_input_index;
            layer_info->inputs.push_back(inputs[input_index]->debugName());

            auto layer_res            = new EltwiseLayerResource();
            auto element_buf          = getValue(inputs[weight_input_index]);
            layer_res->element_handle = ConvertHalfHandle(element_buf);
            layer_res->element_shape  = element_buf.GetBufferDims();

            net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);
        } else {
            layer_param->weight_input_index = -1;
            for (auto &input : node->inputs()) {
                layer_info->inputs.push_back(input->debugName());
                if (layer_info->inputs.size() == 2) {
                    break;
                }
            }
        }

        for (auto &output : node->outputs()) {
            layer_info->outputs.push_back(output->debugName());
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

class FlattenTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_FLATTEN;
        layer_info->type_str = "Flatten";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<FlattenLayerParam>();
        layer_param->axis = getValue<int64_t>(inputs[1]);
        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

// func: gelu(const at::Tensor & self)
class GeluTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_GELU;
        layer_info->type_str                  = "GELU";
        layer_info->name                      = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());
        layer_info->param = std::make_shared<LayerParam>();

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: layer_norm(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight={}, const c10::optional<at::Tensor> & bias={}, double eps=1e-05, bool cudnn_enable=true);
class LayerNormTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_LAYER_NORM;
        layer_info->type_str                  = "LayerNorm";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = node->inputs();

        // https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html?highlight=layernorm#torch.nn.LayerNorm
        // Assume TorchScript is well-formed, weight, bias are present,
        // weight.shape, bias.shape = normalized_shape
        layer_info->inputs.push_back(inputs[0]->debugName()); // input
        layer_info->inputs.push_back(inputs[2]->debugName()); // weight
        layer_info->inputs.push_back(inputs[3]->debugName()); // bias
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        const auto normalized_shape = getValue<std::vector<int64_t>>(inputs[1]);
        const auto eps              = getValue<float>(inputs[4]);
        auto layer_param = std::make_shared<LayerNormLayerParam>();
        layer_param->reduce_dims_size = normalized_shape.size();
        layer_param->eps = eps;
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);
        net_resource->constant_map[inputs[2]->debugName()] = std::make_shared<RawBuffer>(getValue(inputs[2])); // weight
        net_resource->constant_map[inputs[3]->debugName()] = std::make_shared<RawBuffer>(getValue(inputs[3])); // bias

        return TNN_OK;
    }
};

// func: linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
class LinearTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_INNER_PRODUCT;
        layer_info->type_str = "InnerProduct";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<InnerProductLayerParam>();
        auto layer_res = new(InnerProductLayerResource);
        const auto weight = inputs[1];
        const auto bias = inputs[2];

        auto weight_buf = getValue(weight);
        auto shape = weight_buf.GetBufferDims();

        // set param accroding to real value, just test here
        layer_param->name = layer_info->name;
        layer_param->num_output = shape[0];
        layer_param->axis = 1;

        layer_res->name = layer_info->name;
        layer_res->weight_handle = ConvertHalfHandle(weight_buf);

        auto bias_buf = getValue(bias);
        if (bias_buf.GetBytesSize() != 0) {
            layer_param->has_bias = 1;
            layer_res->bias_handle = ConvertHalfHandle(bias_buf);
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

// func: matmul(Tensor self, Tensor other) -> Tensor
class MatMulTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_MATMUL;
        layer_info->type_str = "MatMul";
        layer_info->name = node->output(0)->debugName();

        // https://pytorch.org/docs/stable/generated/torch.matmul.html?highlight=matmul#torch.matmul
        // Torch matmul has two inputs, no weight resource.
        // param.weight_position == 1 by default. axis == 0 by default.
        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->inputs.push_back(node->inputs()[1]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<MatMulLayerParam>();
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(new(MatMulLayerResource));

        return TNN_OK;
    }
};

// func: aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
class PermuteTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_PERMUTE;
        layer_info->type_str = "Permute";
        layer_info->name = node->output(0)->debugName();

        // https://pytorch.org/docs/stable/generated/torch.permute.html?highlight=permute#torch.permute
        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<PermuteLayerParam>();
        std::vector<int> permute_orders;
        for (auto dim : getValue<std::vector<int64_t>>(node->inputs()[1])) {
            permute_orders.emplace_back(static_cast<int>(dim));
        }
        layer_param->orders = permute_orders;
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);
        return TNN_OK;
    }
};

// func: hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!
class HardTanhTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_CLIP;
        layer_info->type_str = "Clip";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<ClipLayerParam>();

        layer_param->min = getValue<float>(inputs[1]);
        layer_param->max = getValue<float>(inputs[2]);

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class HardSigmoidTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_HARDSIGMOID;
        layer_info->type_str = "Hardsigmoid";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<HardSigmoidLayerParam>();

        layer_param->alpha = 1.0f / 6;
        layer_param->beta = 0.5f;

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class HardSwishTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_HARDSWISH;
        layer_info->type_str = "Hardswish";
        layer_info->name = node->output(0)->debugName();

        const auto& inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<HardSwishLayerParam>();

        layer_param->alpha = 1.0f / 6;
        layer_param->beta = 0.5f;

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class BatchNormTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_BATCH_NORM;
        layer_info->type_str                  = "BatchNormCxx";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        const auto weight       = inputs[1];
        const auto bias         = inputs[2];
        const auto running_mean = inputs[3];
        const auto running_var  = inputs[4];
        const auto eps          = getValue<float>(inputs[7]);

        auto layer_param = std::make_shared<BatchNormLayerParam>();
        auto layer_res   = new BatchNormLayerResource();

        layer_param->eps = eps;

        {
            auto weight_buf = getValue(weight);
            auto bias_buf   = getValue(bias);
            auto mean_buf   = getValue(running_mean);
            auto var_buf    = getValue(running_var);

            auto fuseResource = [&](RawBuffer &gamma, RawBuffer &beta, RawBuffer &mean, RawBuffer &var,
                                    float eps) -> std::pair<RawBuffer, RawBuffer> {
                const int size       = gamma.GetDataCount();
                const auto data_type = gamma.GetDataType();
                auto gamma_fp32      = ConvertHalfHandle(gamma);
                auto beta_fp32       = ConvertHalfHandle(beta);
                auto mean_fp32       = ConvertHalfHandle(mean);
                auto var_fp32        = ConvertHalfHandle(var);
                auto *gamma_ptr      = gamma_fp32.force_to<float *>();
                auto *beta_ptr       = beta_fp32.force_to<float *>();
                auto *mean_ptr       = mean_fp32.force_to<float *>();
                auto *var_ptr        = var_fp32.force_to<float *>();

                auto scale      = std::shared_ptr<float>(new float[size], [](float *p) { delete[] p; });
                auto bias       = std::shared_ptr<float>(new float[size], [](float *p) { delete[] p; });
                auto *scale_ptr = scale.get();
                auto *bias_ptr  = bias.get();

                for (int i = 0; i < size; i++) {
                    double sqrt_var = 1.0 / std::sqrt(static_cast<double>(var_ptr[i] + eps));
                    bias_ptr[i]     = beta_ptr[i] - static_cast<float>(static_cast<double>(gamma_ptr[i]) *
                                                                   static_cast<double>(mean_ptr[i]) * sqrt_var);
                    scale_ptr[i]    = static_cast<float>(static_cast<double>(gamma_ptr[i]) * sqrt_var);
                }

                const int byte_size = size * sizeof(float);
                auto scale_buf_fp32 = RawBuffer(byte_size, reinterpret_cast<char *>(scale_ptr), gamma.GetBufferDims());
                auto bias_buf_fp32  = RawBuffer(byte_size, reinterpret_cast<char *>(bias_ptr), beta.GetBufferDims());
                // auto scale_buf      = data_type == DATA_TYPE_HALF ? ConvertFloatToHalf(scale_buf_fp32) : scale_buf_fp32;
                // auto bias_buf       = data_type == DATA_TYPE_HALF ? ConvertFloatToHalf(bias_buf_fp32) : bias_buf_fp32;

                return std::make_pair(scale_buf_fp32, bias_buf_fp32);
            };

            auto scaleAndBias = fuseResource(weight_buf, bias_buf, mean_buf, var_buf, eps);

            layer_res->name         = layer_info->name;
            layer_res->scale_handle = scaleAndBias.first;
            layer_res->bias_handle  = scaleAndBias.second;
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

class ConcatTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_CONCAT;
        layer_info->type_str                  = "Concat";
        layer_info->name                      = node->output(0)->debugName();

        const auto inputs      = node->inputs();
        const auto tensor_list = inputs[0];
        for (const auto input : tensor_list->node()->inputs()) {
            layer_info->inputs.push_back(input->debugName());
        }
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param  = std::make_shared<ConcatLayerParam>();
        layer_param->axis = static_cast<int>(getValue<int64_t>(inputs[1]));
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class UnsqueezeTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_UNSQUEEZE;
        layer_info->type_str                  = "Unsqueeze";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<UnsqueezeLayerParam>();

        layer_param->axes = {static_cast<int>(getValue<int64_t>(inputs[1]))};

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class GatherTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_GATHER;
        layer_info->type_str                  = "Gather";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<GatherLayerParam>();
        auto layer_res   = new GatherLayerResource();

        layer_param->axis                = static_cast<int>(getValue<int64_t>(inputs[1]));
        layer_param->data_in_resource    = false;
        layer_param->indices_in_resource = true;

        int index        = getValue<int64_t>(inputs[2]);
        auto indices_buf = RawBuffer(4, reinterpret_cast<char *>(&index), {});
        indices_buf.SetDataType(DATA_TYPE_INT32);
        layer_res->indices = indices_buf;

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        return TNN_OK;
    }
};

// func: slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
class StridedSliceTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_STRIDED_SLICE_V2;
        layer_info->type_str                  = "StridedSliceV2";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<StrideSliceV2LayerParam>();

        // Rule to set default values for param: start, end of aten::slice
        // is defined in pytorch/aten/src/ATen/TensorIndexing.h
        layer_param->axes    = {static_cast<int>(getValue<int64_t>(inputs[1]))};
        layer_param->strides = {static_cast<int>(getValue<int64_t>(inputs[4]))};

        if (inputs[2]->type()->kind() == c10::TypeKind::NoneType) {
            layer_param->begins = {layer_param->strides[0]<0 ? INT_MAX : 0};
        } else {
            layer_param->begins = {static_cast<int>(getValue<int64_t>(inputs[2]))};
        }
        if (inputs[2]->type()->kind() == c10::TypeKind::NoneType) {
            layer_param->ends = {layer_param->strides[0]<0 ? INT_MIN : INT_MAX};
        } else {
            auto end = getValue<int64_t>(inputs[3]);
            layer_param->ends = {end > INT_MAX? INT_MAX : static_cast<int>(end)};
        }

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class SigmoidTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_SIGMOID;
        layer_info->type_str = "Sigmoid";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        layer_info->param = std::make_shared<LayerParam>();

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class SizeTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        for (int i = 0; i < node->output()->uses().size(); i++) {
            if (node->output()->uses()[i].user->kind() != at::prim::ListConstruct) {
                return false;
            } else {
                auto& converter = GetGlobalTorchConvertMap()["prim::ListConstruct"];
                if (!converter->IsSupported(node->output()->uses()[i].user))
                    return false;
                }
        }
        return true;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        // generate shape layer
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_SHAPE;
            layer_info->type_str                  = "Shape";
            layer_info->name                      = node->output(0)->debugName() + "_shape";

            layer_info->inputs.push_back(node->inputs()[0]->debugName());
            layer_info->outputs.push_back(node->outputs()[0]->debugName() + "_shape");

            layer_info->param = std::make_shared<LayerParam>();

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
        }

        // generate gather layer
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_GATHER;
            layer_info->type_str                  = "Gather";
            layer_info->name                      = node->output(0)->debugName() + "_gather";

            layer_info->inputs.push_back(node->outputs()[0]->debugName() + "_shape");
            layer_info->outputs.push_back(node->outputs()[0]->debugName() + "_gather");

            auto layer_param                 = std::make_shared<GatherLayerParam>();
            layer_param->axis                = 0;
            layer_param->indices_in_resource = true;

            layer_info->param = layer_param;

            const auto indices = getValue(node->inputs()[1]);
            auto layer_res     = std::make_shared<GatherLayerResource>();
            layer_res->indices = indices;

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
            net_resource->resource_map[layer_info->name] = layer_res;
        }

        // generate unsqueeze layer
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_UNSQUEEZE;
            layer_info->type_str                  = "Unsqueeze";
            layer_info->name                      = node->output(0)->debugName();

            layer_info->inputs.push_back(node->outputs()[0]->debugName() + "_gather");
            layer_info->outputs.push_back(node->outputs()[0]->debugName());

            auto layer_param  = std::make_shared<UnsqueezeLayerParam>();
            layer_param->axes = {0};

            layer_info->param = layer_param;

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
        }

        return TNN_OK;
    }
};

// func: aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
//       aten::softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor, NOT SUPPORTED NOW
//       dtype NOT SUPPORTED NOW.
class SoftmaxTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_SOFTMAX;
        layer_info->type_str = "SoftmaxCaffe";
        layer_info->name = node->output(0)->debugName();

        // https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html?highlight=softmax#torch.nn.Softmax
        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<SoftmaxLayerParam>();
        layer_param->axis = static_cast<int>(getValue<int64_t>(node->inputs()[1]));
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: split.Tensor(Tensor(a -> *) self, int split_size, int dim=0) -> Tensor(a)[]
class SplitTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        if (node->outputs().at(0)->node()->kind() == c10::prim::ListUnpack) {
            return true;
        }
        return true;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_SPLITTORCH;
        layer_info->type_str = "SplitTorch";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        // auto unpack_node = node->output()->uses()[0].user;
        auto unpack_node = node->next();
	for (const auto output : unpack_node->outputs()) {
            layer_info->outputs.push_back(output->debugName());
        }

        auto layer_param = std::make_shared<SplitTorchLayerParam>();
        layer_param->split_size = static_cast<int>(getValue<int64_t>(node->inputs()[1]));
        layer_param->axis = static_cast<int>(getValue<int64_t>(node->inputs()[2]));
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class ToTorchConverter : public TorchOpConverter {
public:
    // Currently, casting data to float is not supported.
    bool IsSupported(const torch::jit::Node *node) {
        const auto& inputs = node->inputs();
        if (inputs[1]->type()->kind() == c10::TypeKind::ScalarTypeType) {
            const auto dtype = getValue<int>(inputs[1]);
            // "6" means float data
            return dtype == 6;
        } else {
            return false;
        }
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_CAST;
        layer_info->type_str = "Cast";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<CastLayerParam>();
        layer_param->to = DATA_TYPE_FLOAT;
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: view(Tensor(a) self, int[] size) -> Tensor(a)
class ReshapeTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_RESHAPE;
        layer_info->type_str                  = "Reshape";
        layer_info->name                      = node->output(0)->debugName();

        for (const auto &input : node->inputs()) {
            layer_info->inputs.push_back(input->debugName());
        }
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param      = std::make_shared<ReshapeLayerParam>();

        if (!toIValue(node->inputs()[1])) {
            // reshpae param need to be calc in runtime
            layer_param->num_axes = 0;
        } else {
            const auto shapes     = getValue<std::vector<int64_t>>(node->inputs()[1]);
            layer_param->num_axes = static_cast<int>(shapes.size());
            for (const auto &shape : shapes) {
                layer_param->shape.emplace_back((int)shape);
            }
        }

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
class AddmmTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_INNER_PRODUCT;
        layer_info->type_str                  = "InnerProduct";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[1]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param  = std::make_shared<InnerProductLayerParam>();
        auto layer_res    = new (InnerProductLayerResource);
        const auto weight = inputs[2];
        const auto bias   = inputs[0];

        auto weight_buf = getValue(weight);
        auto shape      = weight_buf.GetBufferDims();
        weight_buf.Permute(shape[0], shape[1]);

        // set param accroding to real value, just test here
        layer_param->name       = layer_info->name;
        layer_param->num_output = shape[1];
        layer_param->axis       = 1;

        layer_res->name          = layer_info->name;
        layer_res->weight_handle = weight_buf;

        auto bias_buf = getValue(bias);
        if (bias_buf.GetBytesSize() != 0) {
            layer_param->has_bias  = 1;
            layer_res->bias_handle = bias_buf;
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

class TransposeTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_PERMUTEV2;
        layer_info->type_str = "PermuteV2";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<PermuteV2LayerParam>();
        layer_param->dim0 = static_cast<int>(getValue<int64_t>(node->inputs()[1]));
        layer_param->dim1 = static_cast<int>(getValue<int64_t>(node->inputs()[2]));

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class UpsampleTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_UPSAMPLE;
        layer_info->type_str = "Upsample";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<UpsampleLayerParam>();
        switch (node->kind()) {
            case at::aten::upsample_nearest2d:
                layer_param->mode = 1;
                if (node->inputs().size() == 3) {
                    auto scales = getValue<std::vector<double>>(node->input(2));
                    layer_param->scales = {(float)scales[1], (float)scales[0]};
                } else if (node->inputs().size() == 4) {
                    layer_param->scales.push_back(getValue<float>(node->input(3)));
                    layer_param->scales.push_back(getValue<float>(node->input(2)));
                }

                break;
            case at::aten::upsample_bilinear2d:
                layer_param->mode = 2;
                if (!toIValue(node->input(1))) {
                    // calc in runtime
                    layer_info->inputs.push_back(node->input(1)->debugName());
                } else {
                    auto output_size = getValue<std::vector<int64_t>>(node->input(1));
                    layer_param->dims = {(int)output_size[0], (int)output_size[1]};
                }
                layer_param->align_corners = getValue<bool>(node->input(2));
                layer_param->scales = {0.f, 0.f};
                if (node->inputs().size() == 5) {
                    layer_param->scales.push_back(getValue<float>(node->input(4)));
                    layer_param->scales.push_back(getValue<float>(node->input(3)));
                }

                break;
            default:
                break;
        } 

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

class ListTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        // only support size + listconstruct, listconstruct + cat
        if (node->inputs().size() == 0) return false;
        auto type = node->inputs().at(0)->type();

        if (type->kind() == c10::TypeKind::IntType) {
            if (node->inputs().at(0)->node()->kind() == c10::aten::size) {
                if (GetGlobalTorchConvertMap().count(node->next()->kind().toQualString()) > 0) {
                    return true;
                }
            }
        } else if (type->kind() == c10::TypeKind::TensorType) {
            if (node->next()->kind() == c10::aten::cat) {
                return true;
            }
        }

        return false;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        auto input_type = node->input(0)->type();
        if (input_type->kind() == c10::TypeKind::TensorType) {
            return TNN_OK;
        }

        if (input_type->kind() == c10::TypeKind::IntType) {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_CONCAT;
            layer_info->type_str                  = "Concat";
            layer_info->name                      = node->output(0)->debugName();

            const auto inputs      = node->inputs();
            const auto tensor_list = inputs[0];
            for (const auto input : inputs) {
                layer_info->inputs.push_back(input->debugName());
            }
            layer_info->outputs.push_back(node->outputs()[0]->debugName());

            auto layer_param  = std::make_shared<ConcatLayerParam>();
            layer_param->axis = 0;
            layer_info->param = layer_param;

            for (const auto &input : inputs) {
                if (!toIValue(input)) continue;
                auto const_buf = getValue(input);
                if (const_buf.GetBytesSize() > 0) {
                    if (*(const_buf.force_to<int *>()) != INT_MAX) {
                        const_buf.SetBufferDims({1});
                        net_resource->constant_map[input->debugName()] = std::make_shared<RawBuffer>(const_buf);
                    }
                }
            }

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
        }

        return TNN_OK;
    }
};

class ListUnpackTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
	    return node->inputs().at(0)->node()->kind() == c10::aten::split;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
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

class FakeQuantizePerTensorAffineTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        //add quantize layer
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_QUANTIZE;
            layer_info->type_str                  = "Quantize";
            layer_info->name                      = node->output(0)->debugName();
    
            const auto input      = node->inputs()[0];
            auto scale_buf = getValue(node->inputs()[1]);
            
            auto layer_res = new(QuantizeLayerResource);
    
            layer_res->scale_handle = ConvertHalfHandle(scale_buf);
            layer_info->inputs.push_back(input->debugName());
    
            layer_info->outputs.push_back(node->outputs()[0]->debugName());
    
            auto layer_param  = std::make_shared<QuantizeLayerParam>();
            layer_param->axis = 0;
            layer_info->param = layer_param;
    
            net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);
    
            ADD_INPUTS_AND_OUTPUTS;
    
            net_structure->layers.push_back(layer_info);
        }

        //add dequantize layer
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_DEQUANTIZE;
            layer_info->type_str                  = "Dequantize";
            layer_info->name                      = node->output(0)->debugName() + "DQ";
    
            const auto input      = node->outputs()[0];
            auto scale_buf = getValue(node->inputs()[1]);
            
            auto layer_res = new(QuantizeLayerResource);
    
            layer_res->scale_handle = ConvertHalfHandle(scale_buf);
            
            layer_info->inputs.push_back(input->debugName());
    
            layer_info->outputs.push_back(node->outputs()[0]->debugName() + "DQ");
    
            auto layer_param  = std::make_shared<QuantizeLayerParam>();
            layer_param->axis = 0;
            layer_info->param = layer_param;
    
            net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);
    
            ADD_INPUTS_AND_OUTPUTS;
    
            net_structure->layers.push_back(layer_info);
        }
        return TNN_OK;
    }
};


class FakeQuantizePerChannelAffineTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        //add quantize layer
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_QUANTIZE;
            layer_info->type_str                  = "Quantize";
            layer_info->name                      = node->output(0)->debugName();
    
            const auto input      = node->inputs()[0];
            auto scale_buf = getValue(node->inputs()[1]);
            
            auto layer_res = new(QuantizeLayerResource);
    
            layer_res->scale_handle = ConvertHalfHandle(scale_buf);
            layer_info->inputs.push_back(input->debugName());
    
            layer_info->outputs.push_back(node->outputs()[0]->debugName());
    
            auto layer_param  = std::make_shared<QuantizeLayerParam>();
            layer_param->axis = static_cast<int>(getValue<int64_t>(node->inputs()[3]));
            layer_info->param = layer_param;
    
            net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);
    
            ADD_INPUTS_AND_OUTPUTS;
    
            net_structure->layers.push_back(layer_info);
        }

        //add dequantize layer
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_DEQUANTIZE;
            layer_info->type_str                  = "Dequantize";
            layer_info->name                      = node->output(0)->debugName() + "DQ";
    
            const auto input      = node->outputs()[0];
            auto scale_buf = getValue(node->inputs()[1]);
            
            auto layer_res = new(QuantizeLayerResource);
    
            layer_res->scale_handle = ConvertHalfHandle(scale_buf);
            
            layer_info->inputs.push_back(input->debugName());
    
            layer_info->outputs.push_back(node->outputs()[0]->debugName() + "DQ");
    
            auto layer_param  = std::make_shared<QuantizeLayerParam>();
            layer_param->axis = static_cast<int>(getValue<int64_t>(node->inputs()[3]));
            layer_info->param = layer_param;
    
            net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);
    
            ADD_INPUTS_AND_OUTPUTS;
    
            net_structure->layers.push_back(layer_info);
        }
        return TNN_OK;
    }
};


REGISTER_TORCH_OP_CONVERTER(Addmm, aten, addmm)
REGISTER_TORCH_OP_CONVERTER(AvgPool, aten, avg_pool2d)
REGISTER_TORCH_OP_CONVERTER(BatchNorm, aten, batch_norm)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, add_)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, add)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, mul)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, div)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, gt)
REGISTER_TORCH_OP_CONVERTER(Concat, aten, cat)
REGISTER_TORCH_OP_CONVERTER(Conv2D, aten, conv2d)
REGISTER_TORCH_OP_CONVERTER(_Conv, aten, _convolution)
REGISTER_TORCH_OP_CONVERTER(Flatten, aten, flatten)
REGISTER_TORCH_OP_CONVERTER(Gather, aten, select)
REGISTER_TORCH_OP_CONVERTER(Gelu, aten, gelu)
REGISTER_TORCH_OP_CONVERTER(HardTanh, aten, hardtanh_)
REGISTER_TORCH_OP_CONVERTER(HardSigmoid, aten, hardsigmoid_)
REGISTER_TORCH_OP_CONVERTER(HardSwish, aten, hardswish_)
REGISTER_TORCH_OP_CONVERTER(LayerNorm, aten, layer_norm)
REGISTER_TORCH_OP_CONVERTER(Linear, aten, linear)
REGISTER_TORCH_OP_CONVERTER(MatMul, aten, matmul)
REGISTER_TORCH_OP_CONVERTER(Permute, aten, permute)
REGISTER_TORCH_OP_CONVERTER(Pool, aten, adaptive_avg_pool2d)
REGISTER_TORCH_OP_CONVERTER(Pool, aten, max_pool2d)
REGISTER_TORCH_OP_CONVERTER(Relu, aten, relu)
REGISTER_TORCH_OP_CONVERTER(Relu, aten, relu_)
REGISTER_TORCH_OP_CONVERTER(Reshape, aten, reshape)
REGISTER_TORCH_OP_CONVERTER(Reshape, aten, view)
REGISTER_TORCH_OP_CONVERTER(Sigmoid, aten, sigmoid)
REGISTER_TORCH_OP_CONVERTER(Size, aten, size)
REGISTER_TORCH_OP_CONVERTER(Softmax, aten, softmax)
REGISTER_TORCH_OP_CONVERTER(Split, aten, split)
REGISTER_TORCH_OP_CONVERTER(StridedSlice, aten, slice)
REGISTER_TORCH_OP_CONVERTER(To, aten, to)
REGISTER_TORCH_OP_CONVERTER(Transpose, aten, transpose)
// REGISTER_TORCH_OP_CONVERTER(Upsample, aten, upsample_bilinear2d)
REGISTER_TORCH_OP_CONVERTER(Upsample, aten, upsample_nearest2d)
REGISTER_TORCH_OP_CONVERTER(Unsqueeze, aten, unsqueeze)


REGISTER_TORCH_OP_CONVERTER(List, prim, ListConstruct)
REGISTER_TORCH_OP_CONVERTER(ListUnpack, prim, ListUnpack)
REGISTER_TORCH_OP_CONVERTER(FakeQuantizePerChannelAffine, aten, fake_quantize_per_channel_affine)
REGISTER_TORCH_OP_CONVERTER(FakeQuantizePerTensorAffine, aten, fake_quantize_per_tensor_affine)

// REGISTER_TORCH_OP_CONVERTER(QuantConv2D, quantized, conv2d)

} // namespace conversion
}

