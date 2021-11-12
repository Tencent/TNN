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

        layer_param->pool_type      = 1;
        layer_param->pad_type       = -1;
        layer_param->kernels_params = {(int)kernel_size[1], (int)kernel_size[0]};
        layer_param->strides        = {(int)stride[1], (int)stride[0]};
        layer_param->pads           = {(int)(padding[1]), (int)(padding[1]), (int)(padding[0]), (int)(padding[0])};
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

class ListTorchConverter : public TorchOpConverter {
public:
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


REGISTER_TORCH_OP_CONVERTER(Conv2D, aten, conv2d)
REGISTER_TORCH_OP_CONVERTER(_Conv, aten, _convolution)
REGISTER_TORCH_OP_CONVERTER(Relu, aten, relu_)
REGISTER_TORCH_OP_CONVERTER(Relu, aten, relu)
REGISTER_TORCH_OP_CONVERTER(Pool, aten, max_pool2d)
REGISTER_TORCH_OP_CONVERTER(AvgPool, aten, avg_pool2d)
REGISTER_TORCH_OP_CONVERTER(Pool, aten, adaptive_avg_pool2d)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, add_)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, add)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, mul)
REGISTER_TORCH_OP_CONVERTER(Flatten, aten, flatten)
REGISTER_TORCH_OP_CONVERTER(Linear, aten, linear)
REGISTER_TORCH_OP_CONVERTER(HardTanh, aten, hardtanh_)
REGISTER_TORCH_OP_CONVERTER(HardSigmoid, aten, hardsigmoid_)
REGISTER_TORCH_OP_CONVERTER(HardSwish, aten, hardswish_)
REGISTER_TORCH_OP_CONVERTER(BatchNorm, aten, batch_norm)
REGISTER_TORCH_OP_CONVERTER(Concat, aten, cat)
REGISTER_TORCH_OP_CONVERTER(Unsqueeze, aten, unsqueeze)
REGISTER_TORCH_OP_CONVERTER(Gather, aten, select)
REGISTER_TORCH_OP_CONVERTER(StridedSlice, aten, slice)
REGISTER_TORCH_OP_CONVERTER(Sigmoid, aten, sigmoid)

REGISTER_TORCH_OP_CONVERTER(List, prim, ListConstruct)

// REGISTER_TORCH_OP_CONVERTER(QuantConv2D, quantized, conv2d)

} // namespace conversion
}
