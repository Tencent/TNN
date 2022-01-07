#include "tnn/network/torch/torch_op_converter.h"
#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>

namespace TNN_NS {

#define BLOB_SCALE_SUFFIX "_scale_data_"

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
            case at::aten::sub:
                layer_info->type     = LAYER_SUB;
                layer_info->type_str = "Sub";
                break;
            case at::aten::mul:
                layer_info->type     = LAYER_MUL;
                layer_info->type_str = "Mul";
                break;
            case at::aten::div:
            case at::aten::floordiv:
                layer_info->type     = LAYER_DIV;
                layer_info->type_str = "Div";
                break;
            case at::aten::gt:
                layer_info->type     = LAYER_GREATER;
                layer_info->type_str = "Greater";
                break;
            case at::aten::eq:
                layer_info->type     = LAYER_EQUAL;
                layer_info->type_str = "Equal";
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
        // aten::linear include batched cases, which is not supported by Inn
        // Convert aten::linear to matmul + add (with bias), matmul only (without bias)
        const auto& inputs = node->inputs();
        const auto weight = inputs[1];
        const auto bias = inputs[2];
        auto weight_buf = getValue(weight);
        auto bias_buf = getValue(bias);
        const auto data_type = weight_buf.GetDataType();
        const bool with_bias = bias_buf.GetBytesSize()!=0;
        std::string matmul_out_name = with_bias ? node->output(0)->debugName()+"_matmul" : node->output(0)->debugName();

        // Matmul layer
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type = LAYER_MATMUL;
            layer_info->type_str = "Matmul";
            layer_info->name = matmul_out_name;

            layer_info->inputs.push_back(node->inputs()[0]->debugName());
            layer_info->outputs.push_back(matmul_out_name);

            const int dim0 = weight_buf.GetBufferDims()[0];
            const int dim1 = weight_buf.GetBufferDims()[1];
            RawBuffer transposed_weight_buf;
            // TODO: Naive 2D weight Transpose here, replace this one with a new faster one in the future.
            if (data_type==DATA_TYPE_HALF) {
                auto *weight_ptr = weight_buf.force_to<fp16_t *>();
                const int weight_byte_size = sizeof(fp16_t)*dim0*dim1;
                fp16_t *temp_weight_ptr = (fp16_t*)std::malloc(weight_byte_size);
                for (int i=0; i<dim0; i++) {
                    for (int j=0; j<dim1; j++) {
                        temp_weight_ptr[j*dim0+i] = weight_ptr[i*dim1+j];
                    }
                }
                std::memcpy(weight_ptr, temp_weight_ptr, weight_byte_size);
                transposed_weight_buf = RawBuffer(weight_byte_size, (char*)(weight_ptr), {dim1, dim0});
                transposed_weight_buf.SetDataType(DATA_TYPE_HALF);
                std::free(temp_weight_ptr);
            } else {
                // FLOAT
                auto *weight_ptr = weight_buf.force_to<float *>();
                const int weight_byte_size = sizeof(float)*dim0*dim1;
                float *temp_weight_ptr = (float*)std::malloc(weight_byte_size);
                for (int i=0; i<dim0; i++) {
                    for (int j=0; j<dim1; j++) {
                        temp_weight_ptr[j*dim0+i] = weight_ptr[i*dim1+j];
                    }
                }
                std::memcpy(weight_ptr, temp_weight_ptr, weight_byte_size);
                transposed_weight_buf = RawBuffer(weight_byte_size, (char*)(weight_ptr), {dim1, dim0});
                std::free(temp_weight_ptr);
            }

            auto layer_res = new MatMulLayerResource();
            layer_res->weight = transposed_weight_buf;

            auto layer_param = std::make_shared<MatMulLayerParam>();
            layer_param->weight_position = 1;
            layer_param->matrix_b_dims = transposed_weight_buf.GetBufferDims();
            layer_info->param = layer_param;

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
            net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);
        } // Matmul

        // add bias if needed.
        if (with_bias) {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type = LAYER_ADD;
            layer_info->type_str = "Add";
            layer_info->name = node->output(0)->debugName();

            // bias->node()->kind() == at::prim::Constant, weight here refers to "bias" of linear.
            auto layer_param = std::make_shared<MultidirBroadcastLayerParam>();
            layer_param->weight_input_index = 1;
            layer_info->param = layer_param;
            layer_info->inputs.push_back(matmul_out_name);
            layer_info->outputs.push_back(node->outputs()[0]->debugName());

            auto layer_res = new EltwiseLayerResource();
            net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

            auto bias_buf = getValue(bias);
            layer_res->element_handle = bias_buf;
            layer_res->element_shape  = bias_buf.GetBufferDims();

            net_structure->layers.push_back(layer_info);

            ADD_INPUTS_AND_OUTPUTS;
        }

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
        if (node->inputs().size() == 2) {
            // aten::size(%in_tensor, %dim)
            for (int i = 0; i < node->output()->uses().size(); i++) {
                if (node->output()->uses()[i].user->kind() != at::prim::ListConstruct) {
                    return false;
                } else {
                    auto& converter = GetGlobalTorchConvertMap()["prim::ListConstruct"];
                    if (!converter->IsSupported(node->output()->uses()[i].user)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        // generate shape layer
        {
            std::string shape_out_name = node->inputs().size() == 2 ? node->output(0)->debugName() + "_shape" : node->output(0)->debugName();

            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_SHAPE;
            layer_info->type_str                  = "Shape";
            layer_info->name                      = shape_out_name;

            layer_info->inputs.push_back(node->inputs()[0]->debugName());
            layer_info->outputs.push_back(shape_out_name);

            layer_info->param = std::make_shared<LayerParam>();

            ADD_INPUTS_AND_OUTPUTS;

            net_structure->layers.push_back(layer_info);
        }

        if (node->inputs().size() == 2) {
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
        auto unpack_node = node->output(0)->uses()[0].user;
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
        const auto& input1_type = inputs[1]->type()->kind();
        if (input1_type == c10::TypeKind::ScalarTypeType || input1_type == c10::TypeKind::IntType) {
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

// func: upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor
// func: upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor
// func: upsample_bilinear2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
// func: upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
class UpsampleTorchConverter : public TorchOpConverter {
public:
    bool IsSupported(const torch::jit::Node *node) {
        // in this mode, upsample param dims will be calc runtime
        // Todo: trt shape tensor should expand hw tensor to nchw tensor
        if (!toIValue(node->input(1))) {
            return false;
        }
        return true;
    }

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
                    if (!toIValue(node->input(1))) {
                        layer_info->inputs.push_back(node->input(1)->debugName() + "_roi");
                        layer_info->inputs.push_back(node->input(1)->debugName() + "_scale");
                        layer_info->inputs.push_back(node->input(1)->debugName());
                        layer_param->scales = {0.f, 0.f};
                        // empty raw buffer just makes tnn not crash
                        net_resource->constant_map[layer_info->inputs[1]] = std::make_shared<RawBuffer>();
                        net_resource->constant_map[layer_info->inputs[2]] = std::make_shared<RawBuffer>();
                    } else {
                        layer_param->scales.push_back(getValue<float>(node->input(3)));
                        layer_param->scales.push_back(getValue<float>(node->input(2)));
                    }
                }

                break;
            case at::aten::upsample_bilinear2d:
                layer_param->mode = 2;
                if (!toIValue(node->input(1))) {
                    // calc in runtime
                    layer_info->inputs.push_back(node->input(1)->debugName() + "_roi");
                    layer_info->inputs.push_back(node->input(1)->debugName() + "_scale");
                    layer_info->inputs.push_back(node->input(1)->debugName());
                    layer_param->scales = {0.f, 0.f};
                    // empty raw buffer just makes tnn not crash
                    net_resource->constant_map[layer_info->inputs[1]] = std::make_shared<RawBuffer>();
                    net_resource->constant_map[layer_info->inputs[2]] = std::make_shared<RawBuffer>();
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

// func: mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
class ReduceTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type_str = "ReduceMean";
        layer_info->name = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<ReduceLayerParam>();
        auto axis = getValue<std::vector<int64_t>>(node->inputs()[1]);
        for(auto value : axis) {
            layer_param->axis.push_back(value);
        }
        layer_param->keep_dims = getValue<bool>(node->inputs()[2]);

        switch (node->kind()) {
            case at::aten::mean:
                layer_info->type = LAYER_REDUCE_MEAN;                 
            default: 
                break;
        }

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: reflection_pad2d(Tensor self, int[4] padding) -> Tensor
class ReflectionPadTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_PAD;
        layer_info->type_str                  = "Pad";
        layer_info->name                      = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param  = std::make_shared<PadLayerParam>();
        layer_param->type = 1;
        const auto pads   = getValue<std::vector<int64_t>>(node->input(1));
        layer_param->pads = {(int)(pads[2]), (int)(pads[3]), (int)(pads[0]), (int)(pads[1]), 0, 0};

        auto t            = layer_param->pads;
        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
class ClipTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_CLIP;
        layer_info->type_str                  = "Clip";
        layer_info->name                      = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param       = std::make_shared<ClipLayerParam>();
        bool min_value_is_none = node->input(1)->type()->kind() == c10::TypeKind::NoneType;
        layer_param->min       = min_value_is_none ? -FLT_MAX : getValue<float>(node->input(1));

        bool max_value_is_none = node->input(2)->type()->kind() == c10::TypeKind::NoneType;
        layer_param->max       = max_value_is_none ? FLT_MAX : getValue<float>(node->input(2));

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

// func: clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
class PowerTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_POWER;
        layer_info->type_str                  = "Power";
        layer_info->name                      = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param         = std::make_shared<PowLayerParam>();
        const auto exponent_type = node->input(1)->type()->kind();
        switch (exponent_type) {
            case c10::TypeKind::IntType:
                layer_param->exponent = static_cast<float>(getValue<int>(node->input(1)));
                break;
            case c10::TypeKind::FloatType:
                layer_param->exponent = getValue<float>(node->input(1));
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

// func: topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
class TopKTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_TOPK;
        layer_info->type_str                  = "TopK";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[1]->debugName());

        auto layer_param = std::make_shared<TopKLayerParam>();

        layer_param->k       = static_cast<int>(getValue<int64_t>(inputs[1]));
        layer_param->axis    = static_cast<int>(getValue<int64_t>(inputs[2]));
        layer_param->largest = static_cast<int>(getValue<bool>(inputs[3]));
        layer_param->sorted  = static_cast<int>(getValue<bool>(inputs[4]));

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
            auto user_type_str = node->output(0)->uses()[0].user->kind().toQualString();
            if (GetGlobalTorchConvertMap().count(user_type_str)) {
                auto& converter = GetGlobalTorchConvertMap()[user_type_str];
                if (converter->IsSupported(node->output(0)->uses()[0].user)) {
                    return true;
                }
            }
        } else if (type->kind() == c10::TypeKind::TensorType) {
            if (node->output(0)->uses()[0].user->kind() == c10::aten::cat) {
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

        torch::jit::Node* in0_node = node->input(0)->node();
        if (in0_node->kind() == c10::aten::split) {
            return true;
        } else if (in0_node->kind() == c10::aten::size) {
            if (in0_node->inputs().size() == 1) {
                // aten::size(%in_tensor), return a list representing shape of the Tensor.
                return true;
            }
        }
        return false;
    }

    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        torch::jit::Node* in0_node = node->input(0)->node();

        if (in0_node->kind() == c10::aten::size && in0_node->inputs().size() == 1) {
            const auto input = node->outputs();
            const auto outputs = node->outputs();
            const int num_dims = outputs.size();
            
            for (int dim=0; dim<num_dims; dim++) {
                std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
                layer_info->type                 = LAYER_GATHER;
                layer_info->type_str             = "Gather";
                layer_info->name                 = outputs[dim]->debugName();

                layer_info->inputs.push_back(node->input(0)->debugName());
                layer_info->outputs.push_back(outputs[dim]->debugName());

                auto layer_param                 = std::make_shared<GatherLayerParam>();
                layer_param->axis                = 0;
                layer_param->indices_in_resource = true;

                layer_info->param  = layer_param;

                int indices        = dim;
                auto layer_res     = std::make_shared<GatherLayerResource>();
                auto indices_buf   = RawBuffer(sizeof(int), reinterpret_cast<char *>(&indices), {1});
                indices_buf.SetDataType(DATA_TYPE_INT32);
                layer_res->indices = indices_buf;

                ADD_INPUTS_AND_OUTPUTS;

                net_structure->layers.push_back(layer_info);
                net_resource->resource_map[layer_info->name] = layer_res;
            }
        }
        
        return TNN_OK;
    }
};

class SqueezeTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_SQUEEZE;
        layer_info->type_str                  = "Squeeze";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<SqueezeLayerParam>();

        layer_param->axes = {static_cast<int>(getValue<int64_t>(inputs[1]))};

        layer_info->param = layer_param;

        ADD_INPUTS_AND_OUTPUTS;

        net_structure->layers.push_back(layer_info);

        return TNN_OK;
    }
};

std::string find_propagate_scale(const torch::jit::Node *node) {
    std::string name                            = node->kind().toQualString();
    std::unordered_set<std::string> noscale_ops = {
        "aten::adaptive_avg_pool2d",
        "aten::cat",
        "aten::flatten",
        "aten::max_pool2d",
        "prim::ListConstruct",
        "prim::ListUnpack"};

    std::unordered_set<std::string> scale_ops = {
        "aten::quantize_per_tensor",
        "quantized::linear",
        "quantized::conv2d",
        "quantized::conv2d_relu",
        "quantized::add_relu"};

    if (scale_ops.find(name) != scale_ops.end()) {
        return node->outputs()[0]->debugName();
    } else if (noscale_ops.find(name) != noscale_ops.end()) {
        return find_propagate_scale(node->inputs()[0]->node());
    }

    return "";
}

RawBuffer merge_weight_scale(RawBuffer &i_scale_buf, RawBuffer &w_scale_buf) {
    const auto i_scale_len    = i_scale_buf.GetDataCount();
    const auto w_scale_len    = w_scale_buf.GetDataCount();
    const auto i_scale        = i_scale_buf.force_to<float *>();
    const auto w_scale        = w_scale_buf.force_to<float *>();
    auto merge_scale_len      = MAX(i_scale_len, w_scale_len);
    RawBuffer merge_scale_buf = RawBuffer(merge_scale_len * sizeof(float));
    for (int i = 0; i < merge_scale_len; i++) {
        int i_scale_idx = i_scale_len == 1 ? 0 : i;
        int w_scale_idx = w_scale_len == 1 ? 0 : i;
        merge_scale_buf.force_to<float *>()[i] = i_scale[i_scale_idx] * w_scale[w_scale_idx];
    }
    return merge_scale_buf;
}

RawBuffer quant_bias(RawBuffer &bias, RawBuffer &scale) {
    RawBuffer bias_quant = RawBuffer(bias.GetDataCount() * sizeof(int32_t));
    bias_quant.SetDataType(DATA_TYPE_INT32);
    const auto bias_len  = bias.GetDataCount();
    const auto scale_len = scale.GetDataCount();
    const auto bias_ptr  = bias.force_to<float *>();
    const auto scale_ptr = scale.force_to<float *>();

    for (int i = 0; i < bias_len; i++) {
        int scale_idx = scale_len == 1 ? 0 : i;
        bias_quant.force_to<int32_t *>()[i] = static_cast<int32_t>(std::nearbyint(bias_ptr[i] / scale_ptr[scale_idx]));
    }

    return bias_quant;
}

void add_blob_scale_resource(RawBuffer &scale_buf, IntScaleResource *layer_res) {
    layer_res->scale_handle = scale_buf;
    auto zero_point_handle = RawBuffer(scale_buf.GetDataCount() * sizeof(int8_t));
    zero_point_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
    layer_res->zero_point_handle = zero_point_handle;
}

// func: quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor
class QuantTensorTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_REFORMAT;
        layer_info->type_str                  = "Reformat";
        layer_info->name                      = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param      = std::make_shared<ReformatLayerParam>();
        layer_param->src_type = DATA_TYPE_FLOAT;
        layer_param->dst_type = DATA_TYPE_INT8;

        layer_info->param = layer_param;

        auto scale_buf          = getValue(node->inputs().at(1));
        auto layer_res          = new (IntScaleResource);
        add_blob_scale_resource(scale_buf, layer_res);

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[node->output(0)->debugName() + BLOB_SCALE_SUFFIX] =
            std::shared_ptr<LayerResource>(layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

class DequantTensorTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_REFORMAT;
        layer_info->type_str                  = "Reformat";
        layer_info->name                      = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param      = std::make_shared<ReformatLayerParam>();
        layer_param->src_type = DATA_TYPE_INT8;
        layer_param->dst_type = DATA_TYPE_FLOAT;

        layer_info->param = layer_param;
        net_structure->layers.push_back(layer_info);

        if (net_resource->resource_map.find(node->input(0)->debugName() + BLOB_SCALE_SUFFIX) ==
            net_resource->resource_map.end()) {
            // dequant op always have input scale
        }

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

//
class QuantConv2DTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_CONVOLUTION;
        layer_info->type_str                  = "QuantizedConvolution";
        layer_info->name                      = node->output(0)->debugName();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        // get conv packed param
        const auto conv_value        = toIValue(node->input(1)).value();
        const auto slots             = conv_value.toObject().get()->slots();
        const auto conv_packed_param = reinterpret_cast<ConvPackedParamsBase<2> *>(slots[0].toCapsule().get());

        // unpack conv packed param
        const auto stride    = conv_packed_param->stride();
        const auto padding   = conv_packed_param->padding();
        const auto dialation = conv_packed_param->dilation();
        const auto group     = conv_packed_param->groups();
        const auto transpose = conv_packed_param->transpose();

        const auto weight_and_bias = conv_packed_param->unpack();
        const auto weight          = std::get<0>(weight_and_bias);
        const auto bias            = std::get<1>(weight_and_bias);

        auto layer_param = std::make_shared<ConvLayerParam>();
        auto layer_res   = new (ConvLayerResource);

        layer_param->quantized = true;

        // set input scale res
        if (net_resource->resource_map.find(node->input(0)->debugName() + BLOB_SCALE_SUFFIX) ==
            net_resource->resource_map.end()) {
            auto propagate_input = find_propagate_scale(node->inputs()[0]->node());
            net_resource->resource_map[node->input(0)->debugName() + BLOB_SCALE_SUFFIX] =
                net_resource->resource_map[propagate_input + BLOB_SCALE_SUFFIX];
        }
        auto i_scale_buf = dynamic_cast<IntScaleResource *>(
                               net_resource->resource_map[node->input(0)->debugName() + BLOB_SCALE_SUFFIX].get())
                               ->scale_handle;

        // convert to tnn param&res
        layer_res->filter_handle = getValue(weight);

        RawBuffer w_scale_buf;
        if (weight.qscheme() == c10::kPerChannelAffine || weight.qscheme() == c10::kPerChannelSymmetric) {
            const auto q_scale = weight.q_per_channel_scales();
            w_scale_buf        = getValue(q_scale);
        } else if (weight.qscheme() == c10::kPerTensorAffine || weight.qscheme() == c10::kPerTensorSymmetric) {
            const auto q_scale               = (float)weight.q_scale();
            w_scale_buf                      = RawBuffer(4);
            *w_scale_buf.force_to<float *>() = q_scale;
        }
        layer_res->scale_handle = merge_weight_scale(i_scale_buf, w_scale_buf);
        auto zero_point_handle = RawBuffer(w_scale_buf.GetDataCount() * sizeof(int8_t));
        zero_point_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
        layer_res->zero_point_handle = zero_point_handle;

        auto shape                  = layer_res->filter_handle.GetBufferDims();
        layer_param->name           = layer_info->name;
        layer_param->pad_type       = -1;
        layer_param->output_channel = shape[0];
        layer_param->input_channel  = shape[1];
        layer_param->kernels        = {shape[3], shape[2]};
        layer_param->dialations     = {(int)dialation[1], (int)dialation[0]};
        layer_param->strides        = {(int)stride[1], (int)stride[0]};
        layer_param->pads           = {(int)padding[1], (int)padding[1], (int)padding[0], (int)padding[0]};
        layer_param->group          = group;
        std::string type_str        = node->kind().toQualString();
        if (type_str.find("relu") != std::string::npos) {
            layer_param->activation_type = ActivationType_ReLU;
        }

        if (bias.has_value()) {
            layer_param->bias      = 1;
            auto bias_buf_float    = getValue(bias.value());
            layer_res->bias_handle = quant_bias(bias_buf_float, layer_res->scale_handle);
        }
        layer_info->param = layer_param;

        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        // set output scale res
        auto o_scale_buf                = getValue(node->inputs().at(2));
        auto o_scale_layer_res          = new (IntScaleResource);
        add_blob_scale_resource(o_scale_buf, o_scale_layer_res);

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[node->output(0)->debugName() + BLOB_SCALE_SUFFIX] =
            std::shared_ptr<LayerResource>(o_scale_layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

class QuantLinearTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type                      = LAYER_INNER_PRODUCT;
        layer_info->type_str                  = "QuantizedInnerProduct";
        layer_info->name                      = node->output(0)->debugName();

        const auto &inputs = node->inputs();

        layer_info->inputs.push_back(node->inputs()[0]->debugName());
        layer_info->outputs.push_back(node->outputs()[0]->debugName());

        auto layer_param = std::make_shared<InnerProductLayerParam>();
        auto layer_res   = new (InnerProductLayerResource);

        layer_param->quantized = true;

        // set input scale res
        if (net_resource->resource_map.find(node->input(0)->debugName() + BLOB_SCALE_SUFFIX) ==
            net_resource->resource_map.end()) {
            auto propagate_input = find_propagate_scale(node->inputs()[0]->node());
            net_resource->resource_map[node->input(0)->debugName() + BLOB_SCALE_SUFFIX] =
                net_resource->resource_map[propagate_input + BLOB_SCALE_SUFFIX];
        }
        auto i_scale_buf = dynamic_cast<IntScaleResource *>(
                               net_resource->resource_map[node->input(0)->debugName() + BLOB_SCALE_SUFFIX].get())
                               ->scale_handle;

        const auto weight_value        = toIValue(node->input(1)).value();
        const auto slots               = weight_value.toObject().get()->slots();
        const auto weight_packed_param = reinterpret_cast<LinearPackedParamsBase *>(slots[0].toCapsule().get());

        const auto weight_unpack = weight_packed_param->unpack();
        const auto weight        = std::get<0>(weight_unpack);
        const auto bias          = std::get<1>(weight_unpack);

        auto weight_buf = getValue(weight);
        auto shape      = weight_buf.GetBufferDims();

        RawBuffer w_scale_buf;
        if (weight.qscheme() == c10::kPerChannelAffine || weight.qscheme() == c10::kPerChannelSymmetric) {
            const auto q_scale = weight.q_per_channel_scales();
            w_scale_buf        = getValue(q_scale);
        } else if (weight.qscheme() == c10::kPerTensorAffine || weight.qscheme() == c10::kPerTensorSymmetric) {
            const auto q_scale               = (float)weight.q_scale();
            w_scale_buf                      = RawBuffer(4);
            *w_scale_buf.force_to<float *>() = q_scale;
        }
        layer_res->scale_handle = merge_weight_scale(i_scale_buf, w_scale_buf);
        auto zero_point_handle = RawBuffer(w_scale_buf.GetDataCount() * sizeof(int8_t));
        zero_point_handle.SetDataType(TNN_NS::DATA_TYPE_INT8);
        layer_res->zero_point_handle = zero_point_handle;

        // set param accroding to real value, just test here
        layer_param->name       = layer_info->name;
        layer_param->num_output = shape[0];
        layer_param->axis       = 1;

        layer_res->name          = layer_info->name;
        layer_res->weight_handle = weight_buf;

        if (bias.has_value()) {
            layer_param->has_bias  = 1;
            auto bias_buf_float    = getValue(bias.value());
            layer_res->bias_handle = quant_bias(bias_buf_float, layer_res->scale_handle);
        }

        layer_info->param = layer_param;

        net_structure->layers.push_back(layer_info);
        net_resource->resource_map[layer_info->name] = std::shared_ptr<LayerResource>(layer_res);

        // set output scale res
        auto o_scale_buf                = getValue(node->inputs().at(2));
        float *o_scale_value            = o_scale_buf.force_to<float *>();
        auto o_scale_layer_res          = new (IntScaleResource);
        add_blob_scale_resource(o_scale_buf, o_scale_layer_res);

        net_resource->resource_map[node->output(0)->debugName() + BLOB_SCALE_SUFFIX] =
            std::shared_ptr<LayerResource>(o_scale_layer_res);

        ADD_INPUTS_AND_OUTPUTS;

        return TNN_OK;
    }
};

class QuantAddReluTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        // get output scale
        auto o_scale_buf                = getValue(node->inputs().at(2));
        float *o_scale_value            = o_scale_buf.force_to<float *>();
        auto o_scale_layer_res          = new (IntScaleResource);
        add_blob_scale_resource(o_scale_buf, o_scale_layer_res);

        // generate quant add layer
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_ADD;
            layer_info->type_str                  = "QuantizedAdd";
            layer_info->name                      = node->output(0)->debugName() + "_add";

            layer_info->inputs.push_back(node->inputs()[0]->debugName());
            layer_info->inputs.push_back(node->inputs()[1]->debugName());
            layer_info->outputs.push_back(node->outputs()[0]->debugName() + "_add");

            auto layer_param                = std::make_shared<MultidirBroadcastLayerParam>();
            layer_param->quantized          = true;
            layer_param->weight_input_index = -1;
            layer_info->param               = layer_param;
            net_structure->layers.push_back(layer_info);

            // set input scale
            if (net_resource->resource_map.find(node->input(0)->debugName() + BLOB_SCALE_SUFFIX) ==
                net_resource->resource_map.end()) {
                auto propagate_input = find_propagate_scale(node->inputs()[0]->node());
                net_resource->resource_map[node->input(0)->debugName() + BLOB_SCALE_SUFFIX] =
                    net_resource->resource_map[propagate_input + BLOB_SCALE_SUFFIX];
            }
            if (net_resource->resource_map.find(node->input(1)->debugName() + BLOB_SCALE_SUFFIX) ==
                net_resource->resource_map.end()) {
                auto propagate_input = find_propagate_scale(node->inputs()[0]->node());
                net_resource->resource_map[node->input(1)->debugName() + BLOB_SCALE_SUFFIX] =
                    net_resource->resource_map[propagate_input + BLOB_SCALE_SUFFIX];
            }

            // set output scale
            net_resource->resource_map[node->outputs()[0]->debugName() + "_add" + BLOB_SCALE_SUFFIX] =
                std::shared_ptr<LayerResource>(o_scale_layer_res);

            ADD_INPUTS_AND_OUTPUTS;
        }

        // generate quant relu layer
        {
            std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
            layer_info->type                      = LAYER_RELU;
            layer_info->type_str                  = "QuantizedReLU";
            layer_info->name                      = node->output(0)->debugName();

            layer_info->inputs.push_back(node->outputs()[0]->debugName() + "_add");
            layer_info->outputs.push_back(node->outputs()[0]->debugName());

            layer_info->param            = std::make_shared<LayerParam>();
            layer_info->param->quantized = true;
            net_structure->layers.push_back(layer_info);

            // set output scale
            net_resource->resource_map[node->outputs()[0]->debugName() + BLOB_SCALE_SUFFIX] =
                net_resource->resource_map[node->outputs()[0]->debugName() + "_add" + BLOB_SCALE_SUFFIX];

            ADD_INPUTS_AND_OUTPUTS;
        }

        return TNN_OK;
    }
};

// func: conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor
class ConvTransposeTorchConverter : public TorchOpConverter {
public:
    Status Convert(const torch::jit::Node *node, NetStructure *net_structure, NetResource *net_resource) {
        std::shared_ptr<LayerInfo> layer_info = std::make_shared<LayerInfo>();
        layer_info->type = LAYER_DECONVOLUTION;
        layer_info->type_str = "Deconvolution";
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
        const auto dialation = getValue<std::vector<int64_t>>(inputs[7]);
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

REGISTER_TORCH_OP_CONVERTER(Addmm, aten, addmm)
REGISTER_TORCH_OP_CONVERTER(AvgPool, aten, avg_pool2d)
REGISTER_TORCH_OP_CONVERTER(BatchNorm, aten, batch_norm)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, add_)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, add)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, mul)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, div)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, floordiv)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, eq)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, gt)
REGISTER_TORCH_OP_CONVERTER(Clip, aten, clamp)
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
REGISTER_TORCH_OP_CONVERTER(Power, aten, pow)
REGISTER_TORCH_OP_CONVERTER(ReflectionPad, aten, reflection_pad2d)
REGISTER_TORCH_OP_CONVERTER(Relu, aten, relu)
REGISTER_TORCH_OP_CONVERTER(Relu, aten, relu_)
REGISTER_TORCH_OP_CONVERTER(Reshape, aten, reshape)
REGISTER_TORCH_OP_CONVERTER(Reshape, aten, view)
REGISTER_TORCH_OP_CONVERTER(Sigmoid, aten, sigmoid)
REGISTER_TORCH_OP_CONVERTER(Sigmoid, aten, sigmoid_)
REGISTER_TORCH_OP_CONVERTER(Size, aten, size)
REGISTER_TORCH_OP_CONVERTER(Softmax, aten, softmax)
REGISTER_TORCH_OP_CONVERTER(Split, aten, split)
REGISTER_TORCH_OP_CONVERTER(StridedSlice, aten, slice)
REGISTER_TORCH_OP_CONVERTER(To, aten, to)
REGISTER_TORCH_OP_CONVERTER(TopK, aten, topk)
REGISTER_TORCH_OP_CONVERTER(Transpose, aten, transpose)
// REGISTER_TORCH_OP_CONVERTER(Upsample, aten, upsample_bilinear2d)
REGISTER_TORCH_OP_CONVERTER(Upsample, aten, upsample_nearest2d)
REGISTER_TORCH_OP_CONVERTER(Unsqueeze, aten, unsqueeze)
REGISTER_TORCH_OP_CONVERTER(Reduce, aten, mean)

REGISTER_TORCH_OP_CONVERTER(List, prim, ListConstruct)
REGISTER_TORCH_OP_CONVERTER(ListUnpack, prim, ListUnpack)
REGISTER_TORCH_OP_CONVERTER(Squeeze, aten, squeeze)
REGISTER_TORCH_OP_CONVERTER(Binary, aten, sub)

REGISTER_TORCH_OP_CONVERTER(DequantTensor, aten, dequantize)
REGISTER_TORCH_OP_CONVERTER(QuantTensor, aten, quantize_per_tensor)
REGISTER_TORCH_OP_CONVERTER(QuantAddRelu, quantized, add_relu)
REGISTER_TORCH_OP_CONVERTER(QuantConv2D, quantized, conv2d)
REGISTER_TORCH_OP_CONVERTER(QuantConv2D, quantized, conv2d_relu)
REGISTER_TORCH_OP_CONVERTER(QuantLinear, quantized, linear)
REGISTER_TORCH_OP_CONVERTER(ConvTranspose, aten, conv_transpose2d)

} // namespace conversion
}

