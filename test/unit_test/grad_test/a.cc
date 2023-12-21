#include <gtest/gtest.h>

#include "test/unit_test/unit_test_common.h"
#include "tnn/core/tnn.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/layer/base_layer.h"

namespace TNN_NS {

struct Input {
    std::string name;
    DimsVector shape;
    std::vector<float> data;
    DataType dtype;
    bool is_const = false;

    std::vector<int32_t> ints_data;

    Input(const std::string& name_, const DimsVector& shape_, const std::vector<float>& data_,
          DataType dtype_ = DATA_TYPE_FLOAT, bool is_const_ = false)
        : name(name_), shape(shape_), data(data_), dtype(dtype_), is_const(is_const_) {
        if (dtype == DATA_TYPE_INT32) {
            ints_data.reserve(data.size());
            for (float v : data) {
                ints_data.push_back(static_cast<int32_t>(v));
            }
        }
    }
};

struct Layer {
    LayerInfo info;
    std::shared_ptr<LayerResource> resource;
};

std::shared_ptr<AbstractModelInterpreter> CreateInterpreter(
    const std::vector<Input>& net_inputs, const std::set<std::string>& net_outputs, const std::vector<Layer>& layers,
    const std::map<std::string, std::shared_ptr<RawBuffer>>& consts) {
    auto interpreter                             = CreateModelInterpreter(MODEL_TYPE_TNN);
    DefaultModelInterpreter* default_interpreter = dynamic_cast<DefaultModelInterpreter*>(interpreter);
    if (!default_interpreter) {
        LOGE("tnn interpreter is nullptr\n");
        return nullptr;
    }

    // net_structure 只需填充 inputs_shape_map、input_data_type_map、outputs、layers、blobs
    // net_resource 只需要填充resource_map、constant_map
    NetStructure* net_structure = default_interpreter->GetNetStructure();
    NetResource* net_resource   = default_interpreter->GetNetResource();

    net_structure->outputs = net_outputs;

    for (const auto& net_input : net_inputs) {
        net_structure->inputs_shape_map[net_input.name]    = net_input.shape;
        net_structure->input_data_type_map[net_input.name] = net_input.dtype;
    }

    for (const auto& layer : layers) {
        net_structure->layers.push_back(std::make_shared<LayerInfo>(layer.info));
        net_structure->blobs.insert(layer.info.inputs.begin(), layer.info.inputs.end());
        net_structure->blobs.insert(layer.info.outputs.begin(), layer.info.outputs.end());
        if (layer.resource) {
            net_resource->resource_map[layer.info.name] = layer.resource;
        }
    }
    net_resource->constant_map = consts;

    return std::shared_ptr<AbstractModelInterpreter>(interpreter);
}

std::shared_ptr<Instance> CreateInstance(std::shared_ptr<AbstractModelInterpreter> interpreter,
                                         const NetworkConfig& network_config, const ModelConfig& model_config) {
    auto instance = std::make_shared<Instance>(network_config, model_config);

    DefaultModelInterpreter* default_interpreter = dynamic_cast<DefaultModelInterpreter*>(interpreter.get());
    Status status = instance->Init(interpreter, default_interpreter->GetNetStructure()->inputs_shape_map);
    if (status != TNN_OK) {
        LOGE("tnn init device instance failed (%s)\n", status.description().c_str());
        return nullptr;
    }
    return instance;
}

std::shared_ptr<Instance> InitTrainNetwork(const std::vector<Input>& net_inputs,
                                           const std::set<std::string>& net_outputs, const std::vector<Layer>& layers,
                                           const std::map<std::string, std::shared_ptr<RawBuffer>>& consts) {
    auto interpreter = CreateInterpreter(net_inputs, net_outputs, layers, consts);

    // NetworkConfig 配置
    NetworkConfig network_config;
    network_config.device_type = DEVICE_ARM;

    TrainConfig& train_config                = network_config.train_config;
    train_config.run_mode                    = TNN_NS::TRAIN_MODE_TRAIN;
    train_config.train_the_whole_model       = true;
    train_config.solver_params.learning_rate = 0.1;

    return CreateInstance(interpreter, network_config, ModelConfig());
}

// 创建一个Layer的后向测试网络：包含两个layer，一个是要测试的layer，一个是ReduceMean用来产生一个标量作为loss
// 1) 学习率：0.1
// 2) Loss使用ReduceMean产生，则要测试的Layer的 dL/dy = 1 / count(y.dims)
std::shared_ptr<Instance> InitLayerGradTestNetwork(
    const std::string& layer_type, const std::vector<Input>& layer_inputs,
    const std::vector<std::string>& layer_outputs, std::shared_ptr<LayerParam> layer_param,
    std::shared_ptr<LayerResource> layer_resource  = nullptr,
    const std::map<std::string, std::shared_ptr<RawBuffer>>& consts = {}) {
    std::vector<Layer> layers(2);

    // 添加要测试的layer：这个算子的输入就是整个网络的输入
    Layer& layer         = layers.front();
    layer.info.type_str = layer_type;
    layer.info.type     = GlobalConvertLayerType(layer.info.type_str);
    layer.info.name     = "test_layer";
    layer.info.inputs.reserve(layer_inputs.size());
    for (const Input& input : layer_inputs) {
        layer.info.inputs.push_back(input.name);
    }
    layer.info.outputs = layer_outputs;
    layer.info.param   = layer_param;
    layer.resource     = layer_resource;

    // 再加一个ReduceMean算子来产生loss： 这个算子的输出就是整个网络的输出
    Layer& loss_layer         = layers.back();
    loss_layer.info.type_str = "ReduceMean";
    loss_layer.info.type     = GlobalConvertLayerType(loss_layer.info.type_str);
    loss_layer.info.name     = "loss_layer";
    loss_layer.info.inputs   = layer.info.outputs;
    loss_layer.info.outputs.push_back("loss");

    auto reduce_layer_param        = std::shared_ptr<ReduceLayerParam>(new ReduceLayerParam());
    reduce_layer_param->all_reduce = 1;
    loss_layer.info.param          = reduce_layer_param;

    std::set<std::string> net_outputs = {loss_layer.info.outputs.front()};
    return InitTrainNetwork(layer_inputs, net_outputs, layers, consts);
}

std::shared_ptr<Mat> Input2Mat(const Input& input) {
    if (input.dtype == DATA_TYPE_INT32) {
        return std::make_shared<Mat>(DEVICE_ARM, NC_INT32, input.shape, (void*)input.ints_data.data());
    } else if (input.dtype == DATA_TYPE_FLOAT) {
        return std::make_shared<Mat>(DEVICE_ARM, NCHW_FLOAT, input.shape, (void*)input.data.data());
    } else {
        return nullptr;
    }
}

std::shared_ptr<RawBuffer> Input2Rawbuffer(const Input& input) {
    if (input.dtype == DATA_TYPE_INT32) {
        return std::make_shared<RawBuffer>(sizeof(int32_t) * input.ints_data.size(), (char*)input.ints_data.data(),
                                           input.shape, input.dtype);
    } else if (input.dtype == DATA_TYPE_FLOAT) {
        return std::make_shared<RawBuffer>(sizeof(float) * input.data.size(), (char*)input.data.data(), input.shape,
                                           input.dtype);
    } else {
        return nullptr;
    }
}

void DoTest(const std::string& layer_type, const std::vector<Input>& layer_inputs, const std::string& layer_output,
            std::shared_ptr<LayerParam> layer_param, std::shared_ptr<LayerResource> layer_resource,
            std::shared_ptr<LayerResource> layer_resource_expected) {
    // 将输入拆分为Mat输入和const常量输入
    std::map<std::string, std::shared_ptr<Mat>> mat_inputs;
    std::map<std::string, std::shared_ptr<RawBuffer>> net_consts;
    for (const auto& input : layer_inputs) {
        if (input.is_const) {
            net_consts[input.name] = Input2Rawbuffer(input);
        } else {
            mat_inputs[input.name] = Input2Mat(input);
        }
    }

    auto instance =
        InitLayerGradTestNetwork(layer_type, layer_inputs, {layer_output}, layer_param, layer_resource, net_consts);
    ASSERT_TRUE(instance != nullptr) << "tnn instance is nullptr, init network failed";

    TNN_NS::MatConvertParam convert_param;
    for (const auto& [input_name, input_mat] : mat_inputs) {
        instance->SetInputMat(input_mat, convert_param, input_name);
    }
    instance->Forward();
    instance->TrainStep();
}

TEST(grad, grad) {
    printf("Hello world\n");

    std::shared_ptr<MultidirBroadcastLayerParam> layer_param      = std::make_shared<MultidirBroadcastLayerParam>();
    std::shared_ptr<EltwiseLayerResource> layer_resource          = std::make_shared<EltwiseLayerResource>();
    std::shared_ptr<EltwiseLayerResource> layer_resource_expected = std::make_shared<EltwiseLayerResource>();

    // 普通输入 + 普通输入
    layer_param->weight_input_index = -1;
    DoTest("Mul",
           {Input("x", {2, 3}, {1, 2, 3, 4, 5, 6}, DATA_TYPE_FLOAT),
            Input("y", {2, 3}, {7, 8, 9, 10, 11, 12}, DATA_TYPE_FLOAT, true /*is_const*/)},
           "z", layer_param, layer_resource, layer_resource_expected);

    // 普通输入 + 内部参数
    layer_param->weight_input_index = 1;

    // 内部参数 + 普通输入
    layer_param->weight_input_index = 0;

    // 常量输入 + 普通输入
    layer_param->weight_input_index = -1;

    // 普通输入 + 常量输入
    layer_param->weight_input_index = -1;

    // 常量输入 + 常量输入
    layer_param->weight_input_index = -1;
}

}  // namespace TNN_NS
