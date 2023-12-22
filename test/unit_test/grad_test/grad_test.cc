#include "test/unit_test/grad_test/grad_test.h"

namespace TNN_NS {

std::map<std::string, std::shared_ptr<Mat>> GradTest::DoLayerGradTest(const std::string& layer_type,
                                                                      const std::vector<Input>& layer_inputs,
                                                                      const std::string& layer_output,
                                                                      std::shared_ptr<LayerParam> layer_param,
                                                                      std::shared_ptr<LayerResource> layer_resource,
                                                                      const std::set<std::string>& net_outputs) {
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

    auto instance = InitLayerGradTestNetwork(layer_type, layer_inputs, {layer_output}, layer_param, net_outputs,
                                             layer_resource, net_consts);

    TNN_NS::MatConvertParam convert_param;
    for (const auto& [input_name, input_mat] : mat_inputs) {
        instance->SetInputMat(input_mat, convert_param, input_name);
    }
    instance->Forward();
    instance->TrainStep();

    BlobMap blobs;
    instance->GetAllOutputBlobs(blobs);
    return Blob2Mat(instance, blobs);
}

// 创建一个Layer的后向测试网络：包含两个layer，一个是要测试的layer，一个是ReduceMean用来产生一个标量作为loss
// 1) 学习率：0.1
// 2) Loss使用ReduceMean产生，则要测试的Layer的 dL/dy = 1 / count(y.dims)
std::shared_ptr<Instance> GradTest::InitLayerGradTestNetwork(
    const std::string& layer_type, const std::vector<Input>& layer_inputs,
    const std::vector<std::string>& layer_outputs, std::shared_ptr<LayerParam> layer_param,
    const std::set<std::string>& net_outputs, std::shared_ptr<LayerResource> layer_resource,
    const std::map<std::string, std::shared_ptr<RawBuffer>>& consts) {
    std::vector<Layer> layers(2);

    // 添加要测试的layer：这个算子的输入就是整个网络的输入
    Layer& layer        = layers.front();
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
    Layer& loss_layer        = layers.back();
    loss_layer.info.type_str = "ReduceMean";
    loss_layer.info.type     = GlobalConvertLayerType(loss_layer.info.type_str);
    loss_layer.info.name     = "loss_layer";
    loss_layer.info.inputs   = layer.info.outputs;
    loss_layer.info.outputs.push_back("loss");

    auto reduce_layer_param        = std::shared_ptr<ReduceLayerParam>(new ReduceLayerParam());
    reduce_layer_param->all_reduce = 1;
    loss_layer.info.param          = reduce_layer_param;

    return CreateInstance(GetNetworkConfig(), layer_inputs, net_outputs, layers, consts);
}

NetworkConfig GradTest::GetNetworkConfig() {
    NetworkConfig network_config;
    network_config.device_type = DEVICE_ARM;
    network_config.precision   = PRECISION_HIGH;

    TrainConfig& train_config                = network_config.train_config;
    train_config.run_mode                    = TNN_NS::TRAIN_MODE_TRAIN;
    train_config.train_the_whole_model       = true;
    train_config.solver_params.learning_rate = 0.1;
    return network_config;
}

}  // namespace TNN_NS
