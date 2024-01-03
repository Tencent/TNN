#include "test/unit_test/grad_test/grad_test.h"

namespace TNN_NS {

Status GradTest::DoLayerGradTest(std::map<std::string, std::shared_ptr<Mat>>& result, const std::string& layer_type,
                                 const std::vector<Input>& layer_inputs, const std::string& layer_output,
                                 std::shared_ptr<LayerParam> layer_param, std::shared_ptr<LayerResource> layer_resource,
                                 const std::set<std::string>& net_outputs) {
    std::vector<Layer> layers = CreateGradTestLayers(layer_type, layer_inputs, {layer_output}, layer_param, layer_resource);

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

    const std::vector<Input>& net_inputs = layer_inputs;

    // 常量在创建Instance的时候就传递进去
    auto instance =  CreateInstance(GetNetworkConfig(), net_inputs, net_outputs, layers, net_consts);

    // 普通输入通过SetInputMat设置
    Status ret = TNN_OK;
    MatConvertParam convert_param;
    for (const auto& [input_name, input_mat] : mat_inputs) {
        ret = instance->SetInputMat(input_mat, convert_param, input_name);
        RETURN_IF_FAIL(ret);
    }

    ret = instance->Forward();
    RETURN_IF_FAIL(ret);
    ret = instance->TrainStep();
    RETURN_IF_FAIL(ret);

    BlobMap blobs;
    ret = instance->GetAllOutputBlobs(blobs);
    RETURN_IF_FAIL(ret);
    result = Blob2Mat(instance, blobs);

    return TNN_OK;
}

// 创建一个Layer的后向测试网络：包含两个layer，一个是要测试的layer，一个是ReduceMean用来产生一个标量作为loss
// 1) 学习率：0.1
// 2) Loss使用ReduceMean产生，则要测试的Layer的 dL/dy = 1 / count(y.dims)
std::vector<Layer> GradTest::CreateGradTestLayers(
    const std::string& layer_type, const std::vector<Input>& layer_inputs,
    const std::vector<std::string>& layer_outputs, std::shared_ptr<LayerParam> layer_param,
    std::shared_ptr<LayerResource> layer_resource) {
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

    return layers;
}

NetworkConfig GradTest::GetNetworkConfig() {
    NetworkConfig network_config;
    network_config.device_type = DEVICE_ARM;
    network_config.precision   = PRECISION_HIGH;

    TrainConfig& train_config                = network_config.train_config;
    train_config.run_mode                    = TNN_NS::TRAIN_MODE_TRAIN;
    train_config.train_the_whole_model       = true;
    train_config.solver_params.learning_rate = GetLearningRate();
    return network_config;
}

float GradTest::GetLearningRate() {
    return 0.1;
}

}  // namespace TNN_NS
