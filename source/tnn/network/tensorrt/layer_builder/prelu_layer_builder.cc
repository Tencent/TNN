#include "tnn/network/tensorrt/layer_builder/tensorrt_plugin_layer_builder.h"

namespace TNN_NS {

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(PRelu, LAYER_PRELU);

bool PReluTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) &&
        inOut[pos].format == nvinfer1::TensorFormat::kNCHW);
}

Status PReluTRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

const char* PReluTRTPluginLayerBuilder::getPluginType() const {
    return "PRelu";
}

nvinfer1::DataType PReluTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* PReluTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    const auto paramlist = dynamic_cast<PReluLayerParam *>(param_);
    if (!paramlist->channel_shared) {
        return TensorRTPluginLayerBuilder::AddToNetwork(network);
    }

    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto output_foreign_tensor = dynamic_cast<ForeignBlob*>(output_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();
    bool int8 = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetInt8Mode();

    const auto activation_type = nvinfer1::ActivationType::kLEAKY_RELU;
    ILayer* last_layer;
    IActivationLayer* activation_layer = network->addActivation(*input_tensor, activation_type);
    if (activation_layer != nullptr) {
        activation_layer->setName(layer_name_.c_str());
        auto resource = dynamic_cast<PReluLayerResource*>(resource_);
        auto scope = resource->slope_handle.force_to<float*>();
        activation_layer->setAlpha(*scope);
        last_layer = activation_layer;
    }

    if (int8) {
        float output_scale_value = std::dynamic_pointer_cast<TensorRTTensor>(
            output_foreign_tensor)->GetIntResource()->scale_handle.force_to<float*>()[0];
        return AddInt8OutputQDQLayers(network, last_layer->getOutput(0), output_foreign_tensor,
            output_scale_value, 1 / output_scale_value);
    }

    return last_layer;
}

DimsExprs PReluTRTPluginLayerBuilder::getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs,
        int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
    return TensorRTPluginLayerBuilder::getOutputDimensions(index, inputs, nbInputs, exprBuilder);
}

const char* PReluPluginCreator::getPluginName() const {
    return "PRelu";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(PRelu, LAYER_PRELU);

}  //  namespace TNN_NS

