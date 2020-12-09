#include "tnn/network/tensorrt/layer_builder/tensorrt_plugin_layer_builder.h"

namespace TNN_NS {

DECLARE_TENSORRT_PLUGIN_LAYER_BUILDER(FusedReduce, LAYER_FUSED_REDUCE);

bool FusedReduceTRTPluginLayerBuilder::supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    return nbInputs == 1 && nbOutputs == 1 && pos < nbInputs + nbOutputs && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR &&
        (inOut[pos].type == nvinfer1::DataType::kFLOAT ||
        inOut[pos].type == nvinfer1::DataType::kHALF);
}

const char* FusedReduceTRTPluginLayerBuilder::getPluginType() const {
    return "FusedReduce";
}

nvinfer1::DataType FusedReduceTRTPluginLayerBuilder::getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
        int nbInputs) const {
    return inputTypes[0];
}

ILayer* FusedReduceTRTPluginLayerBuilder::AddToNetwork(INetworkDefinition* network) {
    auto input_foreign_tensor = dynamic_cast<ForeignBlob*>(input_blobs_[0])->GetForeignTensor();
    auto input_tensor = std::dynamic_pointer_cast<TensorRTTensor>(input_foreign_tensor)->GetTensor();

    return TensorRTPluginLayerBuilder::AddToNetwork(network);
}

const char* FusedReducePluginCreator::getPluginName() const {
    return "FusedReduce";
}

REGISTER_TENSORRT_PLUGIN_LAYER_BUILDER(FusedReduce, LAYER_FUSED_REDUCE);

}  //  namespace TNN_NS