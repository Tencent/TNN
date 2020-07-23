// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <mutex>

#include "tnn/network/tensorrt/layer_builder/tensorrt_plugin_layer_builder.h"

namespace TNN_NS {

TensorRTPluginLayerBuilder::TensorRTPluginLayerBuilder(LayerType type) : BaseLayerBuilder(type) {
}

TensorRTPluginLayerBuilder::~TensorRTPluginLayerBuilder() {
}

Status TensorRTPluginLayerBuilder::Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& input_blobs,
        std::vector<Blob*>& output_blobs, AbstractDevice* device) {
    input_blobs_ = input_blobs;
    output_blobs_ = output_blobs;

    param_    = param;
    resource_ = resource;

    Build();
    auto status = InferOutputDataType();
    if (status != TNN_OK) {
        return status;
    }

    status = InferOutputShape();
    LOGD("InferOutputShape: name:%s shape:%d %d %d %d \n", param->name.c_str(), output_blobs[0]->GetBlobDesc().dims[0],
         output_blobs[0]->GetBlobDesc().dims[1], output_blobs[0]->GetBlobDesc().dims[2],
         output_blobs[0]->GetBlobDesc().dims[3]);
    if (status != TNN_OK) {
        return status;
    }
    auto dims = output_blobs[0]->GetBlobDesc().dims;
    for (auto item : dims) {
        if (item <= 0) {
            LOGE("Error: layer(%s) output dims is invalid\n", layer_name_.c_str());
            return Status(TNNERR_LAYER_ERR, "layer output dims is invalid");
        }
    }

    layer_acc_ = device->CreateLayerAcc(type_);
    if (layer_acc_ != NULL) {
        return layer_acc_->Init(context, param, resource, input_blobs_, output_blobs_);
    } else {
        LOGE("layer acc of type(%d) is nil\n", type_);
        return Status(TNNERR_LAYER_ERR, "layer acc is nil");
    }

    return TNN_OK;
}

Status TensorRTPluginLayerBuilder::Reshape() {
    return TNN_OK;
}

Status TensorRTPluginLayerBuilder::Forward() {
    return TNN_OK;
}

int TensorRTPluginLayerBuilder::getNbOutputs() const {
    return output_blobs_.size();
}

Dims TensorRTPluginLayerBuilder::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    auto shape = output_blobs_[index]->GetBlobDesc();
    return DimsCHW(shape.dims[1], shape.dims[2], shape.dims[3]);
}

bool TensorRTPluginLayerBuilder::supportFormat(nvinfer1::DataType type, PluginFormat format) const {
    if (type == nvinfer1::DataType::kFLOAT && format == PluginFormat::kNCHW) {
            return true;
    }
    return false; 
}

void TensorRTPluginLayerBuilder::configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
            nvinfer1::DataType type, PluginFormat format, int maxBatchSize) {
    m_type = type;
    m_format = format;
}

int TensorRTPluginLayerBuilder::initialize() {
    return 0;
}

void TensorRTPluginLayerBuilder::terminate() {
}

size_t TensorRTPluginLayerBuilder::getWorkspaceSize(int maxBatchSize) const {
    return 0;
}

// enqueue

size_t TensorRTPluginLayerBuilder::getSerializationSize() {
    return sizeof(m_type) + sizeof(m_format);
}

void TensorRTPluginLayerBuilder::serialize(void* buffer) {
    char* d = reinterpret_cast<char*>(buffer);
    write(d, m_type);
    write(d, m_format);
}

std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>& GetTensorRTPluginLayerBuilderCreatorMap() {
    // static shared_ptr of LayerCreatorMap.
    static std::once_flag once;
    static std::shared_ptr<std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>> creators;
    std::call_once(once, []() { creators.reset(new std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>); });
    return *creators;
}

TensorRTPluginLayerBuilder* CreateTensorRTPluginLayerBuilder(LayerType type) {
    TensorRTPluginLayerBuilder* cur_layer = nullptr;
    auto& map = GetTensorRTPluginLayerBuilderCreatorMap();
    if (map.count(type) > 0) {
        auto base_layer = map[type]->CreateLayerBuilder();
        cur_layer = dynamic_cast<TensorRTPluginLayerBuilder*>(base_layer);
    }
    return cur_layer;
}


}  //  namespace TNN_NS