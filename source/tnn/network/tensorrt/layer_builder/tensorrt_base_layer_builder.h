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

#ifndef TNN_SOURCE_TNN_NETWORK_TENSORRT_LAYER_BUILDER_TENSORRT_BASE_LAYER_BUILDER_H_
#define TNN_SOURCE_TNN_NETWORK_TENSORRT_LAYER_BUILDER_TENSORRT_BASE_LAYER_BUILDER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvUtils.h"
#include "NvInferPlugin.h"

#include "tnn/core/macro.h"
#include "tnn/layer/base_layer.h"
#include "tnn/core/abstract_device.h"
#include "tnn/core/blob.h"
#include "tnn/core/context.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/status.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource.h"
#include "tnn/network/tensorrt/shape_tensor.h"
#include "tnn/network/tensorrt/tensorrt_tensor.h"
#include "tnn/extern_wrapper/base_layer_builder.h"
#include "tnn/extern_wrapper/foreign_blob.h"

using namespace nvinfer1;
using namespace plugin;

namespace TNN_NS {

class TensorRTNetwork_;

// @brief BaseLayer Builder, defines the layer builder interface
class TensorRTBaseLayerBuilder: public BaseLayerBuilder {
public:
    explicit TensorRTBaseLayerBuilder(LayerType type);

    // @brief virtual destructor
    virtual ~TensorRTBaseLayerBuilder();

    // @brief virtual layer init
    virtual Status Init(Context* context, LayerParam* param, LayerResource* resource, std::vector<Blob*>& inputs,
            std::vector<Blob*>& outputs, AbstractDevice* device, bool enable_const_folder=true) = 0;

    // @brief virtual Reshape recalculate the output tensor dims
    virtual Status Reshape();

    // @brief layer infer
    virtual Status Forward() = 0;

    // @brief add layer to tensorRT network
    virtual ILayer* AddToNetwork(INetworkDefinition* network) = 0;

    // @brief calculate the output tensor dims
    virtual Status InferOutputShape();

    // @brief check whether is a plugin layer
    bool IsPluginLayer();

    // @brief set constant resource
    virtual void SetConstantResource(ConstantResource* consts);

    // @brief set tensorrt_network
    void SetNetwork(TensorRTNetwork_ *network);

protected:
    // @brief Build the foreign network
    virtual Status Build();

    ILayer* AddInt8OutputQDQLayers(nvinfer1::INetworkDefinition* network, ITensor* tensor,
        std::shared_ptr<ForeignTensor> foreign_tensor, float quant_scale, float dequant_scale);

    ILayer* AddInt8WeightQDQLayers(nvinfer1::INetworkDefinition* network, RawBuffer* weight,
        nvinfer1::Weights &kernelWeights, RawBuffer* bias, nvinfer1::Weights &biasWeights,
        float scale, std::vector<int> dims);

    std::vector<ITensor*> GetInputITensors();

    std::vector<ITensor*> GetOutputITensors();

    std::shared_ptr<BaseLayer> m_layer;
    std::vector<float*> int8_weight_data;
    bool is_plugin;

    TensorRTNetwork_* m_network;
};

class TensorRTLayerBuilder;
class TensorRTLayerPluginBuilder;

TensorRTBaseLayerBuilder* CreateTensorRTBaseLayerBuilder(LayerType type);

// @brief TensorRTLayerBuilderCreator register map
std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>& GetTRTLayerBuilderCreatorMap();

// @brief TensorRTPluginLayerBuilderCreator register map
std::map<LayerType, std::shared_ptr<LayerBuilderCreator>>& GetTRTPluginLayerBuilderCreatorMap();

}  //  TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TENSORRT_LAYER_BUILDER_TENSORRT_BASE_LAYER_BUILDER_H_
