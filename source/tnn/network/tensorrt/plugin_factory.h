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

#ifndef TNN_SOURCE_TNN_NETWORK_TENSORRT_PLUGIN_FACTORY_H_
#define TNN_SOURCE_TNN_NETWORK_TENSORRT_PLUGIN_FACTORY_H_

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "tnn/network/tensorrt/layer_builder/tensorrt_plugin_layer_builder.h"

namespace TNN_NS {

class TensorRTNetwork;

class PluginFactory : public nvinfer1::IPluginFactory {
public:
    PluginFactory(TensorRTNetwork* net);

    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;

private:
    TensorRTNetwork* m_net;
};

}  //  namespace TNN_NS

#endif  //  TNN_SOURCE_TNN_NETWORK_TENSORRT_PLUGIN_FACTORY_H_