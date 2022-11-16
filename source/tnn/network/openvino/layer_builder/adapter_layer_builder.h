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

#ifndef TNN_SOURCE_TNN_DEVICE_OPENVINO_LAYER_BUILDER_ADAPTER_LAYER_BUILDER_H_
#define TNN_SOURCE_TNN_DEVICE_OPENVINO_LAYER_BUILDER_ADAPTER_LAYER_BUILDER_H_

#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"

namespace TNN_NS {

class AdapterOVLayerBuilder : public OpenVINOLayerBuilder {
public:
    AdapterOVLayerBuilder(LayerType layer_type) : OpenVINOLayerBuilder(layer_type){};
    virtual ~AdapterOVLayerBuilder(){};

protected:
    virtual Status InferOutputShape() {
        return TNN_OK;
    };
    virtual Status InferOutputDataType() {
        return TNN_OK;
    };
    virtual Status Build();
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENVINO_LAYER_BUILDER_ADAPTER_LAYER_BUILDER_H__
