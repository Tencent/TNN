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

// author: sanerzheng@tencent.com

#ifndef TNN_SOURCE_TNN_TRAIN_GRAD_GRAD_H
#define TNN_SOURCE_TNN_TRAIN_GRAD_GRAD_H

#include <string>
#include <set>
#include "core/tnn.h"
#include "tnn/train/grad/layer_grad.h"
#include "tnn/core/default_network.h"

namespace TNN_NS {
namespace train {
class GradManager {
public:
    GradManager(AbstractNetwork* network_);
    ~GradManager() = default;
    Status CalcuteGrads(Blob* loss);
    inline std::map<Blob*, std::shared_ptr<RawBuffer> >& GetBackWardGradsBlob(){
        return backward_grads_blob_;
    };
    inline std::map<RawBuffer*, std::shared_ptr<RawBuffer> >& GetBackWardGradsResource(){
        return backward_grads_resource_;
    };
    static void RegisterLayerGrad(LayerType type, LayerGrad* layer_grad_p);
private:
    static std::map<LayerType, std::shared_ptr<LayerGrad>> GetLayerGradMap();
    static std::set<RawBuffer* > trainables_;
    AbstractNetwork* network_;
    std::map<Blob*, std::shared_ptr<RawBuffer> > backward_grads_blob_;
    std::map<RawBuffer*, std::shared_ptr<RawBuffer>> backward_grads_resource_;
};
template <typename T>
class LayerGradRegister {
public:
    explicit LayerGradRegister(LayerType type) {
        GradManger::RegisterLayerGrad(type, new T());
    }
};

#define REGISTER_LAYER_GRAD(type_string, layer_type)                                                                      \
    LayerGradRegister<##type_string##LayerGrad> g_##layer_type##_layer_grad_register(        \
        layer_type);


} // namespace train
} // namespace TNN_NS
#endif  // TNN_SOURCE_TNN_TRAIN_GRAD_GRAD_H