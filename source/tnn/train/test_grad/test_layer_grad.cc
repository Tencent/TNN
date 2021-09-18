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

#include "tnn/train/test_grad/test_layer_grad.h"

namespace TNN_NS {
namespace train {
LayerGradTest::~LayerGradTest(){
    if(context.config)
        delete context.config;
    if(context.network)
        delete context.network;
}
Status LayerGradTest::GenerateContext() {
    context.config = new NetworkConfig();
}

Status LayerGradTestManager::RunTestGrad() {
    auto& grad_test_map = GetLayerGradTestMap();
    Status status;
    for(auto iter: grad_test_map) {
        status = iter.second->TestGrad();
        RETURN_ON_NEQ(status, TNN_OK);
    }
}   
}
}