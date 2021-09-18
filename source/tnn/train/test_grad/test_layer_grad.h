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

#ifndef TNN_SOURCE_TNN_TRAIN_TEST_GRAD_TEST_CONCAT_GRAD_H
#define TNN_SOURCE_TNN_TRAIN_TEST_GRAD_TEST_CONCAT_GRAD_H
#include "tnn/train/grad/layer_grad.h"
namespace TNN_NS {
namespace train {
class LayerGradTest {
public:
    LayerGradTest(){};
    virtual ~LayerGradTest() = default;
    virtual Status GenerateContext() = 0;
    virtual Status TestGrad() = 0;
protected:
    std::shared_ptr<BaseLayer> layer = nullptr;
    TrainContext context;
};
class LayerGradTestManager {
public:
    LayerGradTestManager(){};
    virtual ~LayerGradTestManager() = default;
    Status RunTestGrad();
    static void RegisterLayerGradTest(LayerType type, std::shared_ptr<LayerGradTest> layer_grad_test_p) {
        GetLayerGradTestMap()[type] = layer_grad_test_p;
    };
    static std::map<LayerType, std::shared_ptr<LayerGradTest>> &GetLayerGradTestMap() {
        static std::map<LayerType, std::shared_ptr<LayerGradTest>> layer_2_grad_test_map;
        return layer_2_grad_test_map;
    };
private:
    
};

template <typename T> class LayerGradRegister {
public:
    explicit LayerGradRegister(LayerType type) {
        LayerGrad::RegisterLayerGrad(type, std::make_shared<T>());
    }
};

#define DECLARE_LAYER_GRAD_TEST_BEGIN(type_string, layer_type)  \
    class type_string##LayerGradTest : public LayerGradTest {  \
    public:                                               \
        virtual ~type_string##LayerGradTest(){};  \
        virtual Status TestGrad();  \
        virtual Status GenerateContext(); \

#define DECLARE_LAYER_GRAD_TEST_END };  

#define REGISTER_LAYER_GRAD_TEST(type_string, layer_type)                                                                   \
    LayerGradRegister<type_string##LayerGradTest> g_##layer_type##_layer_grad_test_register(layer_type);

} // namespace train
} // namespace TNN_NS

#endif