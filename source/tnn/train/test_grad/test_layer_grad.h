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
    //virtual Status GenerateContext() = 0;
    virtual Status TestGrad() = 0;
};
class LayerGradTestManager {
public:
    LayerGradTestManager(){};
    virtual ~LayerGradTestManager() = default;
    static Status RunTestGrad();
    static void RegisterLayerGradTest(LayerType type, std::shared_ptr<LayerGradTest> layer_grad_test_p) {
        GetLayerGradTestMap()[type] = layer_grad_test_p;
    };
    static std::map<LayerType, std::shared_ptr<LayerGradTest>> &GetLayerGradTestMap() {
        static std::map<LayerType, std::shared_ptr<LayerGradTest>> layer_2_grad_test_map;
        return layer_2_grad_test_map;
    };
};

template <typename T> class LayerGradTestRegister {
public:
    explicit LayerGradTestRegister(LayerType type) {
        LayerGradTestManager::RegisterLayerGradTest(type, std::make_shared<T>());
    }
};

#define DECLARE_LAYER_GRAD_TEST_BEGIN(type_string, layer_type)  \
    class type_string##LayerGradTest : public LayerGradTest {  \
    public:                                               \
        virtual ~type_string##LayerGradTest(){};  \
        virtual Status TestGrad();  \

#define DECLARE_LAYER_GRAD_TEST_END };  

#define REGISTER_LAYER_GRAD_TEST(type_string, layer_type)                                                                   \
    LayerGradTestRegister<type_string##LayerGradTest> g_##layer_type##_layer_grad_test_register(layer_type);

using NameShapes = std::vector<std::pair<std::string, DimsVector>>;
using BlobShapes = std::vector<std::pair<Blob*, DimsVector>>;
using NameBuffers = std::vector<std::pair<std::string, RawBuffer>>;

template<typename T>
int InitRandom(T* host_data, size_t n, T range_min, T range_max, bool except_zero);
Status generate_raw_buffer(std::map<Blob *, std::shared_ptr<RawBuffer>>& buffers, const BlobShapes& shapes, DeviceType device_type, DataFormat data_format, DataType data_type, bool generate_data, bool except_zero=false);
Status generate_blob(std::vector<Blob*>& blobs, const NameShapes& shapes, DeviceType device_type, DataFormat data_format, DataType data_type, bool generate_data, bool except_zero=false);
void free_blobs(std::vector<Blob*>& blobs);
void output_buffer(RawBuffer* buffer, const std::string name = "");
void output_blob(Blob* blob, const std::string name = "");
void ouput_data(void* data, const DimsVector dims, const std::string name);

} // namespace train
} // namespace TNN_NS

#endif