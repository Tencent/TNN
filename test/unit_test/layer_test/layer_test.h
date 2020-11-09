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

#ifndef TNN_TEST_UNIT_TEST_LAYER_TEST_LAYER_TEST_H_
#define TNN_TEST_UNIT_TEST_LAYER_TEST_LAYER_TEST_H_

#include <gtest/gtest.h>

#include "test/flags.h"
#include "test/test_utils.h"
#include "test/unit_test/layer_test/layer_test_utils.h"
#include "test/unit_test/unit_test_macro.h"
#include "tnn/core/abstract_device.h"
#include "tnn/core/common.h"
#include "tnn/core/context.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/layer/base_layer.h"

#define EXPECT_EQ_OR_RETURN(status, target)                                                                            \
    if ((status) != (target))                                                                                          \
    return (status)

namespace TNN_NS {

class LayerTest : public ::testing::Test {
protected:
    static void SetUpTestCase();

    void Run(LayerType, LayerParam* param, LayerResource* resource, std::vector<BlobDesc>& inputs_desc,
             std::vector<BlobDesc>& outputs_desc);

    static void TearDownTestCase();

private:
    Status Init(LayerType, LayerParam* param, LayerResource* resource, std::vector<BlobDesc>& inputs_desc,
                std::vector<BlobDesc>& outputs_desc);
    Status Reshape();
    Status Forward();
    Status Compare();
    Status DeInit();

protected:
    static AbstractDevice* cpu_;
    static AbstractDevice* device_;
    static Context* cpu_context_;
    static Context* device_context_;

    LayerParam* param_;
    BaseLayer* cpu_layer_;
    BaseLayer* device_layer_;
    std::vector<Blob*> cpu_inputs_;
    std::vector<Blob*> cpu_outputs_;
    std::vector<Blob*> device_inputs_;
    std::vector<Blob*> device_outputs_;
    int ensure_input_positive_ = 0;

private:
    Status CreateLayers(LayerType type);

    Status CreateInputBlobs(std::vector<BlobDesc>& inputs_desc);

    Status InitInputBlobsDataRandom(LayerType type);

    Status InitLayers(LayerType type, LayerParam* param, LayerResource* resource, std::vector<BlobDesc>& inputs_desc,
                      std::vector<BlobDesc>& outputs_desc);

    Status CreateOutputBlobs(std::vector<BlobDesc>& outputs_desc);

    Status AllocateInputBlobs();
    Status AllocateOutputBlobs();

    virtual float GetCalcMflops(LayerParam* param, std::vector<Blob*> inputs, std::vector<Blob*> outputs) {
        return 0.f;
    }
    virtual float GetCalcDramThrp(float avg_time);
};

}  // namespace TNN_NS

#endif  // TNN_TEST_UNIT_TEST_LAYER_TEST_LAYER_TEST_H_
