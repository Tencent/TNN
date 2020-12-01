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

#include "test/unit_test/layer_test/test_binary_layer.h"

namespace TNN_NS {

BinaryLayerTest::BinaryLayerTest(LayerType type) {
    layer_type_ = type;
}

bool BinaryLayerTest::InputParamCheck(const DataType& data_type, const DeviceType& dev, const int batch) {
    if (data_type == DATA_TYPE_INT8 && DEVICE_ARM != dev) {
        return true;
    }

    if (batch > 1 && DEVICE_METAL == dev) {
        return true;
    }

    return false;
}

void BinaryLayerTest::RunBinaryTest(std::string layer_type_str, bool resource_positive) {
    // get param
    int batch           = std::get<0>(GetParam());
    int channel         = std::get<1>(GetParam());
    int input_size      = std::get<2>(GetParam());
    int input_count     = std::get<3>(GetParam());
    int param_size_type = std::get<4>(GetParam());
    int weight_idx      = std::get<5>(GetParam());
    DataType data_type  = std::get<6>(GetParam());
    DeviceType dev      = ConvertDeviceType(FLAGS_dt);

    if (InputParamCheck(data_type, dev, batch)) {
        GTEST_SKIP();
    }

    if (batch > 1 && param_size_type == 3 && DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    std::vector<int> param_dims;
    int param_count = 1;
    if (0 == param_size_type) {
        param_count = 1;
        param_dims  = {1, 1, 1, 1};
    } else if (1 == param_size_type) {
        param_count = channel;
        param_dims  = {1, channel, 1, 1};
    } else if (2 == param_size_type) {
        param_count = channel * input_size * input_size;
        param_dims  = {1, channel, input_size, input_size};
    } else if (3 == param_size_type) {
        param_count = input_size * input_size;
        param_dims  = {1, 1, input_size, input_size};
    }

    std::shared_ptr<EltwiseLayerResource> resource = nullptr;
    if (input_count == 1) {
        resource = std::shared_ptr<EltwiseLayerResource>(new EltwiseLayerResource());
        RawBuffer buffer(param_count * sizeof(float));
        float* buffer_data = buffer.force_to<float*>();
        if (resource_positive) {
            InitRandom(buffer_data, param_count, 0.001f, 1.0f);
        } else {
            InitRandom(buffer_data, param_count, 1.0f);
        }
        resource->element_handle = buffer;
        resource->element_shape  = param_dims;
    }

    // param
    std::shared_ptr<MultidirBroadcastLayerParam> param = nullptr;
    if (LAYER_HARDSWISH == layer_type_) {
        param = std::shared_ptr<MultidirBroadcastLayerParam>(new HardSwishLayerParam());
    } else {
        param = std::shared_ptr<MultidirBroadcastLayerParam>(new MultidirBroadcastLayerParam());
    }

    param->name               = "Binary";
    param->weight_input_index = weight_idx;

    std::vector<int> input0_dims;
    std::vector<int> input1_dims;
    // blob desc
    std::vector<BlobDesc> inputs_desc;
    if (1 == input_count) {
        if (-1 == weight_idx) {
            // this case doesn't exist
            return;
        } else if (0 == weight_idx) {
            input0_dims = {batch, channel, input_size, input_size};
        } else if (1 == weight_idx) {
            input0_dims = {batch, channel, input_size, input_size};
        }
    } else if (2 == input_count) {
        if (-1 == weight_idx) {
            // the size of input are same
            input0_dims = {batch, channel, input_size, input_size};
            input1_dims = {batch, channel, input_size, input_size};
        } else {
            std::vector<int> weight_dims;
            if (0 == param_size_type) {
                weight_dims = {1, 1, 1, 1};
            } else if (1 == param_size_type) {
                weight_dims = {1, channel, 1, 1};
            } else if (2 == param_size_type) {
                weight_dims = {1, channel, input_size, input_size};
            } else if (3 == param_size_type) {
                weight_dims = {1, 1, input_size, input_size};
            }

            if (0 == weight_idx) {
                input0_dims = weight_dims;
                input1_dims = {batch, channel, input_size, input_size};
            } else if (1 == weight_idx) {
                input0_dims = {batch, channel, input_size, input_size};
                input1_dims = weight_dims;
            }
        }
    } else {
        // not support yet
        return;
    }

    Precision precision = PRECISION_AUTO;
    if (DATA_TYPE_INT8 == data_type) {
        param->quantized = true;
    } else if (DATA_TYPE_BFP16 == data_type) {
        precision = PRECISION_LOW;
    }

    std::shared_ptr<AbstractModelInterpreter> interpreter;
    if (1 == input_count) {
        interpreter = GenerateInterpreter(layer_type_str, {input0_dims}, param, resource);
    } else if (2 == input_count) {
        interpreter = GenerateInterpreter(layer_type_str, {input0_dims, input1_dims}, param);
    }
    Run(interpreter, precision);
}

}  // namespace TNN_NS
