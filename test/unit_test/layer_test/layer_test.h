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
#include "tnn/core/instance.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/core/tnn.h"
#include "tnn/layer/base_layer.h"
#include "tnn/interpreter/default_model_interpreter.h"

#define EXPECT_EQ_OR_RETURN(status, target)                                                                            \
    if ((status) != (target))                                                                                          \
    return (status)

namespace TNN_NS {

class LayerTest : public ::testing::Test {
protected:
    static void SetUpTestCase();

    void Run(std::shared_ptr<AbstractModelInterpreter> interp, Precision precision = PRECISION_AUTO, DataFormat cpu_input_data_format = DATA_FORMAT_AUTO, DataFormat device_input_data_format = DATA_FORMAT_AUTO);

    bool CheckDataTypeSkip(DataType data_type);

    static void TearDownTestCase();

private:
    Status Init(std::shared_ptr<AbstractModelInterpreter> interp, Precision precision, DataFormat cpu_input_data_format, DataFormat device_input_data_format);
    Status Forward();
    Status Compare();
    Status DeInit();

protected:
    int ensure_input_positive_ = 0;
    int integer_input_min_ = 0;
    int integer_input_max_ = 1;

    static std::shared_ptr<Instance> instance_cpu_;
    static std::shared_ptr<Instance> instance_device_;
    static std::shared_ptr<Instance> instance_ocl_cache_;

private:
    Status GenerateRandomBlob(Blob* cpu_blob, Blob* device_blob, void* command_queue_dev, int magic_num);
    int CompareBlob(Blob* cpu_blob, Blob* device_blob, void* command_queue_dev);
    int CompareDims(DimsVector dims_a, DimsVector dims_b);

    Status InitInputBlobsDataRandom();
};

}  // namespace TNN_NS

#endif  // TNN_TEST_UNIT_TEST_LAYER_TEST_LAYER_TEST_H_
