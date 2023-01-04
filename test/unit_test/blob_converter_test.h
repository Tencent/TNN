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

#ifndef TNN_TEST_UNIT_TEST_BLOB_CONVERTER_TEST_H_
#define TNN_TEST_UNIT_TEST_BLOB_CONVERTER_TEST_H_

#include <gtest/gtest.h>

#include "test/flags.h"
#include "test/test_utils.h"
#include "tnn/core/abstract_device.h"
#include "tnn/core/common.h"
#include "tnn/core/context.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/utils/blob_converter.h"

namespace TNN_NS {

class BlobConverterTest : public ::testing::TestWithParam<std::tuple<int, int, int, float, float, bool, MatType, DataType>> {
public:
    static void SetUpTestCase();
    static void TearDownTestCase();

protected:
    int Compare(Blob* cpu_blob, Blob* device_blob);
    bool TestFilterCheck(const DataType& blob_data_type, const DeviceType& dev,
                         const MatType& mat_type, const int batch, const int channel,
                         const int input_size, const bool reverse_channel);
    int OpenCLMatTest(Mat& cpu_mat_in,
                       MatConvertParam& from_mat_param, MatConvertParam& to_mat_param,
                       const DimsVector& dims, const int in_size, const int out_size,
                       MatType mat_type, const int mat_channel, const int channel,
                       BlobConverter& device_converter, void* device_command_queue,
                       void* mat_out_ref_data);
    static AbstractDevice* cpu_;
    static AbstractDevice* device_;
    static Context* cpu_context_;
    static Context* device_context_;
};

}  // namespace TNN_NS

#endif  // TNN_TEST_UNIT_TEST_BLOB_CONVERTER_TEST_H_
