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

#include "test/unit_test/blob_converter_test.h"
#include "test/unit_test/unit_test_common.h"
#include "test/unit_test/unit_test_macro.h"
#include "tnn/core/blob_int8.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "utils/network_helpers.h"

namespace TNN_NS {

AbstractDevice* BlobConverterTest::cpu_;
AbstractDevice* BlobConverterTest::device_;
Context* BlobConverterTest::cpu_context_;
Context* BlobConverterTest::device_context_;

void BlobConverterTest::SetUpTestCase() {
    NetworkConfig config;
    config.device_type = ConvertDeviceType(FLAGS_dt);
    if (FLAGS_lp.length() > 0) {
        config.library_path = {FLAGS_lp};
    }
    TNN_NS::Status ret = TNN_NS::TNN_OK;

    // cpu
    cpu_ = GetDevice(DEVICE_NAIVE);
    ASSERT(cpu_ != NULL);

    cpu_context_ = cpu_->CreateContext(0);
    ASSERT(cpu_context_ != NULL);

    // device
    device_ = GetDevice(config.device_type);
    ASSERT(device_ != NULL);

    device_context_ = device_->CreateContext(config.device_id);
    ASSERT(device_context_ != NULL);

    ret = device_context_->LoadLibrary(config.library_path);
    ASSERT(ret == TNN_OK);
}

void BlobConverterTest::TearDownTestCase() {
    delete cpu_context_;
    delete device_context_;
}

INSTANTIATE_TEST_SUITE_P(BlobConverterTest, BlobConverterTest,
                         ::testing::Combine(
                            // batch
                            testing::Values(1, 2),
                            // channel
                            testing::Values(1, 3, 4),
                            // inputsize
                            testing::Values(1, 15, 16),
                            // scale
                            testing::Values(0.5, 1.0),
                            // bias
                            testing::Values(0.0, 1.0),
                            // reverse_channel
                            testing::Values(false),
                            // mat type
                            testing::Values(N8UC4, N8UC3, NGRAY, NNV12, NNV21,
                                            NCHW_FLOAT),  // datatype
                            testing::Values(DATA_TYPE_FLOAT, DATA_TYPE_INT8)));

TEST_P(BlobConverterTest, BlobConverterTest) {
    int batch               = std::get<0>(GetParam());
    int channel             = std::get<1>(GetParam());
    int input_size          = std::get<2>(GetParam());
    float scale             = std::get<3>(GetParam());
    float bias              = std::get<4>(GetParam());
    bool reverse_channel    = std::get<5>(GetParam());
    MatType mat_type        = std::get<6>(GetParam());
    DataType blob_data_type = std::get<7>(GetParam());

    DeviceType dev = ConvertDeviceType(FLAGS_dt);
    if (blob_data_type == DATA_TYPE_INT8 && DEVICE_ARM != dev) {
        GTEST_SKIP();
    }

    if (DEVICE_METAL == dev) {
        GTEST_SKIP();
    }

    if (mat_type == N8UC3 && channel != 3) {
        GTEST_SKIP();
    } else if (mat_type == N8UC4 && channel != 3 && channel != 4) {
        GTEST_SKIP();
    } else if (mat_type == NGRAY && channel != 1) {
        GTEST_SKIP();
    } else if ((mat_type == NNV12 || mat_type == NNV21) && (channel != 3 || input_size % 2 != 0 || DEVICE_OPENCL == dev)) {
        GTEST_SKIP();
    }

    int mat_channel;
    if (mat_type == N8UC4) {
        mat_channel = 4;
    } else {
        mat_channel = channel;
    }
    int in_size       = batch * mat_channel * input_size * input_size;
    int out_size      = in_size;
    int out_nchw_size = batch * channel * input_size * input_size;

    DimsVector dims = {batch, channel, input_size, input_size};

    void* mat_in_data           = nullptr;
    void* mat_out_ref_nchw_data = nullptr;
    void* mat_out_dev_nchw_data = nullptr;
    void* mat_out_ref_data      = nullptr;
    void* mat_out_dev_data      = nullptr;

    mat_out_ref_nchw_data = malloc(out_nchw_size * sizeof(float));
    mat_out_dev_nchw_data = malloc(out_nchw_size * sizeof(float));

    if (mat_type == NCHW_FLOAT) {
        mat_in_data = malloc(in_size * sizeof(float));
        InitRandom(static_cast<float*>(mat_in_data), in_size, 0.0f, 1.0f);
        mat_out_ref_data = malloc(out_size * sizeof(float));
        mat_out_dev_data = malloc(out_size * sizeof(float));
    } else {
        mat_in_data = malloc(in_size * sizeof(uint8_t));
        InitRandom(static_cast<uint8_t*>(mat_in_data), in_size, static_cast<uint8_t>(0), static_cast<uint8_t>(255));
        mat_out_ref_data = malloc(out_size * sizeof(uint8_t));
        mat_out_dev_data = malloc(out_size * sizeof(uint8_t));
    }

    // blob desc
    BlobDesc cpu_blob_desc, device_blob_desc;
    cpu_blob_desc.dims        = {batch, channel, input_size, input_size};
    cpu_blob_desc.device_type = DEVICE_NAIVE;
    cpu_blob_desc.data_type   = blob_data_type;
    cpu_blob_desc.data_format = GetDefaultDataFormat(DEVICE_NAIVE);
    Blob *cpu_blob, *device_blob;

    device_blob_desc             = cpu_blob_desc;
    DeviceType device_type       = device_->GetDeviceType();
    device_blob_desc.device_type = device_type;
    device_blob_desc.data_format = GetDefaultDataFormat(device_type);
    float max_i8_diff            = 0;
    if (blob_data_type == DATA_TYPE_FLOAT) {
        cpu_blob    = new Blob(cpu_blob_desc);
        device_blob = new Blob(device_blob_desc);
    } else {
        auto int_scale = CreateIntScale(channel);
        auto scaleptr  = int_scale->scale_handle.force_to<float*>();
        for (int i = 0; i < channel; i++) {
            auto s = fabs(scaleptr[i]);
            if (s != 0 && 1.0 / s > max_i8_diff)
                max_i8_diff = 1.0 / s;
        }
        auto tmp = new BlobInt8(cpu_blob_desc);
        tmp->SetIntResource(int_scale);
        cpu_blob = tmp;

        tmp = new BlobInt8(device_blob_desc);
        tmp->SetIntResource(int_scale);
        device_blob = tmp;
    }

    BlobHandleAllocate(cpu_blob, cpu_);
    BlobHandleAllocate(device_blob, device_);
    void* device_command_queue;
    device_context_->GetCommandQueue(&device_command_queue);

    BlobConverter cpu_converter(cpu_blob);
    BlobConverter device_converter(device_blob);

    MatConvertParam from_mat_param;
    from_mat_param.reverse_channel = reverse_channel;

    if (mat_type != NCHW_FLOAT) {
        from_mat_param.scale = {scale * 1, scale * 2, scale * 3, scale * 4};
        from_mat_param.bias  = {bias, bias * 2, bias * 3, bias * 4};
    }

    Mat mat_in(DEVICE_NAIVE, mat_type, dims, mat_in_data);
    Status ret;
    ret = cpu_converter.ConvertFromMat(mat_in, from_mat_param, NULL);
    if (ret != TNN_OK) {
        LOGE("cpu converter convert mat to blob failed, mat type: %d\n", mat_type);
        FAIL();
    }
    ret = device_converter.ConvertFromMat(mat_in, from_mat_param, device_command_queue);
    if (ret != TNN_OK) {
        LOGE("device converter convert mat to blob failed, mat type: %d\n", mat_type);
        FAIL();
    }

    MatConvertParam to_mat_param;
    to_mat_param.reverse_channel = reverse_channel;
    Mat mat_out_ref_nchw(DEVICE_NAIVE, NCHW_FLOAT, dims, mat_out_ref_nchw_data);
    ret = cpu_converter.ConvertToMat(mat_out_ref_nchw, to_mat_param, NULL);
    if (ret != TNN_OK) {
        LOGE("cpu converter convert blob to mat failed, mat type: %d\n", NCHW_FLOAT);
        FAIL();
    }
    Mat mat_out_dev_nchw(DEVICE_NAIVE, NCHW_FLOAT, mat_out_dev_nchw_data);
    ret = device_converter.ConvertToMat(mat_out_dev_nchw, to_mat_param, device_command_queue);
    if (ret != TNN_OK) {
        LOGE("device converter convert blob to mat failed, mat type: %d\n", NCHW_FLOAT);
        FAIL();
    }
    int cmp_result    = 0;
    float compare_eps = blob_data_type == DATA_TYPE_INT8 ? max_i8_diff + 0.01 : 0.01;

    cmp_result |= CompareData(static_cast<float*>(mat_out_ref_nchw_data), static_cast<float*>(mat_out_dev_nchw_data),
                              out_nchw_size, compare_eps);

    Mat mat_out_ref(DEVICE_NAIVE, mat_type, dims, mat_out_ref_data);
    Mat mat_out_dev(DEVICE_NAIVE, mat_type, dims, mat_out_dev_data);
    if (mat_type != NCHW_FLOAT && dev != DEVICE_ARM) {
        to_mat_param.scale = {scale, scale, scale, scale};
        to_mat_param.bias  = {bias, bias, bias, bias};
        ret = cpu_converter.ConvertToMat(mat_out_ref, to_mat_param, NULL);
        if (ret != TNN_OK) {
            LOGE("cpu converter convert blob to mat failed, mat type: %d\n", mat_type);
            FAIL();
        }
        ret = device_converter.ConvertToMat(mat_out_dev, to_mat_param, device_command_queue);
        if (ret != TNN_OK) {
            LOGE("device converter convert blob to mat failed, mat type: %d\n", mat_type);
            FAIL();
        }
        cmp_result |= CompareData(static_cast<uint8_t*>(mat_out_ref_data), static_cast<uint8_t*>(mat_out_dev_data),
                                  mat_channel, channel, out_size);
    }

    EXPECT_EQ(0, cmp_result);

    free(mat_in_data);
    free(mat_out_ref_nchw_data);
    free(mat_out_dev_nchw_data);
    free(mat_out_ref_data);
    free(mat_out_dev_data);
}

}  // namespace TNN_NS
