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

#include "test/unit_test/mat_converter_test.h"
#include "test/unit_test/unit_test_common.h"
#include "test/unit_test/unit_test_macro.h"
#include "tnn/core/blob_int8.h"
#include "tnn/utils/mat_converter.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "utils/network_helpers.h"

namespace TNN_NS {

AbstractDevice* MatConverterTest::cpu_;
AbstractDevice* MatConverterTest::device_;
Context* MatConverterTest::cpu_context_;
Context* MatConverterTest::device_context_;

void MatConverterTest::SetUpTestCase() {
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

void MatConverterTest::TearDownTestCase() {
    delete cpu_context_;
    delete device_context_;
}

int MatConverterTest::CreateTestData(int batch, int channel, int input_size, MatType mat_type) {
    int mat_channel;
    if (mat_type == N8UC4) {
        mat_channel = 4;
    } else {
        mat_channel = channel;
    }

    int in_size             = batch * mat_channel * input_size * input_size;
    out_size_               = in_size;
    mat_in_data_            = nullptr;
    mat_out_ref_data_       = nullptr;
    mat_out_dev_data_       = nullptr;

    if (mat_type == NCHW_FLOAT) {
        mat_in_data_ = malloc(in_size * sizeof(float));
        InitRandom(static_cast<float*>(mat_in_data_), in_size, 0.0f, 1.0f);
        mat_out_ref_data_ = malloc(out_size_ * sizeof(float));
        mat_out_dev_data_ = malloc(out_size_ * sizeof(float));
    } else {
        mat_in_data_ = malloc(in_size * sizeof(uint8_t));
        InitRandom(static_cast<uint8_t*>(mat_in_data_), in_size, static_cast<uint8_t>(0), static_cast<uint8_t>(255));
        mat_out_ref_data_ = malloc(out_size_ * sizeof(uint8_t));
        mat_out_dev_data_ = malloc(out_size_ * sizeof(uint8_t));
    }

    return 0;
}

int MatConverterTest::DestroyTestData()
{
    free(mat_in_data_);
    free(mat_out_ref_data_);
    free(mat_out_dev_data_);

    return 0;
}

INSTANTIATE_TEST_SUITE_P(MatConverterTest, MatConverterTest,
                         ::testing::Combine(
                            // batch
                            testing::Values(1, 2),
                            // channel
                            testing::Values(1, 3, 4),
                            // input size
                            testing::Values(3, 10, 20, 128),
                            // mat type
                            testing::Values(N8UC4, N8UC3, NGRAY,
                                            NCHW_FLOAT),
                            // converter test param
                            testing::Values(
                                // Copy
                                // MatConverterTestParam(MatConverterType::Copy),
                                // Resize
                                // MatConverterTestParam(MatConverterType::Resize, 0.1, 0.1, INTERP_TYPE_LINEAR),
                                // Crop
                                // MatConverterTestParam(MatConverterType::Crop, 0, 0, 10, 10),
                                // WarpAffine
                                MatConverterTestParam(MatConverterType::WarpAffine, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                      INTERP_TYPE_LINEAR, BORDER_TYPE_CONSTANT, 0.0),
                                MatConverterTestParam(MatConverterType::WarpAffine, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                                      INTERP_TYPE_LINEAR, BORDER_TYPE_CONSTANT, 0.0),
                                MatConverterTestParam(MatConverterType::WarpAffine, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                                                      INTERP_TYPE_LINEAR, BORDER_TYPE_CONSTANT, FLT_MIN),
                                MatConverterTestParam(MatConverterType::WarpAffine, 1, 0, 1, 0, 1, 1,
                                                      INTERP_TYPE_LINEAR, BORDER_TYPE_CONSTANT, FLT_MIN),
                                MatConverterTestParam(MatConverterType::WarpAffine, 1, 0, 50, 0, 1, 100,
                                                      INTERP_TYPE_LINEAR, BORDER_TYPE_CONSTANT, FLT_MIN),
                                MatConverterTestParam(MatConverterType::WarpAffine, 2, 1, 100, 3, 7, 50,
                                                      INTERP_TYPE_LINEAR, BORDER_TYPE_CONSTANT, FLT_MIN),
                                MatConverterTestParam(MatConverterType::WarpAffine, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                                      INTERP_TYPE_LINEAR, BORDER_TYPE_CONSTANT, FLT_MIN))
                            ));

TEST_P(MatConverterTest, MatConverterTest) {
    int batch                                       = std::get<0>(GetParam());
    int channel                                     = std::get<1>(GetParam());
    int input_size                                  = std::get<2>(GetParam());
    MatType mat_type                                = std::get<3>(GetParam());
    MatConverterTestParam mat_converter_test_param  = std::get<4>(GetParam());
    MatConverterType mat_converter_type             = mat_converter_test_param.mat_converter_type;

    DeviceType device_type  = ConvertDeviceType(FLAGS_dt);
    // warp affine only support N8UC4 on OpenCL for now
    if (device_type == DEVICE_OPENCL && mat_converter_type == MatConverterType::WarpAffine && !(mat_type == N8UC4))
    {
        GTEST_SKIP();
    }

    if ((mat_type == NGRAY && channel != 1) || (mat_type == N8UC3 && channel != 3) || (mat_type == N8UC4 && channel != 4))
    {
        GTEST_SKIP();
    }

    int rtn = CreateTestData(batch, channel, input_size, mat_type);
    EXPECT_EQ(rtn, 0);


    DimsVector dims         = {batch, channel, input_size, input_size};
    Mat cpu_in_mat          = Mat(DEVICE_NAIVE, mat_type, dims, mat_in_data_);
    Mat cpu_ref_mat         = Mat(DEVICE_NAIVE, mat_type, dims, mat_out_ref_data_);
    Mat cpu_out_mat         = Mat(DEVICE_NAIVE, mat_type, dims, mat_out_dev_data_);
    Mat device_mat          = Mat(device_type, mat_type, dims);
    Mat device_in_mat       = Mat(device_type, mat_type, dims);
    int cmp_result          = 0;
    void* device_command_queue;
    device_context_->GetCommandQueue(&device_command_queue);
    switch (mat_converter_type)
    {
        case MatConverterType::Copy:
        {
            #if 0
            Mat *src, *dst;
            src = &cpu_in_mat;
            dst = &device_mat;
            MatConverter gpu_converter(src, dst);
            printf("copy start\n");
            gpu_converter.Copy(cpu_in_mat, device_mat, device_command_queue);

            gpu_converter.Copy(device_mat, cpu_out_mat, device_command_queue);
            printf("copy down\n");

            cmp_result |= CompareData(static_cast<uint8_t*>(mat_out_dev_data_), static_cast<uint8_t*>(mat_in_data_),
                                      channel, channel, out_size_);

            EXPECT_EQ(0, cmp_result);
            #endif
            break;
        }
        case MatConverterType::Resize:
        {
            LOGE("mat converter resize test start\n");
            LOGE("mat converter resize test end\n");
            break;
        }
        case MatConverterType::Crop:
        {
            LOGE("mat converter crop test start\n");
            LOGE("mat converter crop test end\n");
            break;
        }
        case MatConverterType::WarpAffine:
        {
            Mat *src, *dst, *dst_ref;
            src         = &cpu_in_mat;
            dst         = &device_mat;
            dst_ref     = &cpu_ref_mat;

            MatConverter host_converter(src, dst_ref);
            LOGE("warp affine on host start\n");
            tnn::Status status = host_converter.WarpAffine(cpu_in_mat, cpu_ref_mat,
                                                           mat_converter_test_param.warp_affine_param,
                                                           device_command_queue);
            if (status == TNN_OK)
            {
                LOGE("warp affine on host done\n");
            }
            else
            {
                LOGE("warp affine on host failed\n");
                FAIL();
            }

            MatConverter device_converter(src, dst);
            LOGE("warp affine on device start\n");
            status = device_converter.Copy(cpu_in_mat, device_in_mat,
                                           device_command_queue);

            status = device_converter.WarpAffine(device_in_mat, device_mat,
                                                 mat_converter_test_param.warp_affine_param,
                                                 device_command_queue);
            if (status == TNN_OK)
            {
                LOGE("warp affine on device done\n");
            }
            else
            {
                LOGE("warp affine on device failed\n");
                FAIL();
            }

            device_converter.Copy(device_mat, cpu_out_mat, device_command_queue);

            cmp_result |= CompareData(static_cast<uint8_t*>(mat_out_ref_data_), static_cast<uint8_t*>(mat_out_dev_data_),
                                      channel, channel, out_size_);

            EXPECT_EQ(0, cmp_result);

            break;
        }
    }

    rtn = DestroyTestData();
    EXPECT_EQ(rtn, 0);
}

}  // namespace TNN_NS
