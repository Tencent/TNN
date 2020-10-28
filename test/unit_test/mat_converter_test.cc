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
#include "tnn/utils/mat_utils.h"
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
   SetUpEnvironment(&cpu_, &device_, &cpu_context_, &device_context_);
}

void MatConverterTest::TearDownTestCase() {
    delete cpu_context_;
    delete device_context_;
}

int MatConverterTest::CreateTestData(int batch, int channel, int input_size, MatType mat_type, int output_size) {
    int mat_channel;
    if (mat_type == N8UC4) {
        mat_channel = 4;
    } else {
        mat_channel = channel;
    }

    int in_size             = batch * mat_channel * input_size * input_size;
    out_size_               = batch * mat_channel * output_size * output_size;
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

int MatConverterTest::DestroyTestData() {
    free(mat_in_data_);
    free(mat_out_ref_data_);
    free(mat_out_dev_data_);

    return 0;
}

bool MatConverterTest::OpenCLTestFilter(const DeviceType& device_type, const MatType& mat_type) {
    return device_type == DEVICE_OPENCL && mat_type != N8UC4;
}

bool MatConverterTest::MetalTestFilter(const DeviceType& device_type, const MatType& mat_type,
                                       const MatConverterType& mat_converter_type, const int batch) {
    // Metal device only supports NCHW_FLOAT and N8UC4 mat
    if (device_type == DEVICE_METAL && !(mat_type == N8UC4 || mat_type == NCHW_FLOAT)) {
        return true;
    }
    // Only Copy supports NCHW_FLOAT
    if (device_type == DEVICE_METAL && mat_type == NCHW_FLOAT && mat_converter_type != MatConverterType::Copy) {
        return true;
    }
    // Metal device only supports N8UC4 mat with batchsize = 1
    if (device_type == DEVICE_METAL && mat_type == N8UC4 && batch != 1) {
        return true;
    }
    // Disable interpolation-related tests on Metal
    if (device_type == DEVICE_METAL && (mat_converter_type == MatConverterType::WarpAffine ||
                mat_converter_type == MatConverterType::Resize)) {
        return true;
    }
    return false;
}

bool MatConverterTest::MatChannelCheck(const MatType& mat_type, const int channel) {
    return (mat_type == NGRAY && channel != 1) ||
           (mat_type == N8UC3 && channel != 3) ||
           (mat_type == N8UC4 && channel != 4);
}

bool MatConverterTest::CvtColorCheck(const DeviceType& device_type, const MatType& mat_type,
                                     const MatConverterType& mat_converter_type,
                                     const ColorConversionType& cvt_type,
                                     const int input_size) {
    if (mat_converter_type == MatConverterType::CvtColor) {
        if (device_type != DEVICE_ARM) {
            return true;
        }
        if (cvt_type == COLOR_CONVERT_BGRTOGRAY && mat_type != N8UC3) {
            return true;
        }
        if (cvt_type == COLOR_CONVERT_BGRATOGRAY && mat_type != N8UC4) {
            return true;
        }
        if ((cvt_type == COLOR_CONVERT_NV12TOBGR || cvt_type == COLOR_CONVERT_NV21TOBGR) &&
            (mat_type != N8UC3 || input_size % 2 != 0)) {
            return true;
        }
        if ((cvt_type == COLOR_CONVERT_NV12TOBGRA || cvt_type == COLOR_CONVERT_NV21TOBGRA) &&
            (mat_type != N8UC4 || input_size % 2 != 0)) {
            return true;
        }
    }
    return false;
}

void MatConverterTest::GetOutputSize(const MatConverterTestParam& mat_converter_test_param,
                                     const MatConverterType& mat_converter_type,
                                     const int input_size,
                                     int& output_size) {
    if (mat_converter_type == MatConverterType::Resize){
        output_size = int(round(mat_converter_test_param.resize_param.scale_h * input_size));
    } else if (mat_converter_type == MatConverterType::Crop) {
        output_size = mat_converter_test_param.crop_param.width;
    } else if (mat_converter_type == MatConverterType::CopyMakeBorder) {
        output_size = input_size + mat_converter_test_param.copy_make_border_param.top +
                      mat_converter_test_param.copy_make_border_param.bottom;
    } else {
        output_size = input_size;
    }
}

INSTANTIATE_TEST_SUITE_P(MatConverterTest, MatConverterTest,
                         ::testing::Combine(
                            // batch
                            testing::Values(1, 2), 
                            // channel
                            testing::Values(1, 3, 4), 
                            // inputsize
                            testing::Values(20, 21, 26, 27),
                            // mat type
                            testing::Values(N8UC4, N8UC3, NGRAY),
                            // converter test param
                            testing::Values(
                                // Copy
                                MatConverterTestParam(MatConverterType::Copy),
                                // Resize
                                MatConverterTestParam(MatConverterType::Resize, 0.5, 0.5, INTERP_TYPE_LINEAR),
                                MatConverterTestParam(MatConverterType::Resize, 0.5, 0.5, INTERP_TYPE_NEAREST),
                                // Crop
                                MatConverterTestParam(MatConverterType::Crop, 0, 0, 10, 10),
                                MatConverterTestParam(MatConverterType::Crop, 5, 5, 10, 10),
                                MatConverterTestParam(MatConverterType::Crop, 3, 7, 10, 10),
                                MatConverterTestParam(MatConverterType::Crop, 7, 3, 10, 10),
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
                                                      INTERP_TYPE_LINEAR, BORDER_TYPE_CONSTANT, FLT_MIN),
                                // CvtColor
                                MatConverterTestParam(MatConverterType::CvtColor, COLOR_CONVERT_BGRTOGRAY),
                                MatConverterTestParam(MatConverterType::CvtColor, COLOR_CONVERT_BGRATOGRAY),
                                MatConverterTestParam(MatConverterType::CvtColor, COLOR_CONVERT_NV12TOBGR),
                                MatConverterTestParam(MatConverterType::CvtColor, COLOR_CONVERT_NV21TOBGR),
                                MatConverterTestParam(MatConverterType::CvtColor, COLOR_CONVERT_NV12TOBGRA),
                                MatConverterTestParam(MatConverterType::CvtColor, COLOR_CONVERT_NV21TOBGRA),
                                // CopyMakeBorder
                                MatConverterTestParam(MatConverterType::CopyMakeBorder, 0, 10, 0, 10,
                                                      BORDER_TYPE_CONSTANT, 0.0),
                                MatConverterTestParam(MatConverterType::CopyMakeBorder, 5, 7, 5, 7,
                                                      BORDER_TYPE_CONSTANT, 255.0),
                                MatConverterTestParam(MatConverterType::CopyMakeBorder, 3, 9, 7, 5,
                                                      BORDER_TYPE_CONSTANT, 100.0),
                                MatConverterTestParam(MatConverterType::CopyMakeBorder, 7, 3, 3, 7,
                                                      BORDER_TYPE_CONSTANT, 50.0)
                                                      )
                            ));

#define CHECK_STATUS                                        \
    if (status != TNN_OK) {                                 \
        std::cout << status.description() << std::endl;     \
        FAIL();                                             \
    }

TEST_P(MatConverterTest, MatConverterTest) {
    int batch                                       = std::get<0>(GetParam());
    int channel                                     = std::get<1>(GetParam());
    int input_size                                  = std::get<2>(GetParam());
    MatType mat_type                                = std::get<3>(GetParam());
    MatConverterTestParam mat_converter_test_param  = std::get<4>(GetParam());
    MatConverterType mat_converter_type             = mat_converter_test_param.mat_converter_type;
    ColorConversionType cvt_type                    = mat_converter_test_param.cvt_type;

    DeviceType device_type  = ConvertDeviceType(FLAGS_dt);
    // warp affine/resize only support N8UC4 on OpenCL for now
    if (OpenCLTestFilter(device_type, mat_type) ||
        MetalTestFilter(device_type, mat_type, mat_converter_type, batch)) {
        GTEST_SKIP();
    }
    if (MatChannelCheck(mat_type, channel) ||
        CvtColorCheck(device_type, mat_type, mat_converter_type, cvt_type, input_size)) {
        GTEST_SKIP();
    }

    int output_size;
    GetOutputSize(mat_converter_test_param, mat_converter_type, input_size, output_size);

    DimsVector dims         = {batch, channel, input_size, input_size};
    DimsVector dims_out     = {batch, channel, output_size, output_size};;
    int rtn = CreateTestData(batch, channel, input_size, mat_type, output_size);
    EXPECT_EQ(rtn, 0);

    Mat cpu_in_mat          = Mat(DEVICE_NAIVE, mat_type, dims, mat_in_data_);
    Mat cpu_ref_mat         = Mat(DEVICE_NAIVE, mat_type, dims_out, mat_out_ref_data_);
    Mat cpu_out_mat         = Mat(DEVICE_NAIVE, mat_type, dims_out, mat_out_dev_data_);
    Mat device_mat          = Mat(device_type, mat_type, dims_out);
    Mat device_in_mat       = Mat(device_type, mat_type, dims);
    int cmp_result          = 0;
    void* device_command_queue;
    device_context_->GetCommandQueue(&device_command_queue);
    switch (mat_converter_type)
    {
        case MatConverterType::Copy:
        {
            TNN_NS::Status status = MatUtils::Copy(cpu_in_mat, device_mat, device_command_queue);
            CHECK_STATUS;

            status = MatUtils::Copy(device_mat, cpu_out_mat, device_command_queue);
            CHECK_STATUS;

            cmp_result |= CompareData(static_cast<uint8_t*>(mat_out_dev_data_), static_cast<uint8_t*>(mat_in_data_),
                                      channel, channel, out_size_);

            EXPECT_EQ(0, cmp_result);
            break;
        }
        case MatConverterType::Resize:
        {
            TNN_NS::Status status = MatUtils::Resize(cpu_in_mat, cpu_ref_mat, mat_converter_test_param.resize_param, NULL);
            CHECK_STATUS;

            status = MatUtils::Copy(cpu_in_mat, device_in_mat,
                                           device_command_queue);
            status = MatUtils::Resize(device_in_mat, device_mat,
                                                 mat_converter_test_param.resize_param,
                                                 device_command_queue);
            CHECK_STATUS;

            MatUtils::Copy(device_mat, cpu_out_mat, device_command_queue);
            cmp_result |= CompareData(static_cast<uint8_t*>(mat_out_ref_data_), static_cast<uint8_t*>(mat_out_dev_data_),
                                      channel, channel, out_size_);
            EXPECT_EQ(0, cmp_result);
            break;
        }
        case MatConverterType::Crop:
        {
            TNN_NS::Status status = MatUtils::Crop(cpu_in_mat, cpu_ref_mat, mat_converter_test_param.crop_param, NULL);
            CHECK_STATUS;

            status = MatUtils::Copy(cpu_in_mat, device_in_mat,
                                           device_command_queue);
            status = MatUtils::Crop(device_in_mat, device_mat,
                                                 mat_converter_test_param.crop_param,
                                                 device_command_queue);
            CHECK_STATUS;

            MatUtils::Copy(device_mat, cpu_out_mat, device_command_queue);
            cmp_result |= CompareData(static_cast<uint8_t*>(mat_out_ref_data_), static_cast<uint8_t*>(mat_out_dev_data_),
                                      channel, channel, out_size_);
            EXPECT_EQ(0, cmp_result);
            break;
        }
        case MatConverterType::WarpAffine:
        {
            TNN_NS::Status status = MatUtils::WarpAffine(cpu_in_mat, cpu_ref_mat,
                                                           mat_converter_test_param.warp_affine_param,
                                                           device_command_queue);
            CHECK_STATUS;

            status = MatUtils::Copy(cpu_in_mat, device_in_mat,
                                           device_command_queue);
            status = MatUtils::WarpAffine(device_in_mat, device_mat,
                                                 mat_converter_test_param.warp_affine_param,
                                                 device_command_queue);
            CHECK_STATUS;

            MatUtils::Copy(device_mat, cpu_out_mat, device_command_queue);
            cmp_result |= CompareData(static_cast<uint8_t*>(mat_out_ref_data_), static_cast<uint8_t*>(mat_out_dev_data_),
                                      channel, channel, out_size_);
            EXPECT_EQ(0, cmp_result);
            break;
        }
        case MatConverterType::CvtColor:
        {
            TNN_NS::Status status = MatUtils::CvtColor(cpu_in_mat, cpu_ref_mat, mat_converter_test_param.cvt_type, NULL);
            CHECK_STATUS;

            status = MatUtils::Copy(cpu_in_mat, device_in_mat,
                                           device_command_queue);
            status = MatUtils::CvtColor(device_in_mat, device_mat,
                                                 mat_converter_test_param.cvt_type,
                                                 device_command_queue);
            CHECK_STATUS;

            MatUtils::Copy(device_mat, cpu_out_mat, device_command_queue);

            if (mat_converter_test_param.cvt_type == COLOR_CONVERT_BGRTOGRAY ||
                mat_converter_test_param.cvt_type == COLOR_CONVERT_BGRATOGRAY) {
                out_size_ /= channel;
            }
            cmp_result |= CompareData(static_cast<uint8_t*>(mat_out_ref_data_), static_cast<uint8_t*>(mat_out_dev_data_),
                                      channel, channel, out_size_);
            EXPECT_EQ(0, cmp_result);
            break;
        }
        case MatConverterType::CopyMakeBorder:
        {
            TNN_NS::Status status = MatUtils::CopyMakeBorder(cpu_in_mat, cpu_ref_mat,
                                                             mat_converter_test_param.copy_make_border_param, NULL);
            CHECK_STATUS;

            status = MatUtils::Copy(cpu_in_mat, device_in_mat,
                                           device_command_queue);
            status = MatUtils::CopyMakeBorder(device_in_mat, device_mat,
                                                 mat_converter_test_param.copy_make_border_param,
                                                 device_command_queue);
            CHECK_STATUS;

            MatUtils::Copy(device_mat, cpu_out_mat, device_command_queue);
            cmp_result |= CompareData(static_cast<uint8_t*>(mat_out_ref_data_), static_cast<uint8_t*>(mat_out_dev_data_),
                                      channel, channel, out_size_);
            EXPECT_EQ(0, cmp_result);
            break;
        }
    }
    rtn = DestroyTestData();
    EXPECT_EQ(rtn, 0);
}

#undef CHECK_STATUS

}  // namespace TNN_NS
