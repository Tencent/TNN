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

void printinput(const uint8_t* ref_data, size_t n) {
    for (unsigned long long i = 0; i < n; i++) {
        LOGE("ERROR AT %llu result:%d\n", i, ref_data[i]);
    }
}

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

INSTANTIATE_TEST_SUITE_P(MatConverterTest, MatConverterTest,
                         ::testing::Combine(
                            // batch
                            testing::Values(1),
                            // channel
                            testing::Values(1, 3, 4),
                            // inputsize
                            testing::Values(158,384,720,580,640),
                            // mat type
                            testing::Values(N8UC4, N8UC3, NGRAY,
                                            NCHW_FLOAT)
                            // converter test param
                            ));

TEST_P(MatConverterTest, MatConverterTest) {
    int batch               = std::get<0>(GetParam());
    int channel             = std::get<1>(GetParam());
    int input_size          = std::get<2>(GetParam());
    MatType mat_type        = std::get<3>(GetParam());
    int output_size = 224;
    if (mat_type != N8UC4 || channel != 4) {
        GTEST_SKIP();
    }
    int mat_channel = channel;
    int in_size       = batch * mat_channel * input_size * input_size;
    int out_size      = in_size;
    int out_nchw_size = batch * channel * input_size * input_size;
    int cmp_result = 0;
    DimsVector dims = {batch, channel, input_size, input_size};
    DimsVector dims_gpu = {batch, 4, input_size, input_size};
    void* mat_in_data           = nullptr;
    void* mat_out_ref_nchw_data = nullptr;
    void* mat_out_dev_nchw_data = nullptr;
    void* mat_out_ref_data      = nullptr;
    void* mat_out_dev_data      = nullptr;
    void* mat_in_data_arm      = nullptr;
    mat_out_ref_nchw_data = malloc(out_nchw_size * sizeof(float));
    mat_out_dev_nchw_data = malloc(out_nchw_size * sizeof(float));

    if (mat_type == NCHW_FLOAT) {
        mat_in_data = malloc(in_size * sizeof(float));
        InitRandom(static_cast<float*>(mat_in_data), in_size, 0.0f, 1.0f);
        mat_out_ref_data = malloc(out_size * sizeof(float));
        mat_out_dev_data = malloc(out_size * sizeof(float));
    } else {
        mat_in_data = malloc(in_size * sizeof(uint8_t));
        mat_in_data_arm = malloc(in_size * sizeof(uint8_t));
        InitRandom(static_cast<uint8_t*>(mat_in_data), in_size, static_cast<uint8_t>(0), static_cast<uint8_t>(255));
        memcpy(mat_in_data_arm, mat_in_data, in_size);
        mat_out_ref_data = malloc(out_size * sizeof(uint8_t));
        mat_out_dev_data = malloc(batch * channel * output_size * output_size * sizeof(uint8_t));
    }

    void* device_command_queue;
    device_context_->GetCommandQueue(&device_command_queue);
    CropParam testparam;
    testparam.top_left_x = 5;
    testparam.top_left_y = 5;
    testparam.width      = 5;
    testparam.height     = 5;
    ResizeParam resize_param;
    resize_param.scale_w = 1.25f;
    resize_param.scale_h = 1.25f;

    void* mat_out_cpu_data = malloc((batch * mat_channel * output_size * output_size) * sizeof(uint8_t));
    DimsVector dims_gpu_out = {batch, channel, output_size, output_size};
    // DimsVector dims_gpu_out = {batch, 4, testparam.width, testparam.height};

    Mat cpu_in(DEVICE_NAIVE, mat_type, dims, mat_in_data);
    // void* mat_out_cpu_data = malloc((testparam.width*testparam.height*channel*batch) * sizeof(uint8_t));


    // printinput(static_cast<uint8_t*>(mat_in_data),out_size);
    // printinput(static_cast<uint8_t*>(mat_in_data_arm),out_size);
    Mat arm_device_in(DEVICE_ARM, N8UC4, dims_gpu, mat_in_data_arm);
    Mat arm_device_out(DEVICE_ARM, N8UC4, dims_gpu_out, mat_out_cpu_data);
    Mat *src_arm,*dst_arm;
    src_arm = &arm_device_in;
    dst_arm = &arm_device_out;
    MatConverter arm_converter(src_arm,dst_arm);
    arm_converter.Resize(arm_device_in, arm_device_out, resize_param, NULL);
    

    Mat device_in(DEVICE_OPENCL, N8UC4, dims_gpu);
    Mat device_out(DEVICE_OPENCL, N8UC4, dims_gpu_out);
    Mat cpu_out(DEVICE_NAIVE, mat_type, dims_gpu_out, mat_out_dev_data);
    // Mat device_out_copy(DEVICE_NAIVE, mat_type, dims_gpu_out, mat_out_cpu_data);
    Mat *src,*dst,*dst_copy;
    src = &cpu_in;
    dst = &device_in;
    // dst_copy = &device_out_copy;
    MatConverter gpu_converter(src,dst);
    printf("copy start\n");
    gpu_converter.Copy(cpu_in, device_in, device_command_queue);
    gpu_converter.Resize(device_in, device_out, resize_param, device_command_queue);
    gpu_converter.Copy(device_out, cpu_out, device_command_queue);
    // MatConverter cpu_converter(src,dst_copy);
    // cpu_converter.Copy(cpu_in, device_out_copy, NULL);
    printf("copy down\n");

    // cmp_result |= CompareData(static_cast<uint8_t*>(mat_out_dev_data), static_cast<uint8_t*>(mat_in_data),
    //                               channel, channel, out_size);
    //gpu_converter.Crop(cpu_src,cpu_dst,testparam,NULL);
    // cmp_result |= CompareData(static_cast<uint8_t*>(mat_out_dev_data), static_cast<uint8_t*>(mat_out_cpu_data),
    //                               channel, channel, testparam.width*testparam.height*channel*batch);
    cmp_result |= CompareData(static_cast<uint8_t*>(mat_out_dev_data), static_cast<uint8_t*>(mat_out_cpu_data),
                                  channel, channel, batch * mat_channel * output_size * output_size);
    // cmp_result |= CompareData(static_cast<float*>(mat_out_dev_data), static_cast<float*>(mat_in_data),
    //                               out_size, 0.01);
    // printf("mat_out_cpu_data :\n");

    // for(int i = 0; i < 100; ++i)
    //     printf("%hhu ", a[i]);
    // printf("mat_out_dev_data :\n");
    // for(int i = 0; i < 100; ++i)
    //     printf("%hhu ", b[i]);
    EXPECT_EQ(0, cmp_result);
    free(mat_in_data);
    free(mat_out_ref_nchw_data);
    free(mat_out_dev_nchw_data);
    free(mat_out_ref_data);
    free(mat_out_dev_data);
    free(mat_out_cpu_data);
}

}  // namespace TNN_NS
