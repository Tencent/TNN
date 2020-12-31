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

#include "test/unit_test/layer_test/layer_test.h"

#include <sys/time.h>
#include "test/unit_test/unit_test_common.h"
#include "test/unit_test/utils/network_helpers.h"
#include "tnn/core/blob_int8.h"
#include "tnn/utils/bfp16.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/blob_memory_size_utils.h"
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

std::shared_ptr<Instance> LayerTest::instance_cpu_       = nullptr;
std::shared_ptr<Instance> LayerTest::instance_device_    = nullptr;
std::shared_ptr<Instance> LayerTest::instance_ocl_cache_ = nullptr;

void LayerTest::SetUpTestCase() {}

void LayerTest::Run(std::shared_ptr<AbstractModelInterpreter> interp, Precision precision) {
#if defined(__OBJC__) && defined(__APPLE__)
    @autoreleasepool{
#endif
    TNN_NS::Status ret = TNN_NS::TNN_OK;

    ret = Init(interp, precision);
    if (ret != TNN_OK) {
        EXPECT_EQ((int)ret, TNN_OK);
        DeInit();
        return;
    }

    ret = InitInputBlobsDataRandom();
    if (ret != TNN_OK) {
        EXPECT_EQ((int)ret, TNN_OK);
        DeInit();
        return;
    }

    ret = Forward();
    if (ret != TNN_OK) {
        EXPECT_EQ((int)ret, TNN_OK);
        DeInit();
        return;
    }

#ifndef TNN_UNIT_TEST_BENCHMARK
    // Compare the result for both cpu and device layer
    ret = Compare();
    if (ret != TNN_OK) {
        EXPECT_EQ((int)ret, TNN_OK);
        DeInit();
        return;
    }
#endif

    DeInit();
    if (ret != TNN_OK) {
        EXPECT_EQ((int)ret, TNN_OK);
        return;
    }
#if defined(__OBJC__) && defined(__APPLE__)
    }
#endif
}

Status LayerTest::Init(std::shared_ptr<AbstractModelInterpreter> interp, Precision precision) {
    TNN_NS::Status ret = TNN_NS::TNN_OK;

    ModelConfig model_config;
    model_config.params.push_back("");
    model_config.params.push_back("");

    NetworkConfig config_cpu;
    config_cpu.device_type = DEVICE_NAIVE;

    NetworkConfig config_device;
    config_device.device_type = ConvertDeviceType(FLAGS_dt);
    config_device.enable_tune_kernel = FLAGS_et;
    if (DEVICE_HUAWEI_NPU == config_device.device_type) {
        config_device.network_type = NETWORK_TYPE_HUAWEI_NPU;
    }
    if (DEVICE_CUDA == config_device.device_type) {
        config_device.network_type = NETWORK_TYPE_TENSORRT;
    }
    if (!FLAGS_ub && (DEVICE_OPENCL == config_device.device_type || DEVICE_METAL == config_device.device_type)) {
        config_device.precision = PRECISION_HIGH;
    } else {
        config_device.precision = precision;
    }
    if (FLAGS_lp.length() > 0) {
        config_device.library_path = {FLAGS_lp};
    }

    instance_cpu_ = std::make_shared<Instance>(config_cpu, model_config);
    if (nullptr == instance_cpu_) {
        LOGE("tnn create cpu instance falied\n");
        return Status(TNNERR_NULL_PARAM, "instance is null");
    }

    instance_device_ = std::make_shared<Instance>(config_device, model_config);
    if (nullptr == instance_device_) {
        LOGE("tnn create device instance falied\n");
        return Status(TNNERR_NULL_PARAM, "instance is null");
    }

    InputShapesMap input_shape = InputShapesMap();
    ret                        = instance_cpu_->Init(interp, input_shape);
    if (ret != TNN_OK) {
        LOGE("tnn init cpu instance falied (%s)\n", ret.description().c_str());
        return ret;
    }
    ret = instance_device_->Init(interp, input_shape);
    if (ret != TNN_OK) {
        LOGE("tnn init device instance falied (%s)\n", ret.description().c_str());
        return ret;
    }

    if (nullptr == instance_ocl_cache_ && DEVICE_OPENCL == config_device.device_type) {
        instance_ocl_cache_ = std::make_shared<Instance>(config_device, model_config);
        if (nullptr == instance_ocl_cache_) {
            LOGE("tnn create ocl cache instance falied\n");
            return Status(TNNERR_NULL_PARAM, "instance is null");
        }

        ret = instance_ocl_cache_->Init(interp, input_shape);
        if (ret != TNN_OK) {
            LOGE("tnn init device instance falied\n");
            return ret;
        }
    }

    return ret;
}

Status LayerTest::Forward() {
    TNN_NS::Status ret = TNN_NS::TNN_OK;

#ifndef TNN_UNIT_TEST_BENCHMARK
    ret = instance_cpu_->Forward();
    EXPECT_EQ_OR_RETURN(ret, TNN_OK);
#endif

#if TNN_PROFILE && defined(TNN_UNIT_TEST_BENCHMARK)
    instance_device_->StartProfile();
#endif

    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    gettimeofday(&time1, &zone);
    float min = FLT_MAX, max = FLT_MIN, sum = 0.0f;
    for (int i = 0; i < FLAGS_ic; ++i) {
        gettimeofday(&time1, &zone);

        ret = instance_device_->Forward();
        EXPECT_EQ_OR_RETURN(ret, TNN_OK);

        gettimeofday(&time2, &zone);
        float delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
        min         = fmin(min, delta);
        max         = fmax(max, delta);
        sum += delta;
    }

#if TNN_PROFILE && defined(TNN_UNIT_TEST_BENCHMARK)
    instance_device_->FinishProfile(true);
#endif

    /*
     * shows the timings of device layer.
     * Used for benchmarking.
     */
    if (FLAGS_ub) {
        printf("device %s time cost: min =   %g ms  |  max =  %g ms  |  avg = %g ms\n", FLAGS_dt.c_str(), min, max,
               sum / (float)FLAGS_ic);
    }

    return ret;
}

Status LayerTest::Compare() {
    BlobMap output_blobs_cpu;
    BlobMap output_blobs_device;
    Status ret = TNN_OK;
    ret        = instance_cpu_->GetAllOutputBlobs(output_blobs_cpu);
    if (ret != TNN_OK)
        return ret;
    ret = instance_device_->GetAllOutputBlobs(output_blobs_device);
    if (ret != TNN_OK)
        return ret;

    void* command_queue;
    ret = instance_device_->GetCommandQueue(&command_queue);
    if (ret != TNN_OK) {
        LOGE("get device command queue failed (%s)\n", ret.description().c_str());
        return ret;
    }

    int cmp_result = 0;
    for (auto blob_item : output_blobs_cpu) {
        cmp_result =
            CompareBlob(output_blobs_cpu[blob_item.first], output_blobs_device[blob_item.first], command_queue);
        if (cmp_result != 0) {
            break;
        }
    }

    EXPECT_EQ(0, cmp_result);
    return TNN_OK;
}

Status LayerTest::DeInit() {
    instance_cpu_.reset();
    instance_device_.reset();

    return TNN_OK;
}

void LayerTest::TearDownTestCase() {
    instance_cpu_.reset();
    instance_device_.reset();
}

Status LayerTest::GenerateRandomBlob(Blob* cpu_blob, Blob* device_blob, void* command_queue_dev, int magic_num) {
    Status ret = TNN_OK;
    // init cpu input blob
    BlobDesc blob_desc                = cpu_blob->GetBlobDesc();
    BlobMemorySizeInfo blob_size_info = Calculate1DMemorySize(blob_desc);
    int blob_count                    = DimsVectorUtils::Count(blob_size_info.dims);

    BlobDesc blob_desc_device = device_blob->GetBlobDesc();
    MatType mat_type          = NCHW_FLOAT;
    if (blob_desc_device.data_type == DATA_TYPE_BFP16) {
        // the value is initialized as bfp16
        mat_type = RESERVED_BFP16_TEST;
    } else if (blob_desc_device.data_type == DATA_TYPE_INT8) {
        // the value is initialized as int8
        mat_type = RESERVED_INT8_TEST;
    } else if (blob_desc_device.data_type == DATA_TYPE_HALF) {
        // the value is initialized as half
        mat_type = RESERVED_FP16_TEST;
    }
    TNN_NS::Mat source(DEVICE_NAIVE, mat_type, blob_desc.dims);
    void* input_data = source.GetData();
    if (mat_type == NCHW_FLOAT) {
        if (ensure_input_positive_) {
            // some layers only supports positive data as input
            InitRandom(static_cast<float*>(input_data), blob_count, 0.0001f, 1.0f + (float)magic_num);
        } else {
            InitRandom(static_cast<float*>(input_data), blob_count, 1.0f + (float)magic_num);
        }
    } else if (mat_type == RESERVED_FP16_TEST) {
        if (ensure_input_positive_) {
            // some layers only supports positive values as input
            InitRandom(static_cast<fp16_t*>(input_data), blob_count, (fp16_t)0.0f, (fp16_t)(1.0f + magic_num));
        } else {
            InitRandom(static_cast<fp16_t*>(input_data), blob_count, (fp16_t)(1.0f + magic_num));
        }
    } else if (mat_type == RESERVED_INT8_TEST) {
        if (ensure_input_positive_) {
            // some layers only supports positive values as input
            InitRandom(static_cast<int8_t*>(input_data), blob_count, (int8_t)0, (int8_t)8);
        } else {
            InitRandom(static_cast<int8_t*>(input_data), blob_count, (int8_t)8);
        }
    } else if (mat_type == RESERVED_BFP16_TEST) {
        if (ensure_input_positive_) {
            InitRandom(static_cast<bfp16_t*>(input_data), blob_count, bfp16_t(0.f), bfp16_t(1.0f + magic_num));
        } else {
            InitRandom(static_cast<bfp16_t*>(input_data), blob_count, bfp16_t(1.0f + magic_num));
        }
    }

    // default param for the blob_converter
    MatConvertParam param;
    param.scale = std::vector<float>(blob_desc.dims[1], 1);
    param.bias  = std::vector<float>(blob_desc.dims[1], 0);

    // CONVERT TO CPU BLOB
    BlobConverter blob_converter_cpu(cpu_blob);
    ret = blob_converter_cpu.ConvertFromMat(source, param, nullptr);
    if (ret != TNN_OK) {
        LOGE("input blob_converter failed (%s)\n", ret.description().c_str());
        return ret;
    }

    // CONVERT TO DEVICE BLOB
    BlobConverter blob_converter(device_blob);
    ret = blob_converter.ConvertFromMat(source, param, command_queue_dev);
    if (ret != TNN_OK) {
        LOGE("input blob_converter failed (%s)\n", ret.description().c_str());
        return ret;
    }
    return ret;
}

int LayerTest::CompareBlob(Blob* cpu_blob, Blob* device_blob, void* command_queue_dev) {
    Status ret            = TNN_OK;
    auto blob_desc_device = device_blob->GetBlobDesc();
    // mat type for both
    MatType mat_type = NCHW_FLOAT;
    if (blob_desc_device.data_type == DATA_TYPE_BFP16) {
        mat_type = RESERVED_BFP16_TEST;
    } else if (blob_desc_device.data_type == DATA_TYPE_INT8) {
        mat_type = RESERVED_INT8_TEST;
    }
    auto dims = cpu_blob->GetBlobDesc().dims;
    int count = DimsVectorUtils::Count(dims);
    // convert cpu blob to mat
    TNN_NS::Mat cpu_mat(DEVICE_NAIVE, mat_type, dims);
    BlobConverter blob_converter_cpu(cpu_blob);
    ret = blob_converter_cpu.ConvertToMat(cpu_mat, MatConvertParam(), nullptr);
    if (ret != TNN_OK) {
        LOGE("output blob_converter failed (%s)\n", ret.description().c_str());
        return -1;
    }

    // convert dev blob to cpu mat nchw
    TNN_NS::Mat dev_cpu_mat(DEVICE_NAIVE, mat_type, dims);
    BlobConverter blob_converter_dev(device_blob);
    ret = blob_converter_dev.ConvertToMat(dev_cpu_mat, MatConvertParam(), command_queue_dev);
    if (ret != TNN_OK) {
        LOGE("output blob_converter failed (%s)\n", ret.description().c_str());
        return -1;
    }

    // compare data
    int cmp_result = 0;
    if (blob_desc_device.data_type == DATA_TYPE_FLOAT) {
        cmp_result |= CompareData(static_cast<float*>(cpu_mat.GetData()), static_cast<float*>(dev_cpu_mat.GetData()),
                                  count, 0.01, 0.0001);
    } else if (blob_desc_device.data_type == DATA_TYPE_HALF) {
        cmp_result |= CompareData(static_cast<float*>(cpu_mat.GetData()), static_cast<float*>(dev_cpu_mat.GetData()),
                                  count, 0.01, 0.001);
    } else if (blob_desc_device.data_type == DATA_TYPE_BFP16) {
        cmp_result |= CompareData(static_cast<bfp16_t*>(cpu_mat.GetData()),
                                  static_cast<bfp16_t*>(dev_cpu_mat.GetData()), count, 0.05);
    } else if (blob_desc_device.data_type == DATA_TYPE_INT8) {
        cmp_result |=
            CompareData(static_cast<int8_t*>(cpu_mat.GetData()), static_cast<int8_t*>(dev_cpu_mat.GetData()), count);
    } else {
        LOGE("UNKNOWN DATA TYPE!");
    }

    if (cmp_result != 0) {
        LOGE("cpu_mat.GetData(): %.6f %.6f %.6f %.6f\n", static_cast<float*>(cpu_mat.GetData())[0],
             static_cast<float*>(cpu_mat.GetData())[1], static_cast<float*>(cpu_mat.GetData())[2],
             static_cast<float*>(cpu_mat.GetData())[3]);
        LOGE("dev_cpu_mat.GetData(): %.6f %.6f %.6f %.6f\n", static_cast<float*>(dev_cpu_mat.GetData())[0],
             static_cast<float*>(dev_cpu_mat.GetData())[1], static_cast<float*>(dev_cpu_mat.GetData())[2],
             static_cast<float*>(dev_cpu_mat.GetData())[3]);
    }

    return cmp_result;
}

Status LayerTest::InitInputBlobsDataRandom() {
    BlobMap input_blobs_cpu;
    BlobMap input_blobs_device;
    Status ret = TNN_OK;
    ret        = instance_cpu_->GetAllInputBlobs(input_blobs_cpu);
    if (ret != TNN_OK)
        return ret;
    ret = instance_device_->GetAllInputBlobs(input_blobs_device);
    if (ret != TNN_OK)
        return ret;

    // CONVERT TO DEVICE BLOB
    void* command_queue;
    ret = instance_device_->GetCommandQueue(&command_queue);
    if (ret != TNN_OK) {
        LOGE("get device command queue failed (%s)\n", ret.description().c_str());
        return ret;
    }

    int index = 0;
    for (auto blob_item : input_blobs_cpu) {
        ret = GenerateRandomBlob(input_blobs_cpu[blob_item.first], input_blobs_device[blob_item.first], command_queue,
                                 index);
        if (ret != TNN_OK) {
            return ret;
        }

        index++;
    }

    return TNN_OK;
}

}  // namespace TNN_NS
