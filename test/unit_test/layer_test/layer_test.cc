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
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

AbstractDevice* LayerTest::cpu_;
AbstractDevice* LayerTest::device_;
Context* LayerTest::cpu_context_;
Context* LayerTest::device_context_;

void LayerTest::SetUpTestCase() {
    SetUpEnvironment(&cpu_, &device_, &cpu_context_, &device_context_);
}

void LayerTest::Run(LayerType type, LayerParam* param, LayerResource* resource, std::vector<BlobDesc>& inputs_desc,
                    std::vector<BlobDesc>& outputs_desc) {
    Status status = TNN_OK;
    // Init cpu and device layer
    status = Init(type, param, resource, inputs_desc, outputs_desc);
    if (status != TNN_OK) {
        EXPECT_EQ((int)status, TNN_OK);
        DeInit();
        return;
    }

    status = Reshape();
    if (status != TNN_OK) {
        EXPECT_EQ((int)status, TNN_OK);
        DeInit();
        return;
    }

    // Run forward for both cpu and device layer
    status = Forward();
    if (status != TNN_OK) {
        EXPECT_EQ((int)status, TNN_OK);
        DeInit();
        return;
    }

#ifndef TNN_UNIT_TEST_BENCHMARK
    // Compare the result for both cpu and device layer
    status = Compare();
    if (status != TNN_OK) {
        EXPECT_EQ((int)status, TNN_OK);
        DeInit();
        return;
    }
#endif

    status = DeInit();
    if (status != TNN_OK) {
        EXPECT_EQ((int)status, TNN_OK);
        return;
    }
}

Status LayerTest::Init(LayerType type, LayerParam* param, LayerResource* resource, std::vector<BlobDesc>& inputs_desc,
                       std::vector<BlobDesc>& outputs_desc) {
    param_        = param;
    Status status = TNN_OK;

    status = CreateLayers(type);
    EXPECT_EQ_OR_RETURN(status, TNN_OK);

    status = CreateInputBlobs(inputs_desc);
    EXPECT_EQ_OR_RETURN(status, TNN_OK);

    status = CreateOutputBlobs(outputs_desc);
    EXPECT_EQ_OR_RETURN(status, TNN_OK);

    status = InitLayers(type, param, resource, inputs_desc, outputs_desc);
    EXPECT_EQ_OR_RETURN(status, TNN_OK);

    status = AllocateInputBlobs();
    EXPECT_EQ_OR_RETURN(status, TNN_OK);

    status = InitInputBlobsDataRandom();
    EXPECT_EQ_OR_RETURN(status, TNN_OK);

    status = AllocateOutputBlobs();
    EXPECT_EQ_OR_RETURN(status, TNN_OK);

    return status;
}

Status LayerTest::CreateLayers(LayerType type) {
    cpu_layer_ = CreateLayer(type);
    if (cpu_layer_ == NULL) {
        LOGE("Error: CreateLayer nil, type:%d\n", type);
        return Status(TNNERR_CREATE_LAYER, "Error: CreateLayer nil, type");
    }

    device_layer_ = CreateLayer(type);
    if (device_layer_ == NULL) {
        LOGE("Error: CreateLayer nil, type:%d\n", type);
        return Status(TNNERR_CREATE_LAYER, "Error: CreateLayer nil, type");
    }
    return TNN_OK;
}

// Create the blob, but not allocate memory
Status LayerTest::CreateInputBlobs(std::vector<BlobDesc>& inputs_desc) {
    for (auto blob_desc : inputs_desc) {
        BlobDesc device_blob_desc    = blob_desc;
        device_blob_desc.device_type = device_->GetDeviceType();

        Blob *cpu_input_blob, *device_input_blob;
        if (blob_desc.data_type == DATA_TYPE_INT8) {
            IntScaleResource* int8_scale = CreateIntScale(blob_desc.dims[1]);
            auto blob                    = new BlobInt8(blob_desc);
            blob->SetIntResource(int8_scale);
            cpu_input_blob = blob;

            blob = new BlobInt8(device_blob_desc);
            blob->SetIntResource(int8_scale);
            device_input_blob = blob;
        } else {
            cpu_input_blob    = new Blob(blob_desc);
            device_input_blob = new Blob(device_blob_desc);
        }
        // RUN FLOAT CPU FOR BF16 UNIT TESTS
        if (cpu_input_blob->GetBlobDesc().data_type == DATA_TYPE_BFP16)
            cpu_input_blob->GetBlobDesc().data_type = DATA_TYPE_FLOAT;
        cpu_inputs_.push_back(cpu_input_blob);
        device_inputs_.push_back(device_input_blob);
    }
    return TNN_OK;
}

/*
 * Create the output blob, but not allocate memory
 */
Status LayerTest::CreateOutputBlobs(std::vector<BlobDesc>& outputs_desc) {
    for (auto blob_desc : outputs_desc) {
        BlobDesc device_blob_desc    = blob_desc;
        device_blob_desc.device_type = device_->GetDeviceType();

        Blob *cpu_output_blob, *device_output_blob;
        if (blob_desc.data_type == DATA_TYPE_INT8) {
            IntScaleResource* int8_scale =
                CreateIntScale(blob_desc.dims.size() > 1 ? blob_desc.dims[1] : cpu_inputs_[0]->GetBlobDesc().dims[1]);
            auto blob = new BlobInt8(blob_desc);
            blob->SetIntResource(int8_scale);
            cpu_output_blob = blob;

            blob = new BlobInt8(device_blob_desc);
            blob->SetIntResource(int8_scale);
            device_output_blob = blob;
        } else {
            cpu_output_blob    = new Blob(blob_desc);
            device_output_blob = new Blob(device_blob_desc);
        }

        // RUN FLOAT CPU FOR BF16 UNIT TESTS
        if (cpu_output_blob->GetBlobDesc().data_type == DATA_TYPE_BFP16)
            cpu_output_blob->GetBlobDesc().data_type = DATA_TYPE_FLOAT;
        cpu_outputs_.push_back(cpu_output_blob);
        device_outputs_.push_back(device_output_blob);
    }
    return TNN_OK;
}

/*
 * Init both cpu layer and the device layer
 */
Status LayerTest::InitLayers(LayerType type, LayerParam* param, LayerResource* resource,
                             std::vector<BlobDesc>& inputs_desc, std::vector<BlobDesc>& outputs_desc) {
    Status status = cpu_layer_->Init(cpu_context_, param, resource, cpu_inputs_, cpu_outputs_, cpu_);
    EXPECT_EQ_OR_RETURN(status, TNN_OK);

    device_context_->SetNumThreads(std::max(1, FLAGS_th));
    status = device_layer_->Init(device_context_, param, resource, device_inputs_, device_outputs_, device_);
    EXPECT_EQ_OR_RETURN(status, TNN_OK);
    return TNN_OK;
}

/*
 * Allocate memory for Input blobs
 */
Status LayerTest::AllocateInputBlobs() {
    for (auto cpu_input_blob : cpu_inputs_) {
        Status status = BlobHandleAllocate(cpu_input_blob, cpu_);
        EXPECT_EQ_OR_RETURN(status, TNN_OK);
    }

    for (auto device_input_blob : device_inputs_) {
        Status status = BlobHandleAllocate(device_input_blob, device_);
        EXPECT_EQ_OR_RETURN(status, TNN_OK);
    }
    return TNN_OK;
}

/*
 * Init blob datas randomly
 */
Status LayerTest::InitInputBlobsDataRandom() {
    void* command_queue;
    device_context_->GetCommandQueue(&command_queue);

    for (int index = 0; index < cpu_inputs_.size(); ++index) {
        // init cpu input blob
        Blob* cpu_input_blob              = cpu_inputs_[index];
        Blob* device_input_blob           = device_inputs_[index];
        BlobDesc blob_desc                = cpu_input_blob->GetBlobDesc();
        BlobHandle cpu_input_handle       = cpu_input_blob->GetHandle();
        BlobMemorySizeInfo blob_size_info = cpu_->Calculate(blob_desc);
        int input_count                   = DimsVectorUtils::Count(blob_size_info.dims);

        MatType mat_type = NCHW_FLOAT;
        if (device_input_blob->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
            // the value is initialized as bfp16
            mat_type = RESERVED_BFP16_TEST;
        } else if (device_input_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
            // the value is initialized as int8
            mat_type = RESERVED_INT8_TEST;
        }
        TNN_NS::Mat source(DEVICE_NAIVE, mat_type, blob_desc.dims);
        void* input_data = source.GetData();
        if (mat_type == NCHW_FLOAT) {
            if (ensure_input_positive_) {
                // some layers only supports positive data as input
                InitRandom(static_cast<float*>(input_data), input_count, 0.0f, 1.0f + (float)index);
            } else {
                InitRandom(static_cast<float*>(input_data), input_count, 1.0f + (float)index);
            }
        } else if (mat_type == RESERVED_INT8_TEST) {
            if (ensure_input_positive_) {
                // some layers only supports positive values as input
                InitRandom(static_cast<int8_t*>(input_data), input_count, (int8_t)0, (int8_t)8);
            } else {
                InitRandom(static_cast<int8_t*>(input_data), input_count, (int8_t)8);
            }
        } else if (mat_type == RESERVED_BFP16_TEST) {
            if (ensure_input_positive_) {
                InitRandom(static_cast<bfp16_t*>(input_data), input_count, bfp16_t(0.f), bfp16_t(1.0f + index));
            } else {
                InitRandom(static_cast<bfp16_t*>(input_data), input_count, bfp16_t(1.0f + index));
            }
        }

        // default param for the blob_converter
        MatConvertParam param;
        param.scale = std::vector<float>(blob_desc.dims[1], 1);
        param.bias  = std::vector<float>(blob_desc.dims[1], 0);

        // CONVERT TO CPU BLOB
        BlobConverter blob_converter_cpu(cpu_input_blob);
        Status ret = blob_converter_cpu.ConvertFromMat(source, param, nullptr);
        if (ret != TNN_OK) {
            LOGE("input blob_converter failed (%s)\n", ret.description().c_str());
        }

        // CONVERT TO DEVICE BLOB
        BlobConverter blob_converter(device_input_blob);
        ret = blob_converter.ConvertFromMat(source, param, command_queue);
        if (ret != TNN_OK) {
            LOGE("input blob_converter failed (%s)\n", ret.description().c_str());
        }
    }
    return TNN_OK;
}

/*
 * Allocate memory for output blobs
 */
Status LayerTest::AllocateOutputBlobs() {
    for (auto cpu_output_blob : cpu_outputs_) {
        Status status = BlobHandleAllocate(cpu_output_blob, cpu_);
        EXPECT_EQ_OR_RETURN(status, TNN_OK);
    }

    for (auto device_output_blob : device_outputs_) {
        Status status = BlobHandleAllocate(device_output_blob, device_);
        EXPECT_EQ_OR_RETURN(status, TNN_OK);
    }
    return TNN_OK;
}

/*
 * Reshape for both cpu and device layer
 */
Status LayerTest::Reshape() {
    Status status = cpu_layer_->Reshape();
    EXPECT_EQ_OR_RETURN(status, TNN_OK);

    status = device_layer_->Reshape();
    EXPECT_EQ_OR_RETURN(status, TNN_OK);

    return TNN_OK;
}

Status LayerTest::Forward() {
    Status status;
#ifndef TNN_UNIT_TEST_BENCHMARK
    status = cpu_layer_->Forward();
    EXPECT_EQ_OR_RETURN(status, TNN_OK);
#endif

#if TNN_PROFILE && defined(TNN_UNIT_TEST_BENCHMARK)
    device_context_->StartProfile();
#endif
    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    gettimeofday(&time1, &zone);
    float min = FLT_MAX, max = FLT_MIN, sum = 0.0f;
    for (int i = 0; i < FLAGS_ic; ++i) {
        gettimeofday(&time1, &zone);

        status = device_context_->OnInstanceForwardBegin();
        EXPECT_EQ_OR_RETURN(status, TNN_OK);

        status = device_layer_->Forward();
        EXPECT_EQ_OR_RETURN(status, TNN_OK);

        status = device_context_->OnInstanceForwardEnd();
        EXPECT_EQ_OR_RETURN(status, TNN_OK);

        status = device_context_->Synchronize();
        EXPECT_EQ_OR_RETURN(status, TNN_OK);

        gettimeofday(&time2, &zone);
        float delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
        min         = fmin(min, delta);
        max         = fmax(max, delta);
        sum += delta;
    }
#if TNN_PROFILE && defined(TNN_UNIT_TEST_BENCHMARK)
    auto profile_result = device_context_->FinishProfile();
    auto result_str = profile_result->GetProfilingDataInfo();
    printf("%s", result_str.c_str());
#endif

    /*
     * shows the timings of device layer.
     * Used for benchmarking.
     */
    if (FLAGS_ub) {
        printf(
            "device %s time cost: min =   %g ms  |  max =  %g ms  |  avg = %g ms |"
            "  gflops = %g G | dram thrp = %g GB/s\n",
            FLAGS_dt.c_str(), min, max, sum / (float)FLAGS_ic,
            GetCalcMflops(param_, cpu_layer_->GetInputBlobs(), cpu_layer_->GetOutputBlobs()) * FLAGS_ic / sum,
            GetCalcDramThrp(sum / (float)FLAGS_ic));
    }
    return TNN_OK;
}

/*
 * Compare the result of cpu layer and device layer.
 * The cpu layer is regarded as reference implementation.
 */
Status LayerTest::Compare() {
    int cmp_result = 0;
    void* command_queue;
    device_context_->GetCommandQueue(&command_queue);
    for (int index = 0; index < cpu_outputs_.size(); ++index) {
        /// cpu ref blob
        Blob* cpu_output_blob = cpu_outputs_[index];
        // dev blob
        Blob* device_output_blob = device_outputs_[index];
        // mat type for both
        MatType mat_type = NCHW_FLOAT;
        if (device_output_blob->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
            mat_type = RESERVED_BFP16_TEST;
        } else if (device_output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
            mat_type = RESERVED_INT8_TEST;
        }
        auto dims = cpu_output_blob->GetBlobDesc().dims;
        int count = DimsVectorUtils::Count(dims);
        // convert cpu blob to mat
        TNN_NS::Mat cpu_mat(DEVICE_NAIVE, mat_type, dims);
        BlobConverter blob_converter_cpu(cpu_output_blob);
        blob_converter_cpu.ConvertToMat(cpu_mat, MatConvertParam(), nullptr);

        // convert dev blob to cpu mat nchw
        TNN_NS::Mat dev_cpu_mat(DEVICE_NAIVE, mat_type, dims);
        BlobConverter blob_converter_dev(device_output_blob);

        Status ret = blob_converter_dev.ConvertToMat(dev_cpu_mat, MatConvertParam(), command_queue);
        if (ret != TNN_OK) {
            LOGE("output blob_converter failed (%s)\n", ret.description().c_str());
            return ret;
        }

        // compare data
        if (device_output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
            cmp_result |= CompareData(static_cast<float*>(cpu_mat.GetData()),
                                      static_cast<float*>(dev_cpu_mat.GetData()), count, 0.01);
        } else if (device_output_blob->GetBlobDesc().data_type == DATA_TYPE_HALF) {
            cmp_result |= CompareData(static_cast<float*>(cpu_mat.GetData()),
                                      static_cast<float*>(dev_cpu_mat.GetData()), count, 0.01);
        } else if (device_output_blob->GetBlobDesc().data_type == DATA_TYPE_BFP16) {
            cmp_result |= CompareData(static_cast<bfp16_t*>(cpu_mat.GetData()),
                                      static_cast<bfp16_t*>(dev_cpu_mat.GetData()), count, 0.05);
        } else if (device_output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8) {
            cmp_result |= CompareData(static_cast<int8_t*>(cpu_mat.GetData()),
                                      static_cast<int8_t*>(dev_cpu_mat.GetData()), count);
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

        if (cmp_result != 0) {
            break;
        }
    }
    EXPECT_EQ(0, cmp_result);
    return TNN_OK;
}

Status LayerTest::DeInit() {
    for (int index = 0; index < cpu_inputs_.size(); ++index) {
        Blob* cpu_input_blob    = cpu_inputs_[index];
        Blob* device_input_blob = device_inputs_[index];
        if (cpu_input_blob->GetBlobDesc().data_type == DATA_TYPE_INT8)
            delete static_cast<BlobInt8*>(cpu_input_blob)->GetIntResource();

        BlobHandleFree(cpu_input_blob, cpu_);
        BlobHandleFree(device_input_blob, device_);
        delete cpu_input_blob;
        delete device_input_blob;
    }
    cpu_inputs_.clear();
    device_inputs_.clear();

    for (int index = 0; index < cpu_outputs_.size(); ++index) {
        Blob* device_output_blob = device_outputs_[index];
        Blob* cpu_output_blob    = cpu_outputs_[index];
        if (cpu_output_blob->GetBlobDesc().data_type == DATA_TYPE_INT8)
            delete static_cast<BlobInt8*>(cpu_output_blob)->GetIntResource();
        BlobHandleFree(cpu_output_blob, cpu_);
        BlobHandleFree(device_output_blob, device_);
        delete cpu_output_blob;
        delete device_output_blob;
    }
    cpu_outputs_.clear();
    device_outputs_.clear();

    delete cpu_layer_;
    delete device_layer_;
    return TNN_OK;
}

void LayerTest::TearDownTestCase() {
    delete cpu_context_;
    delete device_context_;
}

float LayerTest::GetCalcDramThrp(float avg_time) {
    float rw_bytes_in_total = 0.f;
    for (int index = 0; index < device_inputs_.size(); ++index) {
        Blob* blob                        = device_inputs_[index];
        BlobDesc blob_desc                = blob->GetBlobDesc();
        BlobMemorySizeInfo blob_size_info = device_->Calculate(blob_desc);
        int input_count                   = DimsVectorUtils::Count(blob_size_info.dims);
        int ele_bytes                     = DataTypeUtils::GetBytesSize(blob_size_info.data_type);
        rw_bytes_in_total += 1.0f * ele_bytes * input_count;
    }
    for (int index = 0; index < device_outputs_.size(); ++index) {
        Blob* blob                        = device_outputs_[index];
        BlobDesc blob_desc                = blob->GetBlobDesc();
        BlobMemorySizeInfo blob_size_info = device_->Calculate(blob_desc);
        int input_count                   = DimsVectorUtils::Count(blob_size_info.dims);
        int ele_bytes                     = DataTypeUtils::GetBytesSize(blob_size_info.data_type);
        rw_bytes_in_total += 1.0f * ele_bytes * input_count;
    }

    return rw_bytes_in_total / 1000.f / 1000.f / avg_time;
}

}  // namespace TNN_NS
