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

#include "atlas_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <fstream>
#include <memory>
#include <mutex>

#include "test_common.h"
#include "tnn/core/instance.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/mat_utils.h"

using namespace TNN_NS;

#define INPUT_8UC3_ENABLE

int GetInputMat(Blob* input_blob, std::string input_file, Mat& output_mat) {
    int ret;
#ifdef INPUT_8UC3_ENABLE
    unsigned char* input_data_ptr = nullptr;
#else
    float* input_data_ptr = nullptr;
#endif

    auto input_data_format = input_blob->GetBlobDesc().data_format;
    printf("input data format: %d\n", input_data_format);
    auto input_dims = input_blob->GetBlobDesc().dims;
    std::vector<int> mat_dims;
    if (input_data_format == DATA_FORMAT_NCHW) {
        mat_dims = input_dims;
    } else if (input_data_format == DATA_FORMAT_NHWC) {
        mat_dims = {input_dims[0], input_dims[3], input_dims[1], input_dims[2]};
    } else {
        printf("invalid input data format: %d\n", input_data_format);
        return -1;
    }
    printf("Mat dims [N C H W]: [%d %d %d %d]\n", mat_dims[0], mat_dims[1], mat_dims[2], mat_dims[3]);

#ifdef INPUT_8UC3_ENABLE
    ret = ReadFromTxtToNHWCU8_Batch(input_data_ptr, input_file, mat_dims);
    // ret = ReadFromNchwtoNhwcU8FromTxt(input_data_ptr, input_file, mat_dims);
#else
    ret = ReadFromTxtToBatch(input_data_ptr, input_file, mat_dims, false);
#endif
    if (nullptr == input_data_ptr)
        return -1;

    int index = 10;
    printf("input_data_ptr[%d] = %f\n", index, (float)input_data_ptr[index]);

#ifdef INPUT_8UC3_ENABLE
    if(mat_dims[1] == 1) {
        output_mat = Mat(DEVICE_NAIVE, NGRAY, mat_dims);
    } else {
        output_mat = Mat(DEVICE_NAIVE, N8UC3, mat_dims);
    }
#else
    output_mat = Mat(DEVICE_NAIVE, NCHW_FLOAT, mat_dims);
#endif

    // copy data into mat
    int mat_byte_size = 0;
    auto tnn_ret = MatUtils::GetMatByteSize(output_mat, mat_byte_size);
    if (tnn_ret != TNN_OK)
        return -1;
    memcpy(output_mat.GetData(), input_data_ptr, mat_byte_size);
    delete input_data_ptr;

    return 0;
}

int GetInputMatWithDvpp(Blob* input_blob, std::string input_file, void* command_queue, Mat& output_mat) {
    int ret;
    Status tnn_ret                = TNN_OK;
    unsigned char* input_data_ptr = nullptr;

    auto input_data_format = input_blob->GetBlobDesc().data_format;
    printf("input data format: %d\n", input_data_format);
    auto input_dims = input_blob->GetBlobDesc().dims;
    int batch       = input_dims[0];
    input_dims[0]   = 1;
    std::vector<int> mat_dims;
    if (input_data_format == DATA_FORMAT_NCHW) {
        mat_dims = input_dims;
    } else if (input_data_format == DATA_FORMAT_NHWC) {
        mat_dims = {input_dims[0], input_dims[3], input_dims[1], input_dims[2]};
    } else {
        printf("invalid input data format: %d\n", input_data_format);
        return -1;
    }
    printf("Mat dims [N C H W]: [%d %d %d %d]\n", mat_dims[0], mat_dims[1], mat_dims[2], mat_dims[3]);

    ret = ReadFromTxtToNHWCU8_Batch(input_data_ptr, input_file, mat_dims);
    // ret = ReadFromNchwtoNhwcU8FromTxt(input_data_ptr, input_file, mat_dims);

    if (nullptr == input_data_ptr)
        return -1;

    int index = 10;
    printf("input_data_ptr[%d] = %f\n", index, (float)input_data_ptr[index]);

    Mat input_mat(DEVICE_NAIVE, N8UC3, mat_dims);

    // copy data into mat
    int mat_byte_size = 0;
    tnn_ret = MatUtils::GetMatByteSize(input_mat, mat_byte_size);
    if (tnn_ret != TNN_OK)
        return -1;
    memcpy(input_mat.GetData(), input_data_ptr, mat_byte_size);
    delete input_data_ptr;

    // copy from host to device
    Mat input_mat_device(DEVICE_ATLAS, N8UC3, mat_dims);
    tnn_ret = MatUtils::Copy(input_mat, input_mat_device, command_queue);
    if (tnn_ret != TNN_OK) {
        printf("Mat Copy falied (%s)\n", tnn_ret.description().c_str());
        return -1;
    }

    Mat mat_resized(DEVICE_ATLAS, NNV12, mat_dims, nullptr);
    ResizeParam resize_param;
    resize_param.scale_w = 1.0f;
    resize_param.scale_h = 1.0f;
    PasteParam paste_param;
    tnn_ret = MatUtils::ResizeAndPaste(input_mat_device, mat_resized, resize_param, paste_param, command_queue);
    if (TNN_OK != tnn_ret) {
        printf("resize mat failed\n");
        return -1;
    }

    std::vector<Mat> input_mat_vec;
    for (int i = 0; i < batch; ++i) {
        input_mat_vec.push_back(mat_resized);
    }

    output_mat = Mat(DEVICE_ATLAS, NNV12, {0, 0, 0, 0}, nullptr);
    tnn_ret    = MatUtils::ConcatMatWithBatch(input_mat_vec, output_mat, command_queue);
    if (TNN_OK != tnn_ret) {
        printf("Concat mat failed\n");
        return -1;
    }

    printf("output_mat dims: [%d %d %d %d]\n", output_mat.GetBatch(), output_mat.GetChannel(), output_mat.GetHeight(),
           output_mat.GetWidth());

    return 0;
}

void* RunTNN(void* param) {
    TNNParam* tnn_param = (TNNParam*)param;
    printf("thread (%d) in...\n", tnn_param->thread_id);

    struct timeval time_begin, time_end;
    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    float delta = 0;

    Status tnn_ret;

    NetworkConfig network_config;
    network_config.network_type = tnn_param->network_type;
    network_config.device_type  = tnn_param->device_type;
    network_config.device_id    = tnn_param->device_id;

    gettimeofday(&time1, NULL);
    auto instance_ = tnn_param->tnn_net->CreateInst(network_config, tnn_ret);
    if (CheckResult("create instance", tnn_ret) != true) {
        printf("error info: %s\n", tnn_ret.description().c_str());
        return nullptr;
    }
    gettimeofday(&time2, NULL);
    delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
    printf("tnn create instance time cost: %g ms\n", delta);

    // Get command queue
    void* command_queue;
    instance_->GetCommandQueue(&command_queue);

    // Reshape
    BlobMap input_blobs_temp;
    instance_->GetAllInputBlobs(input_blobs_temp);
    InputShapesMap input_shapemap;
    input_shapemap[input_blobs_temp.begin()->first]    = input_blobs_temp.begin()->second->GetBlobDesc().dims;
    input_shapemap[input_blobs_temp.begin()->first][0] = tnn_param->batch_size;
    tnn_ret                                            = instance_->Reshape(input_shapemap);
    if (tnn_ret != TNN_OK) {
        printf("instance Reshape() falied (%s)\n", tnn_ret.description().c_str());
        return nullptr;
    }

    // Get input/output blobs
    BlobMap input_blobs, output_blobs;
    instance_->GetAllInputBlobs(input_blobs);
    Blob* input = input_blobs.begin()->second;
    for (auto it = input_blobs.begin(); it != input_blobs.end(); ++it) {
        printf("input(%s) data_format: %s  data shape [ %d %d %d %d ]\n", it->first.c_str(),
               DATA_FORMAT_NHWC == it->second->GetBlobDesc().data_format ? "NHWC" : "NCHW",
               it->second->GetBlobDesc().dims[0], it->second->GetBlobDesc().dims[1], it->second->GetBlobDesc().dims[2],
               it->second->GetBlobDesc().dims[3]);
    }
    instance_->GetAllOutputBlobs(output_blobs);

    for (auto it = output_blobs.begin(); it != output_blobs.end(); ++it) {
        printf("output(%s) data_format: %s  data shape [ %d %d %d %d ]\n", it->first.c_str(),
               DATA_FORMAT_NHWC == it->second->GetBlobDesc().data_format ? "NHWC" : "NCHW",
               it->second->GetBlobDesc().dims[0], it->second->GetBlobDesc().dims[1], it->second->GetBlobDesc().dims[2],
               it->second->GetBlobDesc().dims[3]);
    }

    // get input mat
    MatConvertParam input_param;
    input_param.scale[0] = 0.00392156862745;
    input_param.scale[1] = 0.00392156862745;
    input_param.scale[2] = 0.00392156862745;
    // input_param.reverse_channel = true;
    Mat input_mat(DEVICE_NAIVE, NCHW_FLOAT, {0, 0, 0, 0}, nullptr);  // useless mat
    if (GetInputMat(input, tnn_param->input_file, input_mat) != 0) {
    // if (GetInputMatWithDvpp(input, tnn_param->input_file, command_queue, input_mat) != 0) {
        printf("get input falied\n");
        return nullptr;
    }

    // BlobConvert
    std::shared_ptr<BlobConverter> input_cvt;
    std::map<std::string, std::shared_ptr<BlobConverter>> output_cvt_map;
    input_cvt.reset(new BlobConverter(input));
    for (auto item : output_blobs) {
        output_cvt_map[item.first].reset(new BlobConverter(item.second));
    }

    tnn_ret = input_cvt->ConvertFromMat(input_mat, input_param, command_queue);
    if (tnn_ret != TNN_OK) {
        printf("ConvertFromMat falied (%s)\n", tnn_ret.description().c_str());
        return nullptr;
    }

    // Forward on atlas device.
    // Also check the running time.
    srand(102);
    std::vector<float> costs;
    const int loopcnt = 10;
    gettimeofday(&time1, &zone);
    for (int i = 0; i < loopcnt; ++i) {
        gettimeofday(&time_begin, NULL);
        tnn_ret = instance_->Forward();
        if (tnn_ret != TNN_OK) {
            printf("instance Forward falied (%s)\n", tnn_ret.description().c_str());
        }
        gettimeofday(&time_end, NULL);
        costs.push_back((time_end.tv_sec - time_begin.tv_sec) * 1000.0 +
                        (time_end.tv_usec - time_begin.tv_usec) / 1000.0);
    }
    gettimeofday(&time2, &zone);
    delta = (time2.tv_sec - time1.tv_sec) * 1000.0 + (time2.tv_usec - time1.tv_usec) / 1000.0;
    printf("time cost: %g ms\n", delta / (float)loopcnt);
    DisplayStats("", costs);

    // copy data from atlas buffer
    // then dump to files
    for (auto output : output_blobs) {
        auto output_dims                 = output.second->GetBlobDesc().dims;
        std::vector<int> output_mat_dims = output_dims;
        if (DATA_FORMAT_NHWC == output.second->GetBlobDesc().data_format) {
            output_mat_dims = {output_dims[0], output_dims[3], output_dims[1], output_dims[2]};
        }
        Mat output_mat(DEVICE_NAIVE, NCHW_FLOAT, output_mat_dims);
        MatConvertParam output_param;
        tnn_ret = output_cvt_map[output.first]->ConvertToMat(output_mat, output_param, command_queue);
        if (tnn_ret != TNN_OK) {
            printf("ConvertToMat falied (%s)\n", tnn_ret.description().c_str());
            continue;
        }

        char temp[16];
        sprintf(temp, "%d", tnn_param->thread_id);
        std::string thread_id_str = temp;
        std::string name_temp     = ReplaceString(output.second->GetBlobDesc().name);
        DumpDataToTxt((float*)output_mat.GetData(), output_mat.GetDims(),
                      "dump_" + name_temp + "thread_" + thread_id_str + ".txt");
    }

    instance_.reset();
    printf("instance reset done!\n");

    printf("thread (%d) exit\n", tnn_param->thread_id);
}
