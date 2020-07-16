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

/*
 * This is a demo for the huawei atlas devices.
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <fstream>
#include <memory>
#include <string>

#include "test_common.h"
#include "tnn/core/instance.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/dims_vector_utils.h"

using namespace TNN_NS;
TNN net_;

std::string ReplaceString(std::string s) {
    char temp[128];
    memset(temp, 0, 128);
    memcpy(temp, s.c_str(), s.length());

    for (int i = 0; i < s.length(); ++i) {
        if ('/' == temp[i] || '\\' == temp[i]) {
            temp[i] = '_';
        }
    }

    std::string ret = temp;
    return ret;
}

/* read input from text files */
int ReadFromNchwtoNhwcU8FromTxt(unsigned char*& img, std::string file_path, std::vector<int> dims) {
    printf("read from txt file! (%s)\n", file_path.c_str());
    std::ifstream f(file_path);
    int dim_size = DimsVectorUtils::Count(dims, 0);
    int chw_size = DimsVectorUtils::Count(dims, 1);
    printf("\tdim:[%d,%d,%d,%d]  size:%d\n", dims[0], dims[1], dims[2], dims[3], dim_size);

    img = (unsigned char*)malloc(dim_size);
    if (img == NULL) {
        printf("allocate memory failed!\n");
        return -1;
    }

    std::shared_ptr<unsigned char> img_org(new unsigned char[chw_size], [](unsigned char* p) { delete[] p; });

    float tmp = 0;
    for (int i = 0; i < chw_size; i++) {
        f >> tmp;
        *(img_org.get() + i) = (unsigned char)tmp;
    }

    int channel = dims[1];
    int height  = dims[2];
    int width   = dims[3];
    for (int c = 0; c < channel; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int src_idx  = c * height * width + h * width + w;
                int dst_idx  = h * width * channel + w * channel + c;
                img[dst_idx] = *(img_org.get() + src_idx);
            }
        }
    }

    int offset = chw_size;
    for (int n = 1; n < dims[0]; ++n) {
        memcpy(img + offset, img, chw_size);
        offset += chw_size;
    }

    f.close();
    return 0;
}

// Read input data from text files and copy to multi batch.
int ReadFromTxtToBatch(float*& img, std::string file_path, std::vector<int> dims, bool nchw_to_nhwc) {
    printf("read from txt file! (%s)\n", file_path.c_str());
    std::ifstream f(file_path);
    int dim_size = TNN_NS::DimsVectorUtils::Count(dims);
    printf("\tdim:[%d,%d,%d,%d]  size:%d\n", dims[0], dims[1], dims[2], dims[3], dim_size);

    img = (float*)malloc(dim_size * sizeof(float));
    if (img == NULL) {
        printf("allocate memory failed!\n");
        return -1;
    }
    printf("allocate input memory size: %d   addr: 0x%x\n", dim_size * sizeof(float), (unsigned long)img);

    int N   = dims[0];
    int C   = dims[1];
    int H   = dims[2];
    int W   = dims[3];
    int chw = C * H * W;

    if (nchw_to_nhwc) {
        // convert from nchw to nhwc
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int idx = h * W * C + w * C + c;
                    f >> img[idx];
    //                img[idx] = img[idx] / 255.0;
                }
            }
        }
    } else {
        for (int i = 0; i < chw; i++) {
            f >> img[i];
    //        img[i] = img[i] / 255.0;
        }
    }

    int offset = chw;
    for (int n = 1; n < N; ++n) {
        memcpy(img + offset, img, chw * sizeof(float));
        offset += chw;
    }

    f.close();
    return 0;
}

// Read input data from text files and copy to multi batch.
int ReadFromTxtToNHWCU8_Batch(unsigned char*& img, std::string file_path, std::vector<int> dims) {
    printf("read from txt file! (%s)\n", file_path.c_str());
    std::ifstream f(file_path);
    int dim_size = TNN_NS::DimsVectorUtils::Count(dims);
    printf("\tdim:[%d,%d,%d,%d]  size:%d\n", dims[0], dims[1], dims[2], dims[3], dim_size);

    img = (unsigned char*)malloc(dim_size);
    if (img == NULL) {
        printf("allocate memory failed!\n");
        return -1;
    }
    printf("allocate input memory size: %d   addr: 0x%x\n", dim_size, (unsigned long)img);

    int N   = dims[0];
    int C   = dims[1];
    int H   = dims[2];
    int W   = dims[3];
    int chw = C * H * W;

    for (int i = 0; i < chw; i++) {
        int temp;
        f >> temp;
        img[i] = (unsigned char)temp;
    }


    int offset = chw;
    for (int n = 1; n < N; ++n) {
        memcpy(img + offset, img, chw);
        offset += chw;
    }

    f.close();
    return 0;
}

struct TNNParam {
    std::string input_file;
    int thread_id = 0;
    TNN* tnn_net;
    Instance* tnn_instance;
};

void* RunTNN(void* param) {
    TNNParam* tnn_param = (TNNParam*)param;
    printf("thread (%d) in...\n", tnn_param->thread_id);

    struct timeval time_begin, time_end;
    struct timezone zone;
    struct timeval time1;
    struct timeval time2;
    float delta = 0;

    int ret;
    Status error;

    // Get command queue
    void* command_queue;
    tnn_param->tnn_instance->GetCommandQueue(&command_queue);

    // Reshape
    BlobMap input_blobs_temp;
    error = tnn_param->tnn_instance->GetAllInputBlobs(input_blobs_temp);
    InputShapesMap input_shapemap;
    input_shapemap[input_blobs_temp.begin()->first] = input_blobs_temp.begin()->second->GetBlobDesc().dims;
    input_shapemap[input_blobs_temp.begin()->first][0] = 1;
    error = tnn_param->tnn_instance->Reshape(input_shapemap);

    // Get input/output blobs
    BlobMap input_blobs, output_blobs;
    error       = tnn_param->tnn_instance->GetAllInputBlobs(input_blobs);
    Blob* input = input_blobs.begin()->second;
    for (auto it = input_blobs.begin(); it != input_blobs.end(); ++it) {
        printf("input(%s) data shape [ %d %d %d %d ]\n", it->first.c_str(), it->second->GetBlobDesc().dims[0],
               it->second->GetBlobDesc().dims[1], it->second->GetBlobDesc().dims[2], it->second->GetBlobDesc().dims[3]);
    }
    tnn_param->tnn_instance->GetAllOutputBlobs(output_blobs);

    for (auto it = output_blobs.begin(); it != output_blobs.end(); ++it) {
        printf("output(%s) data shape [ %d %d %d %d ]\n", it->first.c_str(), it->second->GetBlobDesc().dims[0],
               it->second->GetBlobDesc().dims[1], it->second->GetBlobDesc().dims[2], it->second->GetBlobDesc().dims[3]);
    }

    // load input
    //float* input_data_ptr = nullptr;
    unsigned char* input_data_ptr = nullptr;
    auto input_dims = input->GetBlobDesc().dims;
    auto input_format = input->GetBlobDesc().data_format;
    if (DATA_FORMAT_NCHW == input_format) {
        //ret = ReadFromTxtToBatch(input_data_ptr, tnn_param->input_file, input_dims, false);
        ret = ReadFromTxtToNHWCU8_Batch(input_data_ptr, tnn_param->input_file, input_dims);
        //ret = ReadFromNchwtoNhwcU8FromTxt(input_data_ptr, tnn_param->input_file, input_dims);
    } else if (DATA_FORMAT_NHWC == input_format) {
        //ret = ReadFromTxtToBatch(input_data_ptr, tnn_param->input_file, {input_dims[0], input_dims[3], input_dims[1], input_dims[2]}, false);
        ret = ReadFromTxtToNHWCU8_Batch(input_data_ptr, tnn_param->input_file, input_dims);
        //ret = ReadFromNchwtoNhwcU8FromTxt(input_data_ptr, tnn_param->input_file, {input_dims[0], input_dims[3], input_dims[1], input_dims[2]});
    } else {
        printf("invalid model input format\n");
        return nullptr;
    }
    if (CheckResult("load input data", ret) != true)
        return nullptr;
    int index = 10;
    printf("input_data_ptr[%d] = %f\n", index, input_data_ptr[index]);

    // BlobConvert
    std::shared_ptr<BlobConverter> input_cvt;
    std::map<std::string, std::shared_ptr<BlobConverter>> output_cvt_map;
    input_cvt.reset(new BlobConverter(input));
    for (auto item : output_blobs) {
        output_cvt_map[item.first].reset(new BlobConverter(item.second));
    }

    Status tnn_ret;
    // copy input data into atlas
    //Mat input_mat(DEVICE_NAIVE, NCHW_FLOAT, input->GetBlobDesc().dims, input_data_ptr);
    Mat input_mat(DEVICE_NAIVE, N8UC3, input->GetBlobDesc().dims, input_data_ptr);
    MatConvertParam input_param;
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
        tnn_ret = tnn_param->tnn_instance->Forward();
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
        Mat output_mat(DEVICE_NAIVE, NCHW_FLOAT, output.second->GetBlobDesc().dims);
        MatConvertParam output_param;
        tnn_ret = output_cvt_map[output.first]->ConvertToMat(output_mat, output_param, command_queue);
        if (tnn_ret != TNN_OK) {
            printf("ConvertToMat falied (%s)\n", tnn_ret.description().c_str());
            continue;
        }

        char temp[16];
        sprintf(temp, "%d", tnn_param->thread_id);
        std::string thread_id_str = temp;
        std::string name_temp = ReplaceString(output.second->GetBlobDesc().name);
        DumpDataToTxt((float*)output_mat.GetData(), output_mat.GetDims(),
                      "dump_" + name_temp + "thread_" + thread_id_str + ".txt");
    }

    if (input_data_ptr != nullptr)
        free(input_data_ptr);


    printf("thread (%d) exit\n", tnn_param->thread_id);
}

int main(int argc, char* argv[]) {
    printf("Run Atlas test ...\n");
    if (argc == 1) {
        printf("./AtlasTest <config_filename> <input_filename>\n");
        return 0;
    } else {
        if (argc < 3) {
            printf("invalid args\n");
            return 0;
        }
        for (int i = 1; i < argc; i++) {
            printf("arg%d: %s\n", i - 1, argv[i]);
        }
    }

    Status error;
    int ret;
    ModelConfig config;
    config.model_type = MODEL_TYPE_ATLAS;
    config.params.push_back(argv[1]);

    error = net_.Init(config);  // init the net
    if (TNN_OK != error) {
        printf("TNN init failed\n");
        return -1;
    }

    std::shared_ptr<Instance> instance[4];

    for (int i = 0; i < 4; ++i) {
        NetworkConfig network_config;
        network_config.network_type = NETWORK_TYPE_ATLAS;
        network_config.device_type  = DEVICE_ATLAS;
        network_config.device_id    = i;

        instance[i] = net_.CreateInst(network_config, error);
        if (CheckResult("create instance", error) != true) {
            printf("error info: %s\n", error.description().c_str());
            return -1;
        }
    }

    TNNParam run_param[4];
    for (int i = 0; i < 4; ++i) {
        run_param[i].input_file = argv[2];
        run_param[i].thread_id = i;
        run_param[i].tnn_net = &net_;
        run_param[i].tnn_instance = instance[i].get();
    }

    RunTNN(&run_param[0]);
    RunTNN(&run_param[1]);
    RunTNN(&run_param[2]);
    RunTNN(&run_param[3]);

    for (int i = 0; i < 4; ++i) {
        instance[i].reset();
    }

    net_.DeInit();
    return 0;
}
