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

#ifndef TNN_PLATFORM_ANDROID_SRC_TEST_COMMON_H_
#define TNN_PLATFORM_ANDROID_SRC_TEST_COMMON_H_

#include <stdio.h>
#include <sys/stat.h>
#include <fstream>
#include <string>
#include <vector>

#include "tnn/utils/dims_vector_utils.h"

bool CheckResult(std::string desc, int ret);

std::string ReplaceString(std::string s);

std::vector<std::string> GetFileList(std::string folder_path);

int ReadFromTxt(float*& img, std::string file_path, std::vector<int> dims, bool nchw_to_nhwc = false);

// Read input data from text files and copy to multi batch.
int ReadFromTxtToBatch(float*& img, std::string file_path, std::vector<int> dims, bool nchw_to_nhwc);

// Read input data from text files and copy to multi batch.
int ReadFromTxtToNHWCU8_Batch(unsigned char*& img, std::string file_path, std::vector<int> dims);

int LoadRgbaFromTxt(unsigned char*& data, std::string file_path, std::vector<int> dims);

int ReadFromNchwtoNhwcU8FromTxt(unsigned char*& img, std::string file_path, std::vector<int> dims);

int ReadFromBin(float*& img, std::string file_path, std::vector<int> dims);

template <class T>
void DumpDataToTxt(T* data, std::vector<int> dims, std::string output_path = "dump_data.txt") {
    printf("save to txt file! (%s)\n", output_path.c_str());
    std::ofstream f(output_path);
    int dim_size = TNN_NS::DimsVectorUtils::Count(dims);
    printf("\tdim:[%d,%d,%d,%d]  size:%d\n", dims[0], dims[1], dims[2], dims[3], dim_size);

    for (int i = 0; i < dim_size; i++) {
        f << data[i];
        f << "\n";
    }
    f.close();
}

template <class T>
void DumpDataToBin(T* data, std::vector<int> dims, std::string output_path = "dump_data.bin") {
    printf("save to bin file! (%s)\n", output_path.c_str());
    std::ofstream f(output_path, std::ios::binary);
    int dim_size = TNN_NS::DimsVectorUtils::Count(dims);
    printf("\tdim:[%d,%d,%d,%d]  size:%d\n", dims[0], dims[1], dims[2], dims[3], dim_size);

    f.write((char*)data, dim_size * sizeof(T));

    f.close();
}

int SpiltResult(float* data, std::vector<int> dims);

void DisplayStats(const std::string& name, const std::vector<float>& costs);

void ParseProtoFile(char* proto_buffer, size_t proto_buffer_length);

int GetFileSize(std::string file_path);

int ReadFromJpeg(char*& img, std::string file_path, int& size);

#endif  // end of TNN_PLATFORM_ANDROID_SRC_COMMON_H_
