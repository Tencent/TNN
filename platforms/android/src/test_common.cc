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

#include "test_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <memory>
#include <dirent.h>

// Check the returned status code, print it if != 0
bool CheckResult(std::string desc, int ret) {
    if (ret != 0) {
        printf("%s failed: ret %d or 0x%X\n", desc.c_str(), ret, ret);
        return false;
    } else {
        printf("%s success!\n", desc.c_str());
        return true;
    }
}

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

std::vector<std::string> GetFileList(std::string folder_path) {
    std::vector<std::string> filenames;
    filenames.clear();

    DIR* dp;
    struct dirent* dirp;
    if ((dp = opendir(folder_path.c_str())) == NULL) {
        printf("Can't open %s\n", folder_path.c_str());
        return filenames;
    }
    while ((dirp = readdir(dp)) != NULL) {
        if (dirp->d_type == DT_REG) {
            char name[256];
            sprintf(name, "%s/%s", folder_path.c_str(), dirp->d_name);
            filenames.push_back(name);
            // printf("push %s\n", name);
        }
    }
    closedir(dp);

    printf("get %lu filenames\n", filenames.size());
    return filenames;
}

// Read input data from text files.
int ReadFromTxt(float*& img, std::string file_path, std::vector<int> dims, bool nchw_to_nhwc) {
    printf("read from txt file! (%s)\n", file_path.c_str());
    std::ifstream f(file_path);
    int dim_size = TNN_NS::DimsVectorUtils::Count(dims);
    printf("\tdim:[%d,%d,%d,%d]  size:%d\n", dims[0], dims[1], dims[2], dims[3], dim_size);

    img = (float*)malloc(dim_size * sizeof(float));
    if (img == NULL) {
        printf("allocate memory failed!\n");
        return -1;
    }

    int N = dims[0];
    int C = dims[1];
    int H = dims[2];
    int W = dims[3];

    if (nchw_to_nhwc) {
        // convert from nchw to nhwc
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int idx = n * H * W * C + h * W * C + w * C + c;
                        f >> img[idx];
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < dim_size; i++)
            f >> img[i];
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
    printf("allocate input memory size: %lu   addr: 0x%lx\n", dim_size * sizeof(float), (unsigned long)img);

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
    printf("allocate input memory size: %d   addr: 0x%lx\n", dim_size, (unsigned long)img);

    int N   = dims[0];
    int C   = dims[1];
    int H   = dims[2];
    int W   = dims[3];
    int chw = C * H * W;

    for (int i = 0; i < chw; i++) {
        int temp;
        f >> temp;
        img[i] = (unsigned char)temp;
        // img[i] = img[i] / 255.0;
    }


    int offset = chw;
    for (int n = 1; n < N; ++n) {
        memcpy(img + offset, img, chw);
        offset += chw;
    }

    f.close();
    return 0;
}

// read rgba data from txt file
int LoadRgbaFromTxt(unsigned char*& data, std::string file_path, std::vector<int> dims) {
    assert(dims[1] == 3);
    printf("read from txt file! (%s)\n", file_path.c_str());
    std::ifstream f(file_path);
    int dim_size = TNN_NS::DimsVectorUtils::Count(dims);
    printf("\tdim:[%d,%d,%d,%d]  size:%d\n", dims[0], dims[1], dims[2], dims[3], dim_size);

    std::shared_ptr<float> img(new float[dim_size], [](float* p) { delete[] p; });

    if (img == nullptr) {
        printf("allocate memory failed!\n");
        return -1;
    }

    for (int i = 0; i < dim_size; i++)
        f >> *(img.get() + i);

    f.close();

    // convert from nchw to rgba
    int height    = dims[2];
    int width     = dims[3];
    int data_size = 4 * height * width;
    data          = (unsigned char*)malloc(data_size);
    memset(data, 0, data_size);
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < 3; ++c) {
                int img_idx    = c * height * width + h * width + w;
                int data_idx   = h * width * 4 + w * 4 + c;
                data[data_idx] = (unsigned char)(*(img.get() + img_idx));
            }
        }
    }

    return 0;
}

/* read input from text files */
int ReadFromNchwtoNhwcU8FromTxt(unsigned char*& img, std::string file_path, std::vector<int> dims) {
    printf("read from txt file! (%s)\n", file_path.c_str());
    std::ifstream f(file_path);
    int dim_size = TNN_NS::DimsVectorUtils::Count(dims, 0);
    int chw_size = TNN_NS::DimsVectorUtils::Count(dims, 1);
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

// read data from binary files
int ReadFromBin(float*& img, std::string file_path, std::vector<int> dims) {
    printf("read from bin file! (%s)\n", file_path.c_str());
    std::ifstream f(file_path, std::ios::binary);
    int dim_size = TNN_NS::DimsVectorUtils::Count(dims);
    printf("\tdim:[%d,%d,%d,%d]  size:%d\n", dims[0], dims[1], dims[2], dims[3], dim_size);

    img = (float*)malloc(dim_size * sizeof(float));
    if (img == NULL) {
        printf("allocate memory failed!\n");
        return -1;
    }

    f.read((char*)img, dim_size * sizeof(float));

    f.close();
    return 0;
}

// split the rgb data and the alpha data,
// Then dump to binary files
int SpiltResult(float* data, std::vector<int> dims) {
    int batch   = dims[0];
    int channel = dims[1];
    int height  = dims[2];
    int width   = dims[3];

    /*
     * only supports single batch and 4 channel (rgba) data.
     */
    assert(batch == 1 && channel == 4);

    printf("save result to bins!\n");
    FILE* fd_rgb = fopen("result_rgb.bin", "wb+");
    if (fd_rgb == NULL) {
        printf("fopen failed!\n");
        return -1;
    }
    FILE* fd_alpha = fopen("result_alpha.bin", "wb+");
    if (fd_alpha == NULL) {
        printf("fopen failed!\n");
        return -1;
    }

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int result_index = c * width * height + h * width + w;
                fwrite(data + result_index, sizeof(float), 1, fd_rgb);
            }
        }
    }

    int c = 3;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int result_index = c * width * height + h * width + w;
            fwrite(data + result_index, sizeof(float), 1, fd_alpha);
        }
    }

    fflush(fd_rgb);
    fflush(fd_alpha);
    fclose(fd_rgb);
    fclose(fd_alpha);
    return 0;
}

// show the running time of the demo.
void DisplayStats(const std::string& name, const std::vector<float>& costs) {
    float max = 0, min = FLT_MAX, sum = 0, avg;
    for (auto v : costs) {
        max = fmax(max, v);
        min = fmin(min, v);
        sum += v;
        printf("[ ] time = %8.3fms\n", v);
    }
    avg = costs.size() > 0 ? sum / costs.size() : 0;
    printf("[ - ] %-24s    max = %8.3fms  min = %8.3fms  avg = %8.3fms\n", name.c_str(), max, avg == 0 ? 0 : min, avg);
}

// Strip the " and \n symbols in proto file.
void ParseProtoFile(char* proto_buffer, size_t proto_buffer_length) {
    // remove all the " and \n character
    size_t fill = 0;
    for (size_t i = 0; i < proto_buffer_length; ++i) {
        if (proto_buffer[i] != '\"' && proto_buffer[i] != '\n') {
            proto_buffer[fill++] = proto_buffer[i];
        }
    }
    proto_buffer[fill] = '\0';
}

int GetFileSize(std::string file_path) {
    struct stat statbuf;
    stat(file_path.c_str(), &statbuf);
    return statbuf.st_size;
}

int ReadFromJpeg(char*& img, std::string file_path, int& size) {
    size = GetFileSize(file_path);
    if (size <= 0) {
        printf("get jpg file size failed\n");
        return -1;
    }

    img = (char*)malloc(size);

    FILE* fd = fopen(file_path.c_str(), "rb");
    if (fd == NULL) {
        printf("open jpg file failed\n");
        return -1;
    }

    int ret = fread(img, 1, size, fd);
    if (ret != size) {
        printf("read jpg file failed\n");
        fclose(fd);
        return -1;
    }

    fclose(fd);

    return 0;
}
