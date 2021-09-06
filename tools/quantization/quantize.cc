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

#include "tnn/core/common.h"
#include "tnn/core/instance.h"
#include "tnn/core/tnn.h"

#include "calibration.h"
#include "calibration_common.h"
#include "file_reader.h"
#include "scale_calculator.h"
#include "tnn/utils/split_utils.h"

#include <dirent.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

using namespace TNN_NS;

bool CheckResult(std::string desc, int ret) {
    if (ret != 0) {
        printf("%s failed: ret %d or 0x%X\n", desc.c_str(), ret, ret);
        return false;
    } else {
        printf("%s success!\n", desc.c_str());
        return true;
    }
}

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

int InitModelConfig(ModelConfig& model_config, std::string proto_file, std::string model_file) {
    FILE* fp = fopen(proto_file.c_str(), "r");
    if (fp == NULL) {
        printf("invalid proto file\n");
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    rewind(fp);
    char* buffer = new char[size + 1];
    memset(buffer, 0, size + 1);
    int ret = fread(buffer, 1, size, fp);
    if (ret != size) {
        printf("read proto_file failed!\n");
        return -1;
    }
    fclose(fp);

    ParseProtoFile(buffer, size);
    std::string buffer_str(buffer);

    model_config.params.push_back(buffer_str);
    delete[] buffer;

    {
        std::ifstream model_stream(model_file);
        if (!model_stream.is_open() || !model_stream.good()) {
            printf("read model_file failed!\n");
            return -1;
        }
        std::string model_content =
            std::string((std::istreambuf_iterator<char>(model_stream)), std::istreambuf_iterator<char>());

        model_config.params.push_back(model_content);
    }

    return 0;
}

bool GetInputType(std::string name, FileFormat& format) {
    int pos = name.rfind('.');
    if (pos == std::string::npos)
        return false;

    std::string suffix = name.substr(pos);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), tolower);
    if (suffix == ".txt") {
        format = TEXT;
    } else if (suffix == ".npy") {
        format = NPY;
    } else if (suffix == ".jpg") {
        format = IMAGE;
    } else if (suffix == ".jpeg") {
        format = IMAGE;
    } else if (suffix == ".png") {
        format = IMAGE;
    } else if (suffix == ".bmp") {
        format = IMAGE;
    } else {
        return false;
    }

    return true;
}

int ImportDataSet(DataSet& dataset, std::string folder_path) {
    dataset.file_list.clear();

    DIR* dp;
    struct dirent* dirp;
    if ((dp = opendir(folder_path.c_str())) == NULL) {
        printf("Can't open %s\n", folder_path.c_str());
        return -1;
    }
    while ((dirp = readdir(dp)) != NULL) {
        if (dirp->d_type == DT_REG) {
            std::string file_name = dirp->d_name;
            FileFormat format     = NOTSUPPORT;
            if (GetInputType(file_name, format)) {
                char full_name[256 + 1];
                snprintf(full_name, 256, "%s/%s", folder_path.c_str(), dirp->d_name);
                dataset.file_list.push_back(std::make_pair(full_name, format));
                printf("\timport: %s  type: %d\n", dirp->d_name, format);
            }
        }
    }
    closedir(dp);

    if (dataset.file_list.size() == 0) {
        printf("no valid input file found!\n");
        return -1;
    }
    printf("import total %lu files\n", dataset.file_list.size());
    return 0;
}

bool CheckNumberString(std::string num_str) {
    const char* num_char = num_str.c_str();

    for (int i = 0; i < num_str.length(); ++i) {
        if (!(num_char[i] >= '0' && num_char[i] <= '9') && num_char[i] != '.' && num_char[i] != '-') {
            return false;
        }
    }
    return true;
}

bool CheckFileName(std::string name) {
    std::ofstream write_stream;
    write_stream.open(name);
    if (!write_stream || !write_stream.is_open() || !write_stream.good()) {
        write_stream.close();
        return false;
    }
    write_stream.close();
    return true;
}

void PrintConfig() {
    printf(
        "usage:\n./quantization_cmd [-h] [-p] <proto file> [-m] <model file> [-i] <input folder> [-b] <val> [-w] <val> "
        "[-n] <val> [-s] <val> [-t] <val> [-o] <output_name>\n"
        "\t-h, --help        \t show this message\n"
        "\t-p, --proto       \t(require) tnn proto file name\n"
        "\t-m, --model       \t(require) tnn model file name\n"
        "\t-i, --input_path  \t(require) the folder of input files\n"
        "\t-b, --blob_method \t(optional) the method to quantize blob\n"
        "\t\t0: MIN_MAX  (default)\n"
        "\t\t2: KL_DIVERGENCE\n"
        "\t\t3: ASY_MIN_MAX\n"
        "\t\t4: ACIQ_GAUS\n"
        "\t\t5: ACIQ_LAPLACE\n"        
        "\t-w, --weight_method\t(optional) the method to quantize weights\n"
        "\t\t0: MIN_MAX  (default)\n"
        "\t\t1: ADMM\n"
        "\t\t3: ASY_MIN_MAX\n"
        "\t-r, --reverse_channel\t(optional) reverse B and R channel when preprocess image\n"
        "\t\t0: the network uses rgb order  (default)\n"
        "\t\t1: the network uses bgr order\n"
        "\t-n, --bias         \t(optional) bias val when preprocess image "
        "input, ie, "
        "0.0,0.0,0.0 \n"
        "\t-s, --scale        \t(optional) scale val when preprocess image "
        "input, ie, "
        "1.0,1.0,1.0 \n"
        "\t\tformula: y = (x - bias) * scale\n"
        "\t-t, --merge_type\t(optional) merge blob/weights channel when quantize blob/weights\n"
        "\t\t0: per-channel mode  (default)\n"
        "\t\t1: mix mode          weight: per-channel  blob: per-tensor\n"
        "\t\t2: per-tersor mode\n"
        "\t-o, --output       \t(optional) specify the name of output\n");
}

int main(int argc, char* argv[]) {
    // Init parameters
    std::string proto_file_name;
    std::string model_file_name;
    std::string input_path;
    std::string output_name = "model";

    CalibrationParam cali_params;

    struct option long_options[] = {{"proto", required_argument, 0, 'p'},
                                    {"model", required_argument, 0, 'm'},
                                    {"input_path", required_argument, 0, 'i'},
                                    {"blob_method", required_argument, 0, 'b'},
                                    {"weight_method", required_argument, 0, 'w'},
                                    {"reverse_channel", required_argument, 0, 'r'},
                                    {"bias", required_argument, 0, 'n'},
                                    {"scale", required_argument, 0, 's'},
                                    {"merge_type", required_argument, 0, 't'},
                                    {"output", required_argument, 0, 'o'},
                                    {"help", no_argument, 0, 'h'},
                                    {0, 0, 0, 0}};

    const char* optstring = "p:m:i:b:w:r:n:s:t:o:h";

    if (argc == 1) {
        PrintConfig();
        return 0;
    }

    while (1) {
        int c = getopt_long(argc, argv, optstring, long_options, nullptr);
        if (c == -1)
            break;

        switch (c) {
            case 'p':
                printf("proto: %s\n", optarg);
                proto_file_name = optarg;
                break;
            case 'm':
                printf("model: %s\n", optarg);
                model_file_name = optarg;
                break;
            case 'i':
                printf("input path: %s\n", optarg);
                input_path = optarg;
                break;
            case 'b':
                printf("blob quantize method: %s\n", optarg);
                cali_params.blob_quantize_method = (CalibrationMethod)atoi(optarg);
                break;
            case 'w':
                printf("weight quantize method: %s\n", optarg);
                cali_params.weights_quantize_method = (CalibrationMethod)atoi(optarg);
                break;
            case 'r': {
                printf("reverse channel: %s\n", optarg);
                int reverse_channel = atoi(optarg);
                if (1 == reverse_channel) {
                    cali_params.reverse_channel = true;
                } else {
                    cali_params.reverse_channel = false;
                }
            } break;
            case 'n': {
                printf("bias: %s\n", optarg);
                std::vector<std::string> array;
                SplitUtils::SplitStr(optarg, array, ",");
                cali_params.input_bias.clear();
                for (auto s : array) {
                    if (!CheckNumberString(s)) {
                        printf("invalid bias value: %s\n", s.c_str());
                        return -1;
                    }
                    cali_params.input_bias.push_back(atof(s.c_str()));
                }
            } break;
            case 's': {
                printf("scale: %s\n", optarg);
                std::vector<std::string> array;
                SplitUtils::SplitStr(optarg, array, ",");
                cali_params.input_scale.clear();
                for (auto s : array) {
                    if (!CheckNumberString(s)) {
                        printf("invalid scale value: %s\n", s.c_str());
                        return -1;
                    }
                    cali_params.input_scale.push_back(atof(s.c_str()));
                }
            } break;
            case 't': {
                printf("merge type: %s\n", optarg);
                int merge_type = atoi(optarg);
                if (0 == merge_type) {
                    cali_params.merge_blob_channel    = false;
                    cali_params.merge_weights_channel = false;
                } else if (1 == merge_type) {
                    cali_params.merge_blob_channel    = true;
                    cali_params.merge_weights_channel = false;
                } else if (2 == merge_type) {
                    cali_params.merge_blob_channel    = true;
                    cali_params.merge_weights_channel = true;
                } else {
                    cali_params.merge_blob_channel    = false;
                    cali_params.merge_weights_channel = false;
                }
            } break;
            case 'o':
                printf("output name: %s\n", optarg);
                output_name = optarg;
                if (!CheckFileName(output_name + ".quantized.tnnproto")) {
                    printf("invaild output name!\n");
                    return 0;
                }
                break;
            case 'h':
            case '?':
                PrintConfig();
                return 0;
            default:
                PrintConfig();
                break;
        }
    }

    ModelConfig model_config;
    int ret = InitModelConfig(model_config, proto_file_name, model_file_name);
    if (CheckResult("init model config", ret) != true)
        return -1;

    NetworkConfig net_config;
    net_config.device_type = DEVICE_NAIVE;    
    DataSet dataset;
    ret = ImportDataSet(dataset, input_path);
    if (CheckResult("import data set", ret) != true)
        return -1;

    Calibration calibration;
    Status status = calibration.Init(net_config, model_config);
    if (status != TNN_OK) {
        printf("calibration init failed!\n");
        return -1;
    }

    ret = calibration.SetCalibrationParams(cali_params);
    if (ret != 0) {
        printf("set calibration params failed!\n");
        return -1;
    }

    status = calibration.RunCalibration(dataset);
    if (status != TNN_OK) {
        printf("calibration run failed!\n");
        return -1;
    }
    status = calibration.Serialize(output_name + ".quantized.tnnproto", output_name + ".quantized.tnnmodel");
    if (status != TNN_OK) {
        printf("calibration serialize failed! (%s)\n", status.description().c_str());
        return -1;
    }
    printf("quantize model success!\n");

    return 0;
}
