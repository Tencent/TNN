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

#include "file_reader.h"
#include "model_checker.h"
#include "tnn/utils/split_utils.h"

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <sstream>

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

DeviceType ConvertDeviceType(std::string device_type) {
    if ("METAL" == device_type) {
        return DEVICE_METAL;
    } else if ("OPENCL" == device_type) {
        return DEVICE_OPENCL;
    } else if ("CUDA" == device_type) {
        return DEVICE_CUDA;
    } else if ("ARM" == device_type) {
        return DEVICE_ARM;
    } else {
        return DEVICE_NAIVE;
    }
}

int InitModelConfig(ModelConfig& model_config, std::string proto_file,
                    std::string model_file) {
    {
        std::ifstream proto_stream(proto_file);
        if (!proto_stream.is_open() || !proto_stream.good()) {
            printf("read proto_file failed!\n");
            return -1;
        }
        auto buffer =
            std::string((std::istreambuf_iterator<char>(proto_stream)),
                        std::istreambuf_iterator<char>());
        model_config.params.push_back(buffer);
    }

    {
        std::ifstream model_stream(model_file);
        if (!model_stream.is_open() || !model_stream.good()) {
            printf("read model_file failed!\n");
            return -1;
        }
        auto buffer =
            std::string((std::istreambuf_iterator<char>(model_stream)),
                        std::istreambuf_iterator<char>());
        model_config.params.push_back(buffer);
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

std::pair<std::string, FileFormat> GetFileInfo(std::string input_path) {
    FileFormat format = NOTSUPPORT;
    if (access(input_path.c_str(), F_OK) == 0) {
        if (GetInputType(input_path, format)) {
            printf("\tfile name: %s  type: %d\n", input_path.c_str(), format);
            return std::make_pair(input_path, format);
        }
    }
    return std::make_pair("", format);
}

void PrintConfig() {
    printf(
        "usage:\n./model_check [-h] [-p] [-m] [-d] [-i] [-o] [-f] [-n] [-s]\n"
        "\t-h, --help     \t show this message\n"
        "\t-p, --proto    \t(require) tnn proto file path\n"
        "\t-m, --model    \t(require) tnn model file path\n"
        "\t-d, --device   \t(require) the device to run to check results, ie, "
        "OPENCL, METAL, ARM, CUDA\n"
        "\t-i, --input    \t(optional) input file\n"
        "\t-o, --output   \t(optional) dump output\n"
        "\t-r, --ref      \t(optional) the reference output to compare\n"
        "\t-n, --bias     \t(optional) bias val when preprocess image "
        "input, ie, "
        "0.0,0.0,0.0 \n"
        "\t-s, --scale    \t(optional) scale val when preprocess image "
        "input, ie, "
        "1.0,1.0,1.0 \n");
}

int main(int argc, char* argv[]) {
    // Init parameters
    std::string proto_file_name;
    std::string model_file_name;

    NetworkConfig net_config;
    ModelConfig model_config;

    ModelCheckerParam model_checker_param;
    model_checker_param.input_file  = std::make_pair("", NOTSUPPORT);
    model_checker_param.input_bias  = {0, 0, 0, 0};
    model_checker_param.input_scale = {1.0f, 1.0f, 1.0f, 1.0f};
    model_checker_param.dump_output = false;
    model_checker_param.ref_file    = std::make_pair("", NOTSUPPORT);

    struct option long_options[] = {{"proto", required_argument, 0, 'p'},
                                    {"model", required_argument, 0, 'm'},
                                    {"device", required_argument, 0, 'd'},
                                    {"input", required_argument, 0, 'i'},
                                    {"output", no_argument, 0, 'o'},
                                    {"ref", required_argument, 0, 'f'},
                                    {"bias", required_argument, 0, 'n'},
                                    {"scale", required_argument, 0, 's'},
                                    {"help", no_argument, 0, 'h'},
                                    {0, 0, 0, 0}};

    const char* optstring = "p:m:d:i:of:n:s:h";

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
            case 'd':
                printf("device: %s\n", optarg);
                net_config.device_type = ConvertDeviceType(optarg);
                break;
            case 'i':
                printf("input file: %s\n", optarg);
                model_checker_param.input_file = GetFileInfo(optarg);
                break;
            case 'o':
                printf("dump output\n");
                model_checker_param.dump_output = true;
                break;
            case 'f':
                printf("reference output file: %s\n", optarg);
                model_checker_param.ref_file = GetFileInfo(optarg);
                break;
            case 'n': {
                printf("bias: %s\n", optarg);
                std::vector<std::string> array;
                SplitUtils::SplitStr(optarg, array, ",");
                model_checker_param.input_bias.clear();
                for (auto s : array) {
                    model_checker_param.input_bias.push_back(atof(s.c_str()));
                }
            } break;
            case 's': {
                printf("scale: %s\n", optarg);
                std::vector<std::string> array;
                SplitUtils::SplitStr(optarg, array, ",");
                model_checker_param.input_scale.clear();
                for (auto s : array) {
                    model_checker_param.input_scale.push_back(atof(s.c_str()));
                }
            } break;
            case 'h':
            case '?':
                PrintConfig();
                return 0;
            default:
                PrintConfig();
                break;
        }
    }

    if ("" == model_checker_param.input_file.first &&
        "" != model_checker_param.ref_file.first) {
        printf("Error: there is no input file for output reference file!\n");
        return -1;
    }

    // only for metal device
    if (net_config.device_type == DEVICE_METAL) {
        //获取当前目录绝对路径
        auto current_absolute_path = std::shared_ptr<char>(
            (char*)calloc(2048, sizeof(char)), [](char* p) { free(p); });
        if (NULL != realpath("./", current_absolute_path.get())) {
            //获取当默认metallib路径
            strcat(current_absolute_path.get(), "/tnn.metallib");
            LOGD("Metal library path:%s\n", current_absolute_path.get());
            net_config.library_path = {
                std::string(current_absolute_path.get())};
        }
    }

    int ret = InitModelConfig(model_config, proto_file_name, model_file_name);
    if (CheckResult("init model config", ret) != true)
        return -1;

    ModelChecker model_checker;
    net_config.precision = PRECISION_HIGH;
    Status status = model_checker.Init(net_config, model_config);
    if (status != TNN_OK) {
        printf("model_checker init failed!\n");
        return -1;
    }

    ret = model_checker.SetModelCheckerParams(model_checker_param);
    if (ret != 0) {
        printf("set model_checker params failed!\n");
        return -1;
    }

    status = model_checker.RunModelChecker();
    if (status != TNN_OK) {
        printf("model check failed!\n");
        return -1;
    }
    printf("model check pass!\n");

    return 0;
}
