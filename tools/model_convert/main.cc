// Copyright 2019 Tencent. All Rights Reserved

#include "tnn/core/common.h"
#include "tnn/core/tnn.h"

#include "model_converter.h"
#include "tnn/utils/split_utils.h"

#include <dirent.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <algorithm>
#include <fstream>
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

std::string GetFileName(std::string file_path) {
    size_t pos_s = file_path.rfind('/');
    size_t pos_e = file_path.rfind('.');
    int len            = 0;
    if (pos_s == std::string::npos) {
        pos_s = 0;
    } else {
        pos_s++;
    }

    if (pos_e == std::string::npos) {
        pos_e = file_path.length();
    }

    len = pos_e - pos_s;
    return file_path.substr(pos_s, len);
}

int InitModelConfig(ModelConfig& model_config, std::string proto_file, std::string model_file) {
    {
        std::ifstream proto_stream(proto_file);
        if (!proto_stream.is_open() || !proto_stream.good()) {
            printf("read proto_file failed!\n");
            return -1;
        }
        auto buffer = std::string((std::istreambuf_iterator<char>(proto_stream)), std::istreambuf_iterator<char>());
        model_config.params.push_back(buffer);
    }

    {
        std::ifstream model_stream(model_file);
        if (!model_stream.is_open() || !model_stream.good()) {
            printf("read model_file failed!\n");
            return -1;
        }
        auto buffer = std::string((std::istreambuf_iterator<char>(model_stream)), std::istreambuf_iterator<char>());
        model_config.params.push_back(buffer);
    }
    return 0;
}

void PrintConfig() {
    printf(
        "usage:\n./model_convert [-h] [-i] [-p] <proto_path> [-m] <model_path> [-v] <version>\n"
        "\t-h, --help        \t show this message\n"
        "\t-i, --info        \t show info of model\n"
        "\t-p, --proto       \t(require) tnn proto file name\n"
        "\t-m, --model       \t(require) tnn model file name\n"
        "\t-v, --version      \t(optional) the model versoin to save\n"
        "\t\t0: RapidnetV1\n"
        "\t\t1: TNN\n"
        "\t\t2: RapidnetV3 (default)\n"
        "\t\t3: TNN_V2\n");
}

int main(int argc, char* argv[]) {
    // Init parameters
    std::string proto_file_name;
    std::string model_file_name;
    std::string output_name;
    rapidnetv3::ModelVersion model_version = rapidnetv3::MV_RPNV3;
    bool show_info                         = false;

    struct option long_options[] = {{"proto", required_argument, 0, 'p'},   {"model", required_argument, 0, 'm'},
                                    {"version", optional_argument, 0, 'v'}, {"help", no_argument, 0, 'h'},
                                    {"info", no_argument, 0, 'i'},          {0, 0, 0, 0}};

    const char* optstring = "p:m:v:hi";

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
            case 'v':
                printf("model version: %s\n", optarg);
                model_version = (rapidnetv3::ModelVersion)atoi(optarg);
                break;
            case 'i':
                show_info = true;
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
    model_config.model_type = MODEL_TYPE_RAPIDNET;
    int ret                 = InitModelConfig(model_config, proto_file_name, model_file_name);
    if (CheckResult("init model config", ret) != true)
        return -1;

    ModelConvertor model_converter;
    model_converter.SetModelVersion(model_version);
    Status status = model_converter.Init(model_config);
    if (status != TNN_OK) {
        printf("model_converter init falied!\n");
        return -1;
    }

    if (show_info) {
        model_converter.DumpModelInfo();
    } else {
        output_name = GetFileName(proto_file_name);
        if (rapidnetv3::MV_RPNV1 == model_version) {
            output_name += "_v1";
        } else if (rapidnetv3::MV_TNN == model_version) {
            output_name += "_tnn";
        } else if (rapidnetv3::MV_TNN_V2 == model_version) {
            output_name += "_tnn_v2";
        } else if (rapidnetv3::MV_RPNV3 == model_version) {
            output_name += "_v3";
        }

        status = model_converter.Serialize(output_name + ".rapidproto", output_name + ".rapidmodel");
        if (status != TNN_OK) {
            printf("model_converter serialize falied!\n");
            return -1;
        }
        printf("convert model success!\n");
    }

    return 0;
}
