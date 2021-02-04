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

#include <getopt.h>

#include "run_tnn_model.h"

void PrintConfig() {
    printf(
        "usage:\n./model_check [-h] [-p] <tnnproto> [-m] <tnnmodel> [-u] <dump_directory_path> \n"
        "\t-h, --help            \t show this message\n"
        "\t-p, --proto           \t(require) tnn proto file path\n"
        "\t-m, --model           \t(require) tnn model file path\n"
        "\t-u, --dump_dir_path   \t(require) dump directory path\n");
}

int main(int argc, char* argv[]) {
    std::string proto_file;
    std::string model_file;
    std::string dump_dir_path;
    struct option long_options[] = {{"proto", required_argument, 0, 'p'},
                                    {"model", required_argument, 0, 'm'},
                                    {"dump_dir_path", required_argument, 0, 'u'},
                                    {"help", no_argument, 0, 'h'},
                                    {0, 0, 0, 0}};
    const char* optstring        = "p:m:u:h";

    while (1) {
        int c = getopt_long(argc, argv, optstring, long_options, nullptr);
        if (c == -1)
            break;

        switch (c) {
            case 'p':
                printf("proto: %s\n", optarg);
                proto_file = optarg;
                break;
            case 'm':
                printf("model: %s\n", optarg);
                model_file = optarg;
                break;
            case 'u':
                printf("dump directory path: %s\n", optarg);
                dump_dir_path = optarg;
                break;
            case 'h':
            case '?':
                PrintConfig();
                return -1;
            default:
                PrintConfig();
                return -1;
        }
    }

    auto tool = AlignTNNModel(proto_file, model_file, dump_dir_path);
    tool.Init();
    auto status = tool.RunAlignTNNModel();

    if (status != TNN_NS::TNN_OK) {
        printf("align model failed!\n");
        return -1;
    }
    printf("Congratulation, model aligned!\n");

    return 0;
}
