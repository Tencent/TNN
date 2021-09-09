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
#include "tnn/core/macro.h"
#include "tnn/core/tnn.h"

#include "file_reader.h"
#include "model_checker.h"
#include "flags.h"
#include "tnn/utils/split_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
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
    std::transform(device_type.begin(), device_type.end(), device_type.begin(), ::toupper);
    if ("METAL" == device_type) {
        return DEVICE_METAL;
    } else if ("OPENCL" == device_type) {
        return DEVICE_OPENCL;
    } else if ("CUDA" == device_type) {
        return DEVICE_CUDA;
    } else if ("ARM" == device_type) {
        return DEVICE_ARM;
    } else if ("HUAWEI_NPU" == device_type) {
        return DEVICE_HUAWEI_NPU;
    } else if ("X86" == device_type) {
        return DEVICE_X86;
    } else {
        return DEVICE_NAIVE;
    }
}

Precision ConvertPrecision(std::string precision) {
    std::transform(precision.begin(), precision.end(), precision.begin(), ::toupper);
    if ("AUTO" == precision) {
        return PRECISION_AUTO;
    } else if ("NORMAL" == precision) {
        return PRECISION_NORMAL;
    } else if ("HIGH" == precision) {
        return PRECISION_HIGH;
    } else if ("LOW" == precision) {
        return PRECISION_LOW;
    } else {
        return PRECISION_HIGH;
    }
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
        std::ifstream model_stream(model_file, std::ios::binary);
        if (!model_stream.is_open() || !model_stream.good()) {
            printf("read model_file failed!\n");
            return -1;
        }
        auto buffer = std::string((std::istreambuf_iterator<char>(model_stream)), std::istreambuf_iterator<char>());
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
#ifdef WIN32
    if (access(input_path.c_str(), 0) == 0) {
#else
    if (access(input_path.c_str(), F_OK) == 0) {
#endif
        if (GetInputType(input_path, format)) {
            printf("\tfile name: %s  type: %d\n", input_path.c_str(), format);
            return std::make_pair(input_path, format);
        }
    }
    return std::make_pair("", format);
}

void ShowUsage() {
    printf(
        "usage:\n./model_check [-h] [-p] <tnnproto> [-m] <tnnmodel> [-d] <device> [-i] <input> [-o] [-e] [-f] "
        "<refernece> [-n] <val> [-s] <val> [-sp] <precision>\n");
    printf("\t-h, <help>     \t%s\n", help_message);
    printf("\t-p, <proto>    \t%s\n", proto_path_message);
    printf("\t-m, <model>    \t%s\n", model_path_message);
    printf("\t-d, <device>   \t%s\n", device_type_message);
    printf("\t-i, <input>    \t%s\n", input_path_message);
    printf("\t-f, <ref>      \t%s\n", output_ref_path_message);
    printf("\t-e, <end>      \t%s\n", cmp_end_message);
    printf("\t-n, <bias>     \t%s\n", bias_message);
    printf("\t-s, <scale>    \t%s\n", scale_message);
    printf("\t\tformula: y = (x - bias) * scale\n");
    printf("\t-b, <batch>    \t%s\n", check_batch_message);
    printf("\t-do, <dir path>   \t%s\n", dump_output_path_message);
    printf("\t-du, <dir path>   \t%s\n", dump_unaligned_layer_path_message);
    printf("\t-sp, <set precision>\t%s\n", set_precision_message);
}

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        ShowUsage();
        return false;
    }

    if (FLAGS_p.empty() || FLAGS_m.empty() || FLAGS_d.empty()) {
        printf("Parameter -p/-m/-d is not set \n");
        ShowUsage();
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    // parse command line params
    if (!ParseAndCheckCommandLine(argc, argv))
        return -1;

    // Init parameters
    std::string proto_file_name = FLAGS_p;
    std::string model_file_name = FLAGS_m;

    NetworkConfig net_config;
    net_config.device_type = ConvertDeviceType(FLAGS_d);
    ModelConfig model_config;

    ModelCheckerParam model_checker_param;
    model_checker_param.input_file        = std::make_pair("", NOTSUPPORT);
    model_checker_param.input_bias        = {0, 0, 0, 0};
    model_checker_param.input_scale       = {1.0f, 1.0f, 1.0f, 1.0f};
    model_checker_param.ref_file          = std::make_pair("", NOTSUPPORT);
    model_checker_param.dump_output_path  = FLAGS_do;
    model_checker_param.dump_dir_path     = FLAGS_a;
    model_checker_param.only_check_output = FLAGS_e;
    model_checker_param.check_batch       = FLAGS_b;
    model_checker_param.dump_unaligned_layer_path = FLAGS_du;

    printf("proto: %s\n", proto_file_name.c_str());
    printf("model: %s\n", model_file_name.c_str());
    printf("device: %s\n", FLAGS_d.c_str());

    if(!FLAGS_i.empty()) {
        printf("input file: %s\n", FLAGS_i.c_str());
        model_checker_param.input_file = GetFileInfo(FLAGS_i);
    }
    if(!FLAGS_f.empty()) {
        printf("reference output file: %s\n", FLAGS_f.c_str());
        model_checker_param.ref_file = GetFileInfo(FLAGS_f);
    }
    if(!FLAGS_n.empty()) {
        printf("bias: %s\n", FLAGS_n.c_str());
        std::vector<std::string> array;
        SplitUtils::SplitStr(FLAGS_n.c_str(), array, ",");
        model_checker_param.input_bias.clear();
        for (auto s : array) {
            model_checker_param.input_bias.push_back(atof(s.c_str()));
        }
    }
    if(!FLAGS_s.empty()) {
        printf("scale: %s\n", FLAGS_s.c_str());
        std::vector<std::string> array;
        SplitUtils::SplitStr(FLAGS_s.c_str(), array, ",");
        model_checker_param.input_scale.clear();
        for (auto s : array) {
            model_checker_param.input_scale.push_back(atof(s.c_str()));
        }
    }
    if(FLAGS_e) {
        printf("compare output only\n");
    }
    if(!FLAGS_do.empty()) {
        printf("dump output\n");
    }
    if(FLAGS_b) {
        printf("check result of multi batch\n");
    }

    if ("" == model_checker_param.input_file.first && "" != model_checker_param.ref_file.first) {
        printf("Error: there is no input file for output reference file!\n");
        return -1;
    }

    // for HuaweiNPU only check output
    if (net_config.device_type == DEVICE_HUAWEI_NPU) {
        model_checker_param.only_check_output = true;
        net_config.network_type               = NETWORK_TYPE_HUAWEI_NPU;
    }

    // for NAIVE only check output
    if (net_config.device_type == DEVICE_NAIVE && model_checker_param.dump_dir_path.empty()) {
        model_checker_param.only_check_output = true;
    }

#ifndef WIN32
    // only for metal device
    if (net_config.device_type == DEVICE_METAL) {
        //获取当前目录绝对路径
        auto current_absolute_path = std::shared_ptr<char>((char*)calloc(2048, sizeof(char)), [](char* p) { free(p); });
        if (NULL != realpath("./", current_absolute_path.get())) {
            //获取当默认metallib路径
            strcat(current_absolute_path.get(), "/tnn.metallib");
            LOGD("Metal library path:%s\n", current_absolute_path.get());
            net_config.library_path = {std::string(current_absolute_path.get())};
        }
    }
#endif

    int ret = InitModelConfig(model_config, proto_file_name, model_file_name);
    if (CheckResult("init model config", ret) != true)
        return -1;

    ModelChecker model_checker;
    auto status = model_checker.SetModelCheckerParams(model_checker_param);
    if (status != TNN_OK) {
        printf("set model_checker params failed! (error: %s)\n", status.description().c_str());
        return -1;
    }
    if ("" == FLAGS_sp) {
        if (model_checker_param.only_check_output) {
            net_config.precision = PRECISION_AUTO;
        } else {
            net_config.precision = PRECISION_HIGH;
        }
    } else {
        net_config.precision = ConvertPrecision(FLAGS_sp);
    }
    // NPU devices always use PRECISION_AUTO
    if (net_config.device_type == DEVICE_HUAWEI_NPU) {
        net_config.precision = PRECISION_AUTO;
    }
    printf("tnn precision %d\n", net_config.precision);
    status = model_checker.Init(net_config, model_config);
    if (status != TNN_OK) {
        printf("model_checker init failed! (error: %s)\n", status.description().c_str());
        return -1;
    }

    status = model_checker.RunModelChecker();
    if (status != TNN_OK) {
        printf("model check failed! (error: %s)\n", status.description().c_str());
        return -1;
    }
    printf("model check pass!\n");

    return 0;
}
