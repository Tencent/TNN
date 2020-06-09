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

#include "test/test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "test/flags.h"
#include "test/test_utils.h"
#include "tnn/core/common.h"
#include "tnn/core/instance.h"
#include "tnn/core/macro.h"
#include "tnn/core/tnn.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/blob_dump_utils.h"
#include "tnn/utils/blob_transfer_utils.h"
#include "tnn/utils/cpu_utils.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/omp_utils.h"
#include "tnn/utils/string_utils.h"

int main(int argc, char* argv[]) {
    return TNN_NS::test::Run(argc, argv);
}

namespace TNN_NS {

namespace test {

    using namespace std::chrono;

    int Run(int argc, char* argv[]) {
        // parse command line params
        if (!ParseAndCheckCommandLine(argc, argv))
            return -1;
#if (DUMP_INPUT_BLOB || DUMP_OUTPUT_BLOB)
        g_tnn_dump_directory = FLAGS_op;
#endif
        /*
         * Set the cpu affinity.
         * usually, -dl 0-3 for little core, -dl 4-7 for big core
         */
        SetCpuAffinity();

        ModelConfig model_config     = GetModelConfig();
        NetworkConfig network_config = GetNetworkConfig();
        TNN net;
        Status ret = net.Init(model_config);
        if (CheckResult("init tnn", ret)) {
            auto instance = net.CreateInst(network_config, ret);
            if (!CheckResult("create instance", ret)) {
                return 0;
            }

            instance->SetCpuNumThreads(std::max(FLAGS_th, 1));
            // set cpu affinity, only work in arm mode

            BlobMap input_blob_maps;
            BlobMap output_blob_maps;
            void* command_queue;
            instance->GetAllInputBlobs(input_blob_maps);
            instance->GetAllOutputBlobs(output_blob_maps);
            instance->GetCommandQueue(&command_queue);
            if (CheckResult("create instance", ret)) {
                srand(102);
                InitInput(input_blob_maps, command_queue);

                for (int i = 0; i < FLAGS_wc; ++i) {
                    ret = instance->Forward();
                }
#if TNN_PROFILE
                instance->StartProfile();
#endif
                auto start = system_clock::now();
                auto end   = system_clock::now();
                float min = FLT_MAX, max = FLT_MIN, sum = 0.0f;
                for (int i = 0; i < FLAGS_ic; ++i) {
                    start       = system_clock::now();
                    ret         = instance->Forward();
                    end         = system_clock::now();
                    float delta = duration_cast<microseconds>(end - start).count() / 1000.0f;
                    min         = static_cast<float>(fmin(min, delta));
                    max         = static_cast<float>(fmax(max, delta));
                    sum += delta;
                }
#if TNN_PROFILE
                instance->FinishProfile(true);
#endif
                CheckResult("Forward", ret);
                char min_str[16];
                snprintf(min_str, 16, "%6.3f", min);
                char max_str[16];
                snprintf(max_str, 16, "%6.3f", max);
                char avg_str[16];
                snprintf(avg_str, 16, "%6.3f", sum / (float)FLAGS_ic);
                std::string model_name = FLAGS_mp;
                if (FLAGS_mp.find_last_of("/") != -1) {
                    model_name = FLAGS_mp.substr(FLAGS_mp.find_last_of("/") + 1);
                }
                if (!FLAGS_op.empty()) {
                    WriteOutput(output_blob_maps, command_queue);
                }

                printf("%-45s time cost: min = %-8s ms  |  max = %-8s ms  |  avg = %-8s ms \n", model_name.c_str(),
                       min_str, max_str, avg_str);
            }
        }
        return 0;
    }

    bool ParseAndCheckCommandLine(int argc, char* argv[]) {
        gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
        if (FLAGS_h) {
            ShowUsage();
            return false;
        }

        if (FLAGS_ic < 1) {
            printf("Parameter -ic should be greater than zero (default 1) \n");
            ShowUsage();
            return false;
        }

        if (FLAGS_mp.empty()) {
            printf("Parameter -mp is not set \n");
            ShowUsage();
            return false;
        }

        return true;
    }

    void ShowUsage() {
        printf("    -h                      %s \n", help_message);
        printf("    -mt \"<model type>\"    %s \n", model_type_message);
        printf("    -mp \"<model path>\"    %s \n", model_path_message);
        printf("    -dt \"<device type>\"   %s \n", device_type_message);
        printf("    -lp \"<library path>\"  %s \n", library_path_message);
        printf("    -ic \"<number>\"        %s \n", iterations_count_message);
        printf("    -wc \"<number>\"        %s \n", warm_up_count_message);
        printf("    -ip \"<path>\"          %s \n", input_path_message);
        printf("    -op \"<path>\"          %s \n", output_path_message);
        printf("    -dl \"<device list>\"   %s \n", device_list_message);
        printf("    -th \"<thread umber>\"  %s \n", cpu_thread_num_message);
        printf("    -it \"<input type>\"    %s \n", input_format_message);
        printf("    -fc \"<format for compare>\"%s \n", output_format_cmp_message);
        printf("    -pr \"<precision >\"    %s \n", precision_message);
    }

    void SetCpuAffinity() {
        // set cpu affinity, only work in arm mode
        if (!FLAGS_dl.empty()) {
            auto split = [=](const std::string str, char delim, std::vector<std::string>& str_vec) {
                std::stringstream ss(str);
                std::string element;
                while (std::getline(ss, element, delim))
                    str_vec.push_back(element);
            };

            std::vector<std::string> devices;
            split(FLAGS_dl, ',', devices);
            std::vector<int> device_list;
            for (auto iter : devices) {
                device_list.push_back(atoi(iter.c_str()));
            }
            CpuUtils::SetCpuAffinity(device_list);
        }
    }

    ModelConfig GetModelConfig() {
        ModelConfig config;
        config.model_type = ConvertModelType(FLAGS_mt);
        if (config.model_type == MODEL_TYPE_TNN || config.model_type == MODEL_TYPE_OPENVINO ||
            config.model_type == MODEL_TYPE_NCNN) {
            std::string network_path = FLAGS_mp;
            int size                 = static_cast<int>(network_path.size());
            std::string model_path;
            /*
             * TNN file names:
             *  xxx.tnnproto  xxx.tnnmodel
             * NCNN file names:
             *  xxx.param xxx.bin
             */
            if (config.model_type == MODEL_TYPE_TNN) {
                model_path = network_path.substr(0, size - 5) + "model";
            } else if (config.model_type == MODEL_TYPE_NCNN) {
                model_path = network_path.substr(0, size - 5) + "bin";
            } else {
                model_path = network_path.substr(0, size - 3) + "bin";
            }

            {
                std::ifstream proto_stream(network_path);
                if (!proto_stream.is_open() || !proto_stream.good()) {
                    printf("read proto_file failed!\n");
                    return config;
                }
                auto buffer =
                    std::string((std::istreambuf_iterator<char>(proto_stream)), std::istreambuf_iterator<char>());
                config.params.push_back(buffer);
            }

            if (config.model_type == MODEL_TYPE_TNN || config.model_type == MODEL_TYPE_NCNN) {
                std::ifstream model_stream(model_path);
                if (!model_stream.is_open() || !model_stream.good()) {
                    config.params.push_back("");
                    return config;
                }
                auto model_content =
                    std::string((std::istreambuf_iterator<char>(model_stream)), std::istreambuf_iterator<char>());

                config.params.push_back(model_content);
            } else {
                config.params.push_back(model_path);
            }
        } else {
            config.params.push_back(FLAGS_mp);
        }
        return config;
    }

    NetworkConfig GetNetworkConfig() {
        NetworkConfig config;
        /*
         * Precision :
         *  HIGH for float computing.
         */
        config.precision = ConvertPrecision(FLAGS_pr);
        /*
         * Device Type:
         *  ARM
         *  OPENCL
         *  ...
         */
        config.device_type = ConvertDeviceType(FLAGS_dt);
        // use model type instead, may change later for same model type with
        // different network type
        config.network_type = ConvertNetworkType(FLAGS_mt);
        if (FLAGS_lp.length() > 0) {
            config.library_path = {FLAGS_lp};
        }
        return config;
    }

    bool CheckResult(std::string desc, Status result) {
        if (result != 0) {
            LOGE("%s failed: %s \n", desc.c_str(), result.description().c_str());
            return false;
        } else {
            LOGD("%s success! \n", desc.c_str());
            return true;
        }
    }

    void InitInput(BlobMap& inputs, void* command_queue) {
        MatConvertParam param;
        for (auto iter : inputs) {
            Blob* device_blob = iter.second;
            BlobConverter blob_converter(device_blob);
            BlobDesc blob_desc = device_blob->GetBlobDesc();
            int data_count     = DimsVectorUtils::Count(blob_desc.dims);

            DataType dtype = DATA_TYPE_INT8;
            /*
             * Input Types:
             *  0: NCHW FLOAT
             *  1: 8UC3
             *  2: 8UC1
             */
            MatType mat_type;
            if (FLAGS_it == 0) {
                dtype    = DATA_TYPE_FLOAT;
                mat_type = NCHW_FLOAT;
            } else if (FLAGS_it == 1) {
                mat_type = N8UC3;
            } else if (FLAGS_it == 2) {
                mat_type = NGRAY;
            }
            if (dtype == DATA_TYPE_INT8) {
                std::fill(param.scale.begin(), param.scale.end(), 1.0f / 255.0f);
                std::fill(param.bias.begin(), param.bias.end(), 0);
            }

            auto size_in_bytes = data_count * DataTypeUtils::GetBytesSize(dtype);
            void* img_data     = malloc(size_in_bytes);
            // GET FLOAT
            if (FLAGS_ip.empty()) {
                for (int i = 0; i < data_count; i++) {
                    if (dtype == DATA_TYPE_FLOAT) {
                        reinterpret_cast<float*>(img_data)[i] = (float)(rand() % 256 - 128) / 128.0f;
                    } else {
                        reinterpret_cast<uint8_t*>(img_data)[i] = (rand() % 256);
                    }
                }
            } else {
                LOGD("input path: %s\n", FLAGS_ip.c_str());
                std::ifstream input_stream(FLAGS_ip);
                for (int i = 0; i < data_count; i++) {
                    if (dtype == DATA_TYPE_FLOAT) {
                        input_stream >> reinterpret_cast<float*>(img_data)[i];
                    } else {
                        input_stream >> reinterpret_cast<uint8_t*>(img_data)[i];
                    }
                }
            }

            if (dtype == DATA_TYPE_FLOAT && blob_desc.dims[1] > 4) {
                param.scale = std::vector<float>(blob_desc.dims[1], 1);
                param.bias  = std::vector<float>(blob_desc.dims[1], 0);
            }
            TNN_NS::Mat img(DEVICE_NAIVE, mat_type, img_data);
            Status ret = blob_converter.ConvertFromMat(img, param, command_queue);
            if (ret != TNN_OK) {
                LOGE("input blob_converter failed (%s)\n", ret.description().c_str());
            }
            free(img_data);
        }
    }

    void WriteOutput(BlobMap& outputs, void* command_queue) {
        std::ofstream f(FLAGS_op);
        MatConvertParam param;

        if (!FLAGS_fc) {
            LOGD("output path: %s\n", FLAGS_op.c_str());
            f << outputs.size() << std::endl;
            for (auto output : outputs) {
                f << output.first;
                Blob* blob      = output.second;
                DimsVector dims = blob->GetBlobDesc().dims;
                f << " " << dims.size();
                for (auto dim : dims) {
                    f << " " << dim;
                }
                f << std::endl;
            }
        } else {
            MatConvertParam param;
            for (auto output : outputs) {
                LOGD("the output name: %s\n", output.first.c_str());
                Blob* blob        = output.second;
                DimsVector dims   = blob->GetBlobDesc().dims;
                std::string shape = "( ";
                for (auto dim : dims) {
                    shape += to_string(dim) + " ";
                }
                shape += ")";
                LOGD("the output shape: %s\n", shape.c_str());
            }
        }
        for (auto output : outputs) {
            Blob* device_blob  = output.second;
            BlobDesc blob_desc = device_blob->GetBlobDesc();
            int data_count     = DimsVectorUtils::Count(blob_desc.dims);
            auto size_in_bytes = data_count * sizeof(float);

            BlobConverter blob_converter(device_blob);
            void* img_data = malloc(size_in_bytes);
            TNN_NS::Mat img(DEVICE_NAIVE, NCHW_FLOAT, img_data);
            Status ret = blob_converter.ConvertToMat(img, param, command_queue);
            if (ret != TNN_OK) {
                LOGE("output blob_converter failed (%s)\n", ret.description().c_str());
            }

            float* data = static_cast<float*>(img_data);
            for (int c = 0; c < data_count; ++c) {
                f << std::fixed << std::setprecision(6) << data[c] << std::endl;
            }
            free(img_data);
        }
        f.close();
        LOGD("the output path: %s\n", FLAGS_op.c_str());
    }

}  // namespace test

}  // namespace TNN_NS
