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
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "test/flags.h"
#include "test/test_utils.h"
#include "test/timer.h"
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
#include "tnn/utils/string_utils_inner.h"

int main(int argc, char* argv[]) {
    return TNN_NS::test::Run(argc, argv);
}

namespace TNN_NS {

namespace test {

    int Run(int argc, char* argv[]) {
        // parse command line params
        if (!ParseAndCheckCommandLine(argc, argv))
            return -1;
#if (DUMP_INPUT_BLOB || DUMP_OUTPUT_BLOB)
        g_tnn_dump_directory = FLAGS_op;
#endif

        // Set the cpu affinity.
        // usually, -dl 0-3 for little core, -dl 4-7 for big core
        // only works when -dl flags were set. benchmark script not set -dl flags
        SetCpuAffinity();

        ModelConfig model_config     = GetModelConfig();
        NetworkConfig network_config = GetNetworkConfig();

        InputShapesMap input_shape = GetInputShapesMap();        

        srand(102);

        TNN net;
        Status ret = net.Init(model_config);
        if (CheckResult("init tnn", ret)) {
            auto instance = net.CreateInst(network_config, ret, input_shape);
            if (!CheckResult("create instance", ret)) {
                return 0;
            }
            instance->SetCpuNumThreads(std::max(FLAGS_th, 1));

            //get blob 
            BlobMap input_blob_map;
            BlobMap output_blob_map;
            void* command_queue;
            instance->GetAllInputBlobs(input_blob_map);
            instance->GetAllOutputBlobs(output_blob_map);
            instance->GetCommandQueue(&command_queue);
                
            //create mat and converter
            MatMap input_mat_map = CreateBlobMatMap(input_blob_map, FLAGS_it);
            InitInputMatMap(input_mat_map);
            auto input_converters_map = CreateBlobConverterMap(input_blob_map);
            auto input_params_map = CreateConvertParamMap(input_mat_map);

            //mat format NCHW_FLOAT
            MatMap output_mat_map = CreateBlobMatMap(output_blob_map, 0);
            auto output_converters_map = CreateBlobConverterMap(output_blob_map);
            auto output_params_map = CreateConvertParamMap(output_mat_map);

            for (int i = 0; i < FLAGS_wc; ++i) {
                for(auto element : input_converters_map) {
                    auto name = element.first;
                    auto blob_converter = element.second;
                    blob_converter->ConvertFromMatAsync(*input_mat_map[name], input_params_map[name], command_queue);
                }
                instance->ForwardAsync(nullptr);
                 
                for(auto element : output_converters_map) {
                    auto name = element.first;
                    auto blob_converter = element.second;
                    blob_converter->ConvertToMat(*output_mat_map[name], output_params_map[name], command_queue);
                }
            }
#if TNN_PROFILE
            instance->StartProfile();
#endif
            
            std::string model_name = FLAGS_mp;
            if(FLAGS_mp.find_last_of("/") != -1) {
                model_name = FLAGS_mp.substr(FLAGS_mp.find_last_of("/") + 1); 
            }   
 
            Timer timer(model_name);

            for (int i = 0; i < FLAGS_ic; ++i) {
                timer.Start();
                for(auto element : input_converters_map) {
                    auto name = element.first;
                    auto blob_converter = element.second;
                    ret = blob_converter->ConvertFromMatAsync(*input_mat_map[name], input_params_map[name], command_queue);
                    if (!CheckResult("ConvertFromMat", ret)) {
                        return 0;
                    }
                }
                ret = instance->ForwardAsync(nullptr);
                if (!CheckResult("Forward", ret)) {
                    return 0;
                }
                for(auto element : output_converters_map) {
                    auto name = element.first;
                    auto blob_converter = element.second;
                    ret = blob_converter->ConvertToMat(*output_mat_map[name], output_params_map[name], command_queue);
                    if (!CheckResult("ConvertToMat", ret)) {
                        return 0;
                    }
                }
                timer.Stop();
            }
#if TNN_PROFILE
            instance->FinishProfile(true);
#endif
            CheckResult("Forward", ret);

            if (!FLAGS_op.empty()) {
                WriteOutput(output_mat_map);
            }
 
            timer.Print();
 
            FreeMatMapMemory(input_mat_map);
            FreeMatMapMemory(output_mat_map);
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
        printf("    -h                      \t%s \n", help_message);
        printf("    -mt \"<model type>\"    \t%s \n", model_type_message);
        printf("    -mp \"<model path>\"    \t%s \n", model_path_message);
        printf("    -dt \"<device type>\"   \t%s \n", device_type_message);
        printf("    -lp \"<library path>\"  \t%s \n", library_path_message);
        printf("    -ic \"<number>\"        \t%s \n", iterations_count_message);
        printf("    -wc \"<number>\"        \t%s \n", warm_up_count_message);
        printf("    -ip \"<path>\"          \t%s \n", input_path_message);
        printf("    -op \"<path>\"          \t%s \n", output_path_message);
        printf("    -dl \"<device list>\"   \t%s \n", device_list_message);
        printf("    -th \"<thread umber>\"  \t%s \n", cpu_thread_num_message);
        printf("    -it \"<input type>\"    \t%s \n", input_format_message);
        printf("    -pr \"<precision >\"    \t%s \n", precision_message);
        printf("    -is \"<input shape>\"   \t%s \n", input_shape_message);
        printf("    -fc \"<format for compare>\t%s \n", output_format_cmp_message);
        printf("    -nt \"<network type>\t%s \n", output_format_cmp_message);
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

    InputShapesMap GetInputShapesMap() {
        InputShapesMap input_shape;
        if(!FLAGS_is.empty()) {
            std::string input_shape_message(FLAGS_is);
            std::string delimiter = "[";
            std::vector<int> input_dim;
            std::ptrdiff_t p1 = 0, p2;
            p2 = input_shape_message.find(delimiter, p1);
            std::string input_name = input_shape_message.substr(p1, p2 -p1);
            p1 = p2 + 1;
            delimiter = ",";
            while (true) {
                p2 = input_shape_message.find(delimiter, p1);
                if (p2 != std::string::npos) {
                    input_dim.push_back(atoi(input_shape_message.substr(p1, p2 - p1).c_str()));
                    p1 = p2 + 1;
                } else {
                    input_dim.push_back(atoi(input_shape_message.substr(p1, input_shape_message.length() - 1 - p1).c_str()));
                    break;
                }
            }
            input_shape[input_name] = input_dim;
        }
        return input_shape;
    }

    ModelConfig GetModelConfig() {
        ModelConfig config;
        config.model_type = ConvertModelType(FLAGS_mt);
        if (config.model_type == MODEL_TYPE_TNN || config.model_type == MODEL_TYPE_OPENVINO ||
            config.model_type == MODEL_TYPE_NCNN) {
            std::string network_path = FLAGS_mp;
            int size                 = static_cast<int>(network_path.size());
            std::string model_path;
            
            // TNN file names: xxx.tnnproto  xxx.tnnmodel
            // NCNN file names: xxx.param xxx.bin
            if (config.model_type == MODEL_TYPE_TNN) {
                model_path = network_path.substr(0, size - 5) + "model";
            } else if (config.model_type == MODEL_TYPE_NCNN) {
                model_path = network_path.substr(0, size - 5) + "bin";
            } else {
                model_path = network_path.substr(0, size - 3) + "bin";
            }

            std::ifstream proto_stream(network_path);
            if (!proto_stream.is_open() || !proto_stream.good()) {
                printf("read proto_file failed!\n");
                return config;
            }
            auto buffer =
                    std::string((std::istreambuf_iterator<char>(proto_stream)), std::istreambuf_iterator<char>());
            config.params.push_back(buffer);

            if (config.model_type == MODEL_TYPE_TNN || config.model_type == MODEL_TYPE_NCNN) {
                std::ifstream model_stream(model_path, std::ios::binary);
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
        // Precision : AUTO for float computing.
        config.precision = ConvertPrecision(FLAGS_pr);

        // Device Type: ARM, OPENECL, ...
        config.device_type = ConvertDeviceType(FLAGS_dt);
        
        // use model type instead, may change later for same model type with
        // different network type
        config.network_type = ConvertNetworkType(FLAGS_nt);
        if (FLAGS_lp.length() > 0) {
            config.library_path = {FLAGS_lp};
        }
        //add for cache; When using Huawei NPU, 
	//it is the path to store the om i.e. config.cache_path = "/data/local/tmp/npu_test/";
        config.cache_path = "";
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

    MatMap CreateBlobMatMap(BlobMap& blob_map, int format_type) {
        MatMap mat_map;
        for (auto iter : blob_map) {
            auto name = iter.first;
            Blob* device_blob = iter.second;
            BlobDesc blob_desc = device_blob->GetBlobDesc();

            // Format Types: (0: NCHW FLOAT), (1: 8UC3), (2: 8UC1)
            DataType data_type = DATA_TYPE_INT8;
            MatType mat_type;
            if (format_type == 0) {
                data_type = DATA_TYPE_FLOAT;
                mat_type = NCHW_FLOAT;
            } else if (format_type == 1) {
                mat_type = N8UC3;
            } else if (format_type == 2) {
                mat_type = NGRAY;
            }
           
            int bytes = DimsVectorUtils::Count(blob_desc.dims) * DataTypeUtils::GetBytesSize(data_type); 
            void* mat_data = malloc(bytes);
            auto mat = std::make_shared<Mat>(DEVICE_NAIVE, mat_type, blob_desc.dims, mat_data);
            mat_map[name] = mat;  
        }
        return mat_map;
    }


    void InitInputMatMap(MatMap& mat_map) {
        for (auto iter : mat_map) {
            auto name = iter.first;
            auto mat = iter.second;
            void* mat_data = mat->GetData();       
            int data_count     = DimsVectorUtils::Count(mat->GetDims());
            auto mat_type = mat->GetMatType();
 
            if (FLAGS_ip.empty()) {
                for (int i = 0; i < data_count; i++) {
                    if (mat_type == NCHW_FLOAT) {
                        reinterpret_cast<float*>(mat_data)[i] = (float)(rand() % 256 - 128) / 128.0f;
                    } else {
                        reinterpret_cast<uint8_t*>(mat_data)[i] = (rand() % 256);
                    }
                }
            } else {
                LOGD("input path: %s\n", FLAGS_ip.c_str());
                std::ifstream input_stream(FLAGS_ip);
                for (int i = 0; i < data_count; i++) {
                    if (mat_type == NCHW_FLOAT) {
                        input_stream >> reinterpret_cast<float*>(mat_data)[i];
                    } else {
                        input_stream >> reinterpret_cast<uint8_t*>(mat_data)[i];
                    }
                }
            }
        }
    }

    std::map<std::string, std::shared_ptr<BlobConverter>> CreateBlobConverterMap(BlobMap& blob_map) {
        std::map<std::string, std::shared_ptr<BlobConverter>> converter_map;
        for(auto iter : blob_map) {
            auto blob_converter = std::make_shared<BlobConverter>(iter.second);
            converter_map[iter.first] = blob_converter;
        }
        return converter_map;
    }

    std::map<std::string, MatConvertParam> CreateConvertParamMap(MatMap& mat_map) {
        std::map<std::string, MatConvertParam> param_map;
        for(auto iter : mat_map) {
            MatConvertParam param;
            auto name = iter.first;
            auto mat = iter.second;
            auto mat_type = mat->GetMatType();
            auto dims = mat->GetDims();
            if(mat_type != NCHW_FLOAT) { 
                std::fill(param.scale.begin(), param.scale.end(), 1.0f / 255.0f); 
                std::fill(param.bias.begin(), param.bias.end(), 0);
            } else if(dims[1] > 4) {
                param.scale = std::vector<float>(dims[1], 1);
                param.bias  = std::vector<float>(dims[1], 0);
            }
            param_map[name] = param;
        }
        return param_map;
    }


    void WriteOutput(MatMap& outputs) {
        std::ofstream f(FLAGS_op);
        LOGD("the output path: %s\n", FLAGS_op.c_str());
        if (!FLAGS_fc) {
            LOGD("output path: %s\n", FLAGS_op.c_str());
            f << outputs.size() << std::endl;
            for (auto output : outputs) {
                f << output.first;
                auto mat      = output.second;
                DimsVector dims = mat->GetDims();
                f << " " << dims.size();
                for (auto dim : dims) {
                    f << " " << dim;
                }
                f << std::endl;
            }
        } else {
            for (auto output : outputs) {
                LOGD("the output name: %s\n", output.first.c_str());
                auto mat        = output.second;
                DimsVector dims   = mat->GetDims();
                std::string shape = "( ";
                for (auto dim : dims) {
                    shape += ToString(dim) + " ";
                }
                shape += ")";
                LOGD("the output shape: %s\n", shape.c_str());
            }
        }
        for (auto output : outputs) {
            auto mat  = output.second;
            int data_count     = DimsVectorUtils::Count(mat->GetDims());
            float* data = reinterpret_cast<float*>(mat->GetData());
            for (int c = 0; c < data_count; ++c) {
                f << std::fixed << std::setprecision(6) << data[c] << std::endl;
            }
        }
        f.close();
    }

    void FreeMatMapMemory(MatMap& mat_map) {
        for(auto iter : mat_map) {
            free(iter.second->GetData());
        }
    }

}  // namespace test

}  // namespace TNN_NS
