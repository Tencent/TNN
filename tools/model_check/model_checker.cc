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

#include "model_checker.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>

#include "file_reader.h"
#include "tnn/core/macro.h"
#include "tnn/interpreter/tnn/model_packer.h"
#include "tnn/interpreter/tnn/objseri.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

ModelChecker::ModelChecker() {
    model_checker_params_.input_file  = std::make_pair("", NOTSUPPORT);
    model_checker_params_.input_bias  = {0, 0, 0, 0};
    model_checker_params_.input_scale = {1.0f, 1.0f, 1.0f, 1.0f};
    model_checker_params_.dump_output = false;
    output_ref_data_map_.clear();
    cpu_blobdata_map.clear();
    check_results.clear();
}

ModelChecker::~ModelChecker() {
    instance_device_.reset();
    instance_cpu_.reset();
    tnn_.reset();
}

Status ModelChecker::Init(NetworkConfig& net_config, ModelConfig& model_config, InputShapesMap inputs_shape) {
    tnn_.reset(new TNN());
    Status status = tnn_->Init(model_config);
    if (status != TNN_OK) {
        LOGE("tnn init falied: %s!\n", status.description().c_str());
        return Status(TNNERR_NET_ERR, "tnn init falied");
    }

    NetworkConfig net_config_cpu;
    net_config_cpu.device_type = DEVICE_NAIVE;
    instance_cpu_              = tnn_->CreateInst(net_config_cpu, status);
    if (status != TNN_OK) {
        LOGE("create cpu instance falied: %s\n", status.description().c_str());
        return Status(TNNERR_INST_ERR, "create cpu instance falied");
    }

    instance_device_ = tnn_->CreateInst(net_config, status);
    if (status != TNN_OK) {
        LOGE("create device instance falied: %s\n", status.description().c_str());
        return Status(TNNERR_INST_ERR, "create device instance falied");
    }

    return TNN_OK;
}

int ModelChecker::SetModelCheckerParams(ModelCheckerParam params) {
    model_checker_params_ = params;
    return 0;
}

Status ModelChecker::RunModelChecker() {
    // feed instance input
    Status ret = FeedInputData();
    if (ret != TNN_OK) {
        return Status(TNNERR_COMMON_ERROR, "feed input data failed");
    }

    // get ref output data
    ret = GetOutputRefData();
    if (ret != TNN_OK) {
        return Status(TNNERR_COMMON_ERROR, "get output reference data failed");
    }

    // get cpu instance blobs data
    ret = GetCpuBlobData();
    if (ret != TNN_OK) {
        return Status(TNNERR_COMMON_ERROR, "get cpu blob data failed");
    }

    // compare between cpu and device
    ret = CompareDeviceAndCpu();
    if (ret != TNN_OK) {
        return Status(TNNERR_COMMON_ERROR, "compare device and cpu data failed");
    }

    // check result
    bool check_pass  = true;
    int failed_count = 0;
    int pass_count   = 0;
    for (auto item : check_results) {
        if (!item.second) {
            failed_count++;
            check_pass = false;
            LOGE("layer is not aligned! (layer name: %s,  layer type: %s)\n", item.first->name.c_str(),
                 item.first->type_str.c_str());
        } else {
            pass_count++;
        }
    }
    if (check_pass) {
        return TNN_OK;
    } else {
        printf("failed layer count: %d    pass layer count: %d\n", failed_count, pass_count);
        if (!check_results.back().second) {
            printf("the last layer check falied!\n");
        }
        return Status(TNNERR_COMMON_ERROR, "model check failed");
    }
}

Status ModelChecker::FeedInputData() {
    BlobMap input_blobs_cpu;
    instance_cpu_->GetAllInputBlobs(input_blobs_cpu);
    bool generate_random_input = true;

    // feed cpu instance input
    std::string input_name = model_checker_params_.input_file.first;
    if (input_blobs_cpu.size() == 1 && input_name != "") {
        FileReader file_reader;
        file_reader.SetBiasValue(model_checker_params_.input_bias);
        file_reader.SetScaleValue(model_checker_params_.input_scale);

        Blob* input_blob_cpu = input_blobs_cpu.begin()->second;
        Status status        = file_reader.Read(input_blob_cpu, input_name, model_checker_params_.input_file.second);
        if (status != TNN_OK) {
            LOGE("read input file (%s) falied!\n", input_name.c_str());
            return Status(TNNERR_COMMON_ERROR, "read input failed");
        }
        generate_random_input = false;
    }

    if (generate_random_input) {
        LOGE("Generate Random input...\n");
        for (auto item : input_blobs_cpu) {
            int data_count  = DimsVectorUtils::Count(item.second->GetBlobDesc().dims);
            float* data_ptr = reinterpret_cast<float*>(item.second->GetHandle().base);
            for (int i = 0; i < data_count; i++) {
                data_ptr[i] = (float)(rand() % 256 - 128) / 128.0f;
            }
        }
    }

    // copy cpu blob data to device blob data
    BlobMap input_blobs_device;
    instance_device_->GetAllInputBlobs(input_blobs_device);
    void* command_queue;
    instance_device_->GetCommandQueue(&command_queue);
    for (auto item : input_blobs_device) {
        MatConvertParam param;
        BlobConverter blob_converter(item.second);
        TNN_NS::Mat cpu_mat(DEVICE_NAIVE, NCHW_FLOAT, input_blobs_cpu[item.first]->GetHandle().base);
        Status ret = blob_converter.ConvertFromMat(cpu_mat, param, command_queue);
        if (ret != TNN_OK) {
            LOGE("input blob_converter failed (%s)\n", ret.description().c_str());
            return Status(TNNERR_COMMON_ERROR, "run blob_converter failed");
        }
    }

    return TNN_OK;
}

Status ModelChecker::GetOutputRefData() {
    std::string output_file_name = model_checker_params_.ref_file.first;
    if ("" != output_file_name) {
        if (TEXT == model_checker_params_.ref_file.second) {
            int num_out;
            std::ifstream f_stream(output_file_name);
            f_stream >> num_out;
            if (num_out == 0) {
                LOGE("invalid output reference file (%s)!  Please make sure the reference file formate right\n", output_file_name.c_str());
                return Status(TNNERR_COMMON_ERROR, "invalid output ref file, the wrong file formate!");
            }
            for (int index = 0; index < num_out; index++) {
                int dims_size = 0;
                int dim       = 1;
                int dim_cnt   = 1;
                std::string name;
                std::shared_ptr<float> data;
                f_stream >> name;
                f_stream >> dims_size;
                for (int i = 0; i < dims_size; i++) {
                    f_stream >> dim;
                    dim_cnt *= dim;
                }
                data.reset(new float[dim_cnt], [](float* p) { delete[] p; });
                float* data_ptr = data.get();

                for (int line = 0; line < dim_cnt; line++) {
                    f_stream >> data_ptr[line];
                }
                output_ref_data_map_[name] = data;
            }

            f_stream.close();
        } else {
            LOGE("invalid output reference file (%s)!\n", output_file_name.c_str());
            return Status(TNNERR_COMMON_ERROR, "invalid output ref file, we only support txt format!");
        }
    }

    return TNN_OK;
}

Status ModelChecker::GetCpuBlobData() {
    BlobStatisticCallback cpu_func_after = [&](std::vector<Blob*>& blobs, LayerInfo* info) {
        for (auto blob : blobs) {
            auto blob_desc        = blob->GetBlobDesc();
            std::string blob_name = blob_desc.name;

            // convert blob
            int blob_data_bytes         = DimsVectorUtils::Count(blob_desc.dims) * sizeof(float);
            cpu_blobdata_map[blob_name] = std::shared_ptr<char>(new char[blob_data_bytes], [](char* p) { delete[] p; });

            void* command_queue;
            instance_cpu_->GetCommandQueue(&command_queue);
            MatConvertParam param;
            BlobConverter blob_converter(blob);
            TNN_NS::Mat cpu_mat(DEVICE_NAIVE, NCHW_FLOAT, cpu_blobdata_map[blob_name].get());
            Status ret = blob_converter.ConvertToMat(cpu_mat, param, command_queue);
            if (ret != TNN_OK) {
                LOGE("cpu blob (name:%s) converte failed (%s)\n", blob_name.c_str(), ret.description().c_str());
            }
        }
    };

    return instance_cpu_->ForwardWithCallback(nullptr, cpu_func_after);
}

Status ModelChecker::CompareDeviceAndCpu() {
    BlobMap output_blobs_device;
    instance_device_->GetAllOutputBlobs(output_blobs_device);

    check_results.clear();

    BlobStatisticCallback device_func_after = [&](std::vector<Blob*>& blobs, LayerInfo* info) {
        bool is_pass = true;

        for (auto blob : blobs) {
            auto blob_desc        = blob->GetBlobDesc();
            std::string blob_name = blob_desc.name;

            // convert blob
            int blob_data_bytes = DimsVectorUtils::Count(blob_desc.dims) * sizeof(float);
            std::shared_ptr<char> device_mat_data(new char[blob_data_bytes], [](char* p) { delete[] p; });

            void* command_queue;
            instance_device_->GetCommandQueue(&command_queue);
            MatConvertParam param;
            BlobConverter blob_converter(blob);
            TNN_NS::Mat device_mat(DEVICE_NAIVE, NCHW_FLOAT, device_mat_data.get());
            Status ret = blob_converter.ConvertToMat(device_mat, param, command_queue);
            if (ret != TNN_OK) {
                LOGE("device blob (name:%s) converte failed (%s)\n", blob_name.c_str(), ret.description().c_str());
            }

            // compare device data with default data
            is_pass &= CompareData(device_mat_data.get(), cpu_blobdata_map[blob_name].get(), blob_desc.dims);

            // compare data with reference file
            if (!output_ref_data_map_.empty()) {
                if (output_blobs_device.find(blob_name) != output_blobs_device.end()) {
                    if (output_ref_data_map_.find(blob_name) != output_ref_data_map_.end()) {
                        auto compare_data = output_ref_data_map_[blob_name];
                        is_pass &= CompareData(device_mat_data.get(), compare_data.get(), blob_desc.dims);
                    } else {
                        LOGE("The output layer name: %s not find in the reference file.\n", blob_name.c_str());
                        is_pass = false;
                    }
                }
            }

            if (model_checker_params_.dump_output) {
                if (output_blobs_device.find(blob_name) != output_blobs_device.end()) {
                    LOGE("dump blob (%s) data\n", blob_name.c_str());
                    DumpBlobData(cpu_blobdata_map[blob_name].get(), blob_desc.dims, "cpu_" + blob_name + ".txt");
                    DumpBlobData(device_mat_data.get(), blob_desc.dims, "device_" + blob_name + ".txt");
                }
            }
        }
        check_results.push_back(std::make_pair(info, is_pass));
    };

    return instance_device_->ForwardWithCallback(nullptr, device_func_after);
}

bool ModelChecker::CompareData(void* device_data, void* cpu_data, DimsVector blob_dims) {
    float ep           = 0.005;
    float* result_data = reinterpret_cast<float*>(device_data);
    float* ref_data    = reinterpret_cast<float*>(cpu_data);

    int data_count = DimsVectorUtils::Count(blob_dims);
    for (unsigned long long i = 0; i < data_count; i++) {
        float diff = static_cast<float>(fabs(result_data[i] - ref_data[i]));
        float sum  = static_cast<float>(fabs(result_data[i]) + fabs(ref_data[i]));
        if (fabs(diff / sum) > ep && fabs(diff) > 1e-3f) {
            LOGE("ERROR AT %llu result %.6f ref %.6f  diff/sum %f  diff %f\n", i, result_data[i], ref_data[i],
                 fabs(diff / sum), fabs(diff));
            return false;
        }
    }

    return true;
}

void ModelChecker::DumpBlobData(void* blob_data, DimsVector blob_dims, std::string output_name) {
    if (blob_dims.size() != 4) {
        LOGE("output blob dims is not equal 4, will not dump data.\n");
        return;
    }

    std::ofstream f_out(output_name.c_str());

    int batch       = blob_dims[0];
    int channel     = blob_dims[1];
    int height      = blob_dims[2];
    int width       = blob_dims[3];
    float* data_ptr = reinterpret_cast<float*>(blob_data);
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channel; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int index = b * channel * height * width + c * height * width + h * width + w;
                    f_out << "[" << b << "," << c << "," << h << "," << w << "] " << data_ptr[index] << std::endl;
                }
            }
        }
    }

    f_out.close();
}

}  // namespace TNN_NS
