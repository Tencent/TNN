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
    instance_device_ = nullptr;
    instance_cpu_ = nullptr;
    tnn_cpu_ = nullptr;
    tnn_device_ = nullptr;
}

Status ModelChecker::Init(NetworkConfig& net_config, ModelConfig& model_config, InputShapesMap inputs_shape) {
    // use tnn_cpu_ and tnn_device_ to avoid network optimize affect in different devices
    // tnn_cpu_ init
    tnn_cpu_.reset(new TNN());
    Status status = tnn_cpu_->Init(model_config);
    if (status != TNN_OK) {
        LOGE("tnn init falied: %s!\n", status.description().c_str());
        return Status(TNNERR_NET_ERR, "tnn init falied");
    }

    NetworkConfig net_config_cpu;
    net_config_cpu.device_type = DEVICE_NAIVE;
    if (net_config.device_type == DEVICE_NAIVE) {
        net_config_cpu = net_config;
    }
    instance_cpu_ = tnn_cpu_->CreateInst(net_config_cpu, status);
    if (status != TNN_OK) {
        LOGE("create cpu instance falied: %s\n", status.description().c_str());
        return status;
    }

    //仅仅比较naive和给定输出的情况
    if (net_config.device_type == DEVICE_NAIVE) {
        instance_device_ = instance_cpu_;
        return TNN_OK;
    }

    // tnn_device_ init
    tnn_device_ = std::make_shared<TNN>();
    status = tnn_device_->Init(model_config);
    if (status != TNN_OK) {
        tnn_device_ = nullptr;
        LOGE("tnn init falied: %s!\n", status.description().c_str());
        return Status(TNNERR_NET_ERR, "tnn init falied");
    }

    if(net_config.device_type == DEVICE_CUDA) {
        net_config.network_type = NETWORK_TYPE_TENSORRT; 
    }

    instance_device_ = tnn_device_->CreateInst(net_config, status);
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
    Status ret = TNN_OK;

    if (model_checker_params_.only_check_output) {
        LOGE("ModelChecker::RunModelChecker only check output of network\n");
        ret = RunModelCheckerOutput();
    } else {
        LOGE("ModelChecker::RunModelChecker check output of all layer\n");
        ret = RunModelCheckerPerLayer();
    }

    return ret;
}

Status ModelChecker::RunModelCheckerPerLayer() {
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

Status ModelChecker::RunModelCheckerOutput() {
    // feed instance input
    auto status = FeedInputData();
    RETURN_ON_NEQ(status, TNN_OK);

    // get ref output data
    status = GetOutputRefData();
    RETURN_ON_NEQ(status, TNN_OK);

    if (output_ref_data_map_.empty() && instance_device_ == instance_cpu_) {
        LOGE("output file must be specified with option -f when check tnn result with device = NAIVE\n");
        return Status(TNNERR_COMMON_ERROR, "output file must be specified with option -f when check tnn result with device = NAIVE");
    }

    // get ref output blob data
    if (output_ref_data_map_.empty() && instance_device_ != instance_cpu_) {
        status = instance_cpu_->Forward();
        RETURN_ON_NEQ(status, TNN_OK);

        status = GetOutputData(instance_cpu_.get(), output_ref_data_map_);
        RETURN_ON_NEQ(status, TNN_OK);
    }

    // get device output blob data
    status = instance_device_->Forward();
    RETURN_ON_NEQ(status, TNN_OK);

    std::map<std::string, std::shared_ptr<char>> device_output_map;
    status = GetOutputData(instance_device_.get(), device_output_map);
    RETURN_ON_NEQ(status, TNN_OK);

    // compare data diff and cos-distance
    bool check_pass = true;
    BlobMap cpu_output_blobs;
    instance_cpu_->GetAllOutputBlobs(cpu_output_blobs);
    BlobMap device_output_blobs;
    instance_device_->GetAllOutputBlobs(device_output_blobs);
    for (auto device_output_item : device_output_map) {
        auto blob_name = device_output_item.first;
        if (output_ref_data_map_.count(blob_name) == 0) {
            LOGE("cpu output don't have blob (name: %s)\n", blob_name.c_str());
            return Status(TNNERR_COMMON_ERROR, "cpu and device output not match");
        }
        auto cpu_blob_dims    = cpu_output_blobs[blob_name]->GetBlobDesc().dims;
        auto device_blob_dims = device_output_blobs[blob_name]->GetBlobDesc().dims;
        //check for dims count
        if(DimsVectorUtils::Count(cpu_blob_dims) != DimsVectorUtils::Count(device_blob_dims)) {
            LOGE("the output dims count of cpu and device are not equal! (blob name: %s)\n", blob_name.c_str());
            return Status(TNNERR_COMMON_ERROR, "the output dims of cpu and device are not same!");
        }

        if (!DimsVectorUtils::Equal(cpu_blob_dims, device_blob_dims)) {
            LOGI("the output dims count of cpu and device are equal, but dims are not same! (blob name: %s)\n", blob_name.c_str());
        }

        printf("\n---- blob (%s) ----\n", blob_name.c_str());
        check_pass &= CompareData(device_output_map[blob_name].get(), output_ref_data_map_[blob_name].get(),
                                  cpu_blob_dims, COSINE);

        if (model_checker_params_.dump_output) {
            LOGE("dump blob (%s) data\n", blob_name.c_str());
            DumpBlobData(output_ref_data_map_[blob_name].get(), cpu_blob_dims, "/Users/darrenyao/Projects/MLProjects/TNN-Github3/tools/convert2tnn/temp_data/cpu_" + blob_name + ".txt");
            DumpBlobData(device_output_map[blob_name].get(), device_blob_dims, "/Users/darrenyao/Projects/MLProjects/TNN-Github3/tools/convert2tnn/temp_data/device_" + blob_name + ".txt");
        }
    }
    if (check_pass) {
        return TNN_OK;
    } else {
        return Status(TNNERR_COMMON_ERROR, "model check failed");
    }
}

Status ModelChecker::FeedInputData() {
    BlobMap input_blobs_cpu;
    auto status = instance_cpu_->GetAllInputBlobs(input_blobs_cpu);
    RETURN_ON_NEQ(status, TNN_OK);

    // feed cpu instance input
    std::string input_name = model_checker_params_.input_file.first;
    if (input_name != "") {
        FileReader file_reader;
        file_reader.SetBiasValue(model_checker_params_.input_bias);
        file_reader.SetScaleValue(model_checker_params_.input_scale);
        status = file_reader.Read(input_blobs_cpu, input_name, model_checker_params_.input_file.second);
        if (status != TNN_OK) {
            LOGE("read input file (%s) falied!\n", input_name.c_str());
            return Status(TNNERR_COMMON_ERROR, "read input failed");
        }
    } else {
        LOGE("Generate Random input...\n");
        for (auto item : input_blobs_cpu) {
            int data_count  = DimsVectorUtils::Count(item.second->GetBlobDesc().dims);
            float* data_ptr = reinterpret_cast<float*>(item.second->GetHandle().base);
            for (int i = 0; i < data_count; i++) {
                data_ptr[i] = (float)(rand() % 256 - 128) / 128.0f;
            }
        }
    }

    if (instance_device_ == instance_cpu_) {
        return TNN_OK;
    }

    // copy cpu blob data to device blob data
    BlobMap input_blobs_device;
    status = instance_device_->GetAllInputBlobs(input_blobs_device);
    RETURN_ON_NEQ(status, TNN_OK);

    void* command_queue;
    status = instance_device_->GetCommandQueue(&command_queue);
    RETURN_ON_NEQ(status, TNN_OK);
    for (auto item : input_blobs_device) {
        MatConvertParam param;
        BlobConverter blob_converter(item.second);
        MatType mat_type = NCHW_FLOAT;
        if(input_blobs_cpu[item.first]->GetBlobDesc().data_type == DATA_TYPE_INT32) {
            mat_type = NC_INT32;
        }

        auto dims = input_blobs_cpu[item.first]->GetBlobDesc().dims;
        TNN_NS::Mat cpu_mat(DEVICE_NAIVE, mat_type, dims, input_blobs_cpu[item.first]->GetHandle().base);
        //NOTE: 待解决，devicve = NAIVE的时候， ConvertFromMat会 crash
        status = blob_converter.ConvertFromMat(cpu_mat, param, command_queue);
        RETURN_ON_NEQ(status, TNN_OK);
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
                LOGE("invalid output reference file (%s)!  Please make sure the reference file formate right\n",
                     output_file_name.c_str());
                return Status(TNNERR_COMMON_ERROR, "invalid output ref file, the wrong file formate!");
            }
            for (int index = 0; index < num_out; index++) {
                int dims_size = 0;
                int dim       = 1;
                int dim_cnt   = 1;
                int data_type;
                std::string name;
                std::shared_ptr<char> data;
                f_stream >> name;
                f_stream >> dims_size;
                for (int i = 0; i < dims_size; i++) {
                    f_stream >> dim;
                    dim_cnt *= dim;
                }
                f_stream >> data_type;
                data.reset(new char[sizeof(float) * dim_cnt], [](char* p) { delete[] p; });
                float* data_ptr = (float*)data.get();

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
            auto ret = GetBlobData(instance_cpu_.get(), blob, cpu_blobdata_map);
            if (ret != TNN_OK) {
                LOGE("get blob data failed (%s)\n", ret.description().c_str());
            }
        }
    };

    return instance_cpu_->ForwardWithCallback(nullptr, cpu_func_after);
}

Status ModelChecker::GetOutputData(Instance* instance, std::map<std::string, std::shared_ptr<char>>& output_map) {
    BlobMap output_blobs;
    instance->GetAllOutputBlobs(output_blobs);

    for (auto blobs_item : output_blobs) {
        auto ret = GetBlobData(instance, blobs_item.second, output_map);
        if (ret != TNN_OK) {
            LOGE("get blob data falied converte failed (%s)\n", ret.description().c_str());
            return ret;
        }
    }

    return TNN_OK;
}

Status ModelChecker::GetBlobData(Instance* instance, Blob* blob,
                                 std::map<std::string, std::shared_ptr<char>>& output_map) {
    auto blob_desc        = blob->GetBlobDesc();
    std::string blob_name = blob_desc.name;

    // convert blob
    int blob_data_bytes   = DimsVectorUtils::Count(blob_desc.dims) * sizeof(float);
    output_map[blob_name] = std::shared_ptr<char>(new char[blob_data_bytes], [](char* p) { delete[] p; });

    void* command_queue;
    instance->GetCommandQueue(&command_queue);
    MatConvertParam param;
    BlobConverter blob_converter(blob);
    TNN_NS::Mat cpu_mat(DEVICE_NAIVE, NCHW_FLOAT, blob_desc.dims, output_map[blob_name].get());
    Status ret = blob_converter.ConvertToMat(cpu_mat, param, command_queue);
    if (ret != TNN_OK) {
        LOGE("blob (name:%s) converte failed (%s)\n", blob_name.c_str(), ret.description().c_str());
        return ret;
    }
    return TNN_OK;
}

Status ModelChecker::CompareDeviceAndCpu() {
    BlobMap output_blobs_device;
    instance_device_->GetAllOutputBlobs(output_blobs_device);

    check_results.clear();

    BlobStatisticCallback device_func_after = [&](std::vector<Blob*>& blobs, LayerInfo* info) {
        bool is_pass = true;

        for (auto blob : blobs) {
            std::map<std::string, std::shared_ptr<char>> device_output_map;
            auto blob_desc        = blob->GetBlobDesc();
            std::string blob_name = blob_desc.name;
            auto ret              = GetBlobData(instance_device_.get(), blob, device_output_map);
            if (ret != TNN_OK) {
                LOGE("get blob data failed (%s)\n", ret.description().c_str());
            }
            char* output_data_ptr = device_output_map[blob_name].get();

            // compare device data with default data
            is_pass &= CompareData(output_data_ptr, cpu_blobdata_map[blob_name].get(), blob_desc.dims);

            // compare data with reference file
            if (!output_ref_data_map_.empty()) {
                if (output_blobs_device.find(blob_name) != output_blobs_device.end()) {
                    if (output_ref_data_map_.find(blob_name) != output_ref_data_map_.end()) {
                        auto compare_data = output_ref_data_map_[blob_name];
                        is_pass &= CompareData(output_data_ptr, compare_data.get(), blob_desc.dims);
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
                    DumpBlobData(output_data_ptr, blob_desc.dims, "device_" + blob_name + ".txt");
                }
            }
        }
        check_results.push_back(std::make_pair(info, is_pass));
    };

    return instance_device_->ForwardWithCallback(nullptr, device_func_after);
}

bool ModelChecker::CompareData(void* device_data, void* cpu_data, DimsVector blob_dims, CompareType type) {
    float* result_data = reinterpret_cast<float*>(device_data);
    float* ref_data    = reinterpret_cast<float*>(cpu_data);
    int data_count     = DimsVectorUtils::Count(blob_dims);

    if (DEFAULT == type) {
        float ep = 0.005;
        for (unsigned long long i = 0; i < data_count; i++) {
            float diff = static_cast<float>(fabs(result_data[i] - ref_data[i]));
            float sum  = static_cast<float>(fabs(result_data[i]) + fabs(ref_data[i]));
            if (fabs(diff / sum) > ep && fabs(diff) > 1e-3f) {
                LOGE("ERROR AT %llu result %.6f ref %.6f  diff/sum %f  diff %f\n", i, result_data[i], ref_data[i],
                     fabs(diff / sum), fabs(diff));
                return false;
            }
        }
    } else if (COSINE == type) {
        double max_diff     = 0;
        int max_diff_idx    = -1;
        double cos_distance = 0;

        double cpu_device_mul = 0;
        double cpu_sum2       = 0.000001;
        double device_sum2    = 0.000001;
        for (unsigned long long i = 0; i < data_count; i++) {
            float diff = static_cast<float>(fabs(result_data[i] - ref_data[i]));
            if (diff > max_diff) {
                max_diff     = diff;
                max_diff_idx = i;
            }
            cpu_device_mul += result_data[i] * ref_data[i];
            cpu_sum2 += ref_data[i] * ref_data[i];
            device_sum2 += result_data[i] * result_data[i];
        }
        cos_distance = cpu_device_mul / std::sqrt(cpu_sum2) / std::sqrt(device_sum2);

        printf("max diff: %lf   index: %d\n", max_diff, max_diff_idx);
        printf("cos distance: %lf\n", cos_distance);
        if (cos_distance < 0.999 || std::isnan(cos_distance) || std::isinf(cos_distance)) {
            return false;
        }
    } else {
        LOGE("unsupport compare data type\n");
    }

    return true;
}

void ModelChecker::DumpBlobData(void* blob_data, DimsVector blob_dims, std::string output_name) {
    std::ofstream f_out(output_name.c_str());

    int count = DimsVectorUtils::Count(blob_dims);
    float* data_ptr = reinterpret_cast<float*>(blob_data);
    for (int index = 0; index < count; ++index) {
        f_out << data_ptr[index] << std::endl;
    }

    f_out.close();
}

}  // namespace TNN_NS
