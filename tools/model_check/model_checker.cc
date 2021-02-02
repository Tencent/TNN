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
#include "tnn/utils/mat_converter_utils.h"

namespace TNN_NS {

ModelChecker::ModelChecker() {
    model_checker_params_.input_file  = std::make_pair("", NOTSUPPORT);
    model_checker_params_.input_bias  = {0, 0, 0, 0};
    model_checker_params_.input_scale = {1.0f, 1.0f, 1.0f, 1.0f};
    model_checker_params_.dump_output = false;
    output_ref_mat_map_.clear();
    cpu_blobdata_map.clear();
    check_results.clear();
}

ModelChecker::~ModelChecker() {
    instance_device_.reset();
    instance_cpu_.reset();
    tnn_cpu_.reset();
    tnn_device_.reset();
}

Status ModelChecker::Init(NetworkConfig& net_config, ModelConfig& model_config) {
    // tnn_ init
    tnn_cpu_.reset(new TNN());
    Status status = tnn_cpu_->Init(model_config);
    if (status != TNN_OK) {
        LOGE("tnn init falied: %s!\n", status.description().c_str());
        return Status(TNNERR_NET_ERR, "tnn init falied");
    }
    InputShapesMap input_shapes;
    //if (model_checker_params_.check_batch) {
    //    status = tnn_cpu_->GetModelInputShapesMap(input_shapes);
    //    if (status != TNN_OK) {
    //        LOGE("tnn get input shape map falied: %s!\n", status.description().c_str());
    //        return status;
    //    }
    //    status = ChangeBatchOfInputShapes(input_shapes);
    //    if (status != TNN_OK) {
    //        LOGE("change batch of input shape map falied: %s!\n", status.description().c_str());
    //        return status;
    //    }
    //}

    NetworkConfig net_config_cpu;
    net_config_cpu.device_type = DEVICE_NAIVE;
    instance_cpu_              = tnn_cpu_->CreateInst(net_config_cpu, status, input_shapes);
    if (status != TNN_OK) {
        LOGE("create cpu instance falied: %s\n", status.description().c_str());
        return Status(TNNERR_INST_ERR, "create cpu instance falied");
    }

    // just compare the output if Device is NAIVE
    if (net_config.device_type == DEVICE_NAIVE) {
        instance_device_ = instance_cpu_;
        return TNN_OK;
    }

    // create device instance
    if (net_config.device_type == DEVICE_CUDA) {
        net_config.network_type = NETWORK_TYPE_TENSORRT;
    }

    // tnn_device_ init
    tnn_device_.reset(new TNN());
    status = tnn_device_->Init(model_config);
    if (status != TNN_OK) {
        LOGE("tnn init falied: %s!\n", status.description().c_str());
        return Status(TNNERR_NET_ERR, "tnn init falied");
    }

    instance_device_ = tnn_device_->CreateInst(net_config, status, input_shapes);
    if (status != TNN_OK) {
        LOGE("create device instance falied: %s\n", status.description().c_str());
        return Status(TNNERR_INST_ERR, "create device instance falied");
    }

    return TNN_OK;
}

Status ModelChecker::SetModelCheckerParams(ModelCheckerParam params) {
    model_checker_params_ = params;
    return TNN_OK;
}

Status ModelChecker::RunModelChecker() {
    Status ret = TNN_OK;

    if (model_checker_params_.only_check_output) {
        ret = RunModelCheckerOutput();
    } else {
        ret = RunModelCheckerPerLayer();
    }

    return ret;
}

Status ModelChecker::ChangeBatchOfInputShapes(InputShapesMap& input_shapes) {
    // check validation
    if (input_shapes.size() <= 0) {
        return Status(TNNERR_INVALID_MODEL, "input shape count less then 0");
    }

    int cur_batch = input_shapes.begin()->second[0];
    for (auto shape : input_shapes) {
        if (cur_batch != shape.second[0]) {
            return Status(TNNERR_INVALID_MODEL, "the batch of each input are not equal");
        }
    }

    // change batch to 2*batch
    for (auto& shape : input_shapes) {
        shape.second[0] = cur_batch * 2;
    }

    return TNN_OK;
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
    Status status = FeedInputData();
    if (status != TNN_OK) {
        return Status(TNNERR_COMMON_ERROR, "feed input data failed");
    }

    // get ref output data
    status = GetOutputRefData();
    RETURN_ON_NEQ(status, TNN_OK);

    if (output_ref_mat_map_.empty() && instance_device_ == instance_cpu_) {
        LOGE("output file must be specified with option -f when check tnn result with device = NAIVE\n");
        return Status(TNNERR_COMMON_ERROR,
                      "output file must be specified with option -f when check tnn result with device = NAIVE");
    }

    // extend output_ref_mat_map_ if test batch
    if (!output_ref_mat_map_.empty()) {
        BlobMap output_blobs_cpu;
        auto status = instance_cpu_->GetAllOutputBlobs(output_blobs_cpu);
        RETURN_ON_NEQ(status, TNN_OK);

        status = ExtendMatMap(output_blobs_cpu, output_ref_mat_map_);
        RETURN_ON_NEQ(status, TNN_OK);
    }

    // get ref output blob data
    if (output_ref_mat_map_.empty() && instance_device_ != instance_cpu_) {
        status = instance_cpu_->Forward();
        RETURN_ON_NEQ(status, TNN_OK);

        status = GetOutputData(instance_cpu_.get(), output_ref_mat_map_);
        RETURN_ON_NEQ(status, TNN_OK);
    }

    // get device output blob data
    status = instance_device_->Forward();
    RETURN_ON_NEQ(status, TNN_OK);

    std::map<std::string, std::shared_ptr<Mat>> device_output_mat_map;
    status = GetOutputData(instance_device_.get(), device_output_mat_map);
    RETURN_ON_NEQ(status, TNN_OK);

    // compare data diff and cos-distance
    bool check_pass = true;
    for (auto device_output_item : device_output_mat_map) {
        auto blob_name = device_output_item.first;
        if (output_ref_mat_map_.count(blob_name) == 0) {
            LOGE("cpu output don't have mat (name: %s)\n", blob_name.c_str());
            return Status(TNNERR_COMMON_ERROR, "cpu and device output not match");
        }
        auto cpu_blob_dims    = output_ref_mat_map_[blob_name]->GetDims();
        auto device_blob_dims = device_output_mat_map[blob_name]->GetDims();
        // check for dims count
        if (!DimsVectorUtils::Equal(cpu_blob_dims, device_blob_dims)) {
            LOGI("the output dims of cpu and device are not same! (blob name: %s)\n", blob_name.c_str());
        }

        int batch = cpu_blob_dims[0];
        int bytesize_perbatch =
            DimsVectorUtils::Count(cpu_blob_dims, 1) * GetMatElementSize(output_ref_mat_map_[blob_name].get());
        printf("\n---- blob (%s) ----\n", blob_name.c_str());
        auto compare_dims = cpu_blob_dims;
        compare_dims[0]   = 1;
        for (int b = 0; b < batch; ++b) {
            printf("\tbatch: %d\n", b);
            int offset = b * bytesize_perbatch;
            check_pass &= CompareData((char*)device_output_mat_map[blob_name]->GetData() + offset,
                                      (char*)output_ref_mat_map_[blob_name]->GetData() + offset, compare_dims, COSINE);
        }

        if (model_checker_params_.dump_output) {
            printf("\ndump blob (%s) data\n", blob_name.c_str());
            DumpBlobData(output_ref_mat_map_[blob_name]->GetData(), cpu_blob_dims,
                         "cpu_" + blob_name + ".txt");
            DumpBlobData(device_output_mat_map[blob_name]->GetData(), device_blob_dims,
                         "device_" + blob_name + ".txt");
        }
    }
    if (check_pass) {
        return TNN_OK;
    } else {
        return Status(TNNERR_COMMON_ERROR, "model check failed");
    }
}

bool ModelChecker::IsDimsCanBeExtend(std::vector<int> src_dims, std::vector<int> dst_dims) {
    if (src_dims.size() != dst_dims.size()) {
        return false;
    }

    if (dst_dims[0] < src_dims[0]) {
        return false;
    }

    if (dst_dims[0] % src_dims[0] != 0) {
        return false;
    }

    for (int i = 1; i < dst_dims.size(); ++i) {
        if (src_dims[i] != dst_dims[i]) {
            return false;
        }
    }

    return true;
}

Status ModelChecker::ExtendMatMap(const BlobMap& blobs_map, std::map<std::string, std::shared_ptr<Mat>>& mat_map) {
    for (auto item : blobs_map) {
        auto blob_name = item.first;
        if (mat_map.count(blob_name) <= 0) {
            LOGE("mat map don't has blob data (name: %s)\n", blob_name.c_str());
            return Status(TNNERR_COMMON_ERROR, "extend falied: mat map is not match with blobs map");
        }

        auto mat      = mat_map[blob_name];
        auto src_dims = mat->GetDims();
        auto dst_dims = item.second->GetBlobDesc().dims;

        if (DimsVectorUtils::Equal(src_dims, dst_dims)) {
            continue;
        }

        printf("Warning: mat map (name: %s) will try to be extended due to dims not match\n", blob_name.c_str());
        if (!IsDimsCanBeExtend(src_dims, dst_dims)) {
            return Status(TNNERR_COMMON_ERROR, "extend falied: dims can't be extend");
        }

        int bytesize_perbatch = DimsVectorUtils::Count(src_dims, 1) * GetMatElementSize(mat.get());
        int src_batch_size    = src_dims[0];
        int dst_batch_size    = dst_dims[0];
        int src_bytesize      = bytesize_perbatch * src_batch_size;

        printf("batch extrend form %d to %d\n", src_batch_size, dst_batch_size);
        std::shared_ptr<Mat> mat_new(new Mat(mat->GetDeviceType(), mat->GetMatType(), dst_dims));
        int batch_idx = 0;
        for (; batch_idx < dst_batch_size - src_batch_size; batch_idx += src_batch_size) {
            memcpy((char*)mat_new->GetData() + batch_idx * bytesize_perbatch, mat->GetData(), src_bytesize);
        }
        int batch_left = dst_batch_size - batch_idx;
        memcpy((char*)mat_new->GetData() + batch_idx * bytesize_perbatch, mat->GetData(),
               batch_left * bytesize_perbatch);

        mat_map[blob_name] = mat_new;
    }

    return TNN_OK;
}

Status ModelChecker::FeedInputData() {
    BlobMap input_blobs_cpu;
    instance_cpu_->GetAllInputBlobs(input_blobs_cpu);
    bool generate_random_input = true;

    // get mat map
    std::map<std::string, std::shared_ptr<Mat>> input_mat_map;
    std::string input_name = model_checker_params_.input_file.first;
    if (input_name != "") {
        FileReader file_reader;
        file_reader.SetBiasValue(model_checker_params_.input_bias);
        file_reader.SetScaleValue(model_checker_params_.input_scale);
        auto status = file_reader.Read(input_mat_map, input_name, model_checker_params_.input_file.second);
        if (status != TNN_OK) {
            LOGE("read input file (%s) falied!\n", input_name.c_str());
            return Status(TNNERR_COMMON_ERROR, "read input failed");
        }

        status = ExtendMatMap(input_blobs_cpu, input_mat_map);
        RETURN_ON_NEQ(status, TNN_OK);
    } else {
        LOGE("Generate Random input...\n");
        for (auto item : input_blobs_cpu) {
            auto dims      = item.second->GetBlobDesc().dims;
            auto data_type = item.second->GetBlobDesc().data_type;
            int data_count = DimsVectorUtils::Count(dims);
            std::shared_ptr<Mat> mat;
            if (DATA_TYPE_FLOAT == data_type) {
                mat             = std::shared_ptr<Mat>(new Mat(DEVICE_NAIVE, NCHW_FLOAT, dims));
                float* data_ptr = reinterpret_cast<float*>(mat->GetData());
                for (int i = 0; i < data_count; i++) {
                    data_ptr[i] = (float)(rand() % 256 - 128) / 128.0f;
                }
            //} else if (DATA_TYPE_INT32 == data_type) {
            //    mat           = std::shared_ptr<Mat>(new Mat(DEVICE_NAIVE, NC_INT32, dims));
            //    int* data_ptr = reinterpret_cast<int*>(mat->GetData());
            //    for (int i = 0; i < data_count; i++) {
            //        data_ptr[i] = rand() % 256 - 128;
            //    }
            } else {
                return Status(TNNERR_COMMON_ERROR, "generate input data failed");
            }
            input_mat_map[item.first] = mat;
        }
    }

    // feed cpu instance input
    for (auto item : input_blobs_cpu) {
        if (input_mat_map.count(item.first) == 0) {
            LOGE("input mat map not found blob data (name: %s)\n", item.first.c_str());
            return Status(TNNERR_COMMON_ERROR, "input mat not match with blobs");
        }
        MatConvertParam param;
        auto status = instance_cpu_->SetInputMat(input_mat_map[item.first], param, item.first);
        RETURN_ON_NEQ(status, TNN_OK);
    }

    if (instance_device_ == instance_cpu_) {
        return TNN_OK;
    }

    // feed device instance input
    BlobMap input_blobs_device;
    auto status = instance_device_->GetAllInputBlobs(input_blobs_device);
    RETURN_ON_NEQ(status, TNN_OK);

    for (auto item : input_blobs_device) {
        if (input_mat_map.count(item.first) == 0) {
            LOGE("input mat map not found blob data (name: %s)\n", item.first.c_str());
            return Status(TNNERR_COMMON_ERROR, "input mat not match with blobs");
        }
        MatConvertParam param;
        status = instance_device_->SetInputMat(input_mat_map[item.first], param, item.first);
        RETURN_ON_NEQ(status, TNN_OK);
    }

    return TNN_OK;
}

Status ModelChecker::GetOutputRefData() {
    std::string output_file_name = model_checker_params_.ref_file.first;

    if ("" != output_file_name) {
        if (TEXT == model_checker_params_.ref_file.second) {
            FileReader file_reader;
            auto status = file_reader.Read(output_ref_mat_map_, output_file_name, TEXT);
            if (status != TNN_OK) {
                LOGE("read input file (%s) falied!\n", output_file_name.c_str());
                return Status(TNNERR_COMMON_ERROR, "read input failed");
            }
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

Status ModelChecker::GetOutputData(Instance* instance, std::map<std::string, std::shared_ptr<Mat>>& output_map) {
    BlobMap output_blobs;
    instance->GetAllOutputBlobs(output_blobs);

    for (auto blobs_item : output_blobs) {
        std::shared_ptr<Mat> mat;
        auto blob_name = blobs_item.first;

        auto ret = instance->GetOutputMat(mat, MatConvertParam(), blob_name, DEVICE_NAIVE, NCHW_FLOAT);
        if (ret != TNN_OK) {
            LOGE("get output mat falied (%s)\n", ret.description().c_str());
            return ret;
        }

        output_map[blob_name] = mat;
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
            if (!output_ref_mat_map_.empty()) {
                if (output_blobs_device.find(blob_name) != output_blobs_device.end()) {
                    if (output_ref_mat_map_.find(blob_name) != output_ref_mat_map_.end()) {
                        auto compare_data = output_ref_mat_map_[blob_name]->GetData();
                        is_pass &= CompareData(output_data_ptr, compare_data, blob_desc.dims);
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
