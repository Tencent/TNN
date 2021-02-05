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

#include "run_tnn_model.h"

#include <cmath>
#include <fstream>
#include <sstream>

#include "tnn/core/macro.h"
#include "tnn/utils/dims_vector_utils.h"

AlignTNNModel::AlignTNNModel(std::string proto_file, std::string model_file, std::string dump_dir_path) {
    proto_file_path_ = proto_file;
    model_file_path_ = model_file;
    dump_dir_path_   = dump_dir_path;

    dump_dir_path_ += (dump_dir_path_[dump_dir_path_.size() - 1] == '/' ? "" : "/");
    input_file_path_ = dump_dir_path_ + "input.txt";
    dump_file_list_  = dump_dir_path_ + "dump_file_list.txt";

    not_align_tnn_data_.clear();
}

AlignTNNModel::~AlignTNNModel() {
    instance_cpu_ = nullptr;
    tnn_cpu_      = nullptr;
}

TNN_NS::ModelConfig AlignTNNModel::GetModelConfig() {
    TNN_NS::ModelConfig model_config;
    {
        std::ifstream proto_stream(proto_file_path_);
        if (!proto_stream.is_open() || !proto_stream.good()) {
            printf("read proto_file failed!\n");
            return model_config;
        }
        auto buffer = std::string((std::istreambuf_iterator<char>(proto_stream)), std::istreambuf_iterator<char>());
        model_config.params.push_back(buffer);
    }

    {
        std::ifstream model_stream(model_file_path_);
        if (!model_stream.is_open() || !model_stream.good()) {
            printf("read model_file failed!\n");
            return model_config;
        }
        auto buffer = std::string((std::istreambuf_iterator<char>(model_stream)), std::istreambuf_iterator<char>());
        model_config.params.push_back(buffer);
    }

    return model_config;
}

TNN_NS::NetworkConfig AlignTNNModel::GetNetworkConfig() {
    TNN_NS::NetworkConfig net_config;
    net_config.device_type = TNN_NS::DEVICE_NAIVE;

    return net_config;
}

TNN_NS::Status AlignTNNModel::GetDumpBlobMap() {
    int num_blob = 0;
    std::ifstream f_stream(dump_file_list_);
    f_stream >> num_blob;
    if (num_blob <= 0) {
        LOGE("%s is invalid!\n", dump_dir_path_.c_str());
        return TNN_NS::Status(TNN_NS::TNNERR_COMMON_ERROR, "invalid output ref file, the wrong file formate!");
    }

    std::string blob_name = "";
    for (int i = 0; i < num_blob; i++) {
        f_stream >> blob_name;
        if (dump_blob_map_.find(blob_name) == dump_blob_map_.end()) {
            dump_blob_map_[blob_name] = 1;
        }
    }

    f_stream.close();

    return TNN_NS::TNN_OK;
}

TNN_NS::Status AlignTNNModel::Init() {
    auto model_config = GetModelConfig();
    auto net_config   = GetNetworkConfig();

    tnn_cpu_.reset(new TNN_NS::TNN());
    TNN_NS::Status status = tnn_cpu_->Init(model_config);
    if (status != TNN_NS::TNN_OK) {
        LOGE("tnn init falied: %s!\n", status.description().c_str());
        return TNN_NS::Status(TNN_NS::TNNERR_NET_ERR, "tnn init falied");
    }

    TNN_NS::NetworkConfig net_config_cpu;
    net_config_cpu.device_type = TNN_NS::DEVICE_NAIVE;
    if (net_config.device_type == TNN_NS::DEVICE_NAIVE) {
        net_config_cpu = net_config;
    }
    instance_cpu_ = tnn_cpu_->CreateInst(net_config_cpu, status);
    if (status != TNN_NS::TNN_OK) {
        LOGE("create cpu instance falied: %s\n", status.description().c_str());
        return status;
    }

    return TNN_NS::TNN_OK;
}

int GetMatElementSize(TNN_NS::Mat* mat) {
    TNN_NS::MatType mat_type = mat->GetMatType();
    if (TNN_NS::NCHW_FLOAT == mat_type || TNN_NS::NCDHW_FLOAT == mat_type) {
        return 4;
    } else if (TNN_NS::NC_INT32 == mat_type) {
        return 4;
    } else if (TNN_NS::N8UC3 == mat_type || TNN_NS::N8UC4 == mat_type || TNN_NS::NGRAY == mat_type ||
               TNN_NS::NNV21 == mat_type || TNN_NS::NNV12 == mat_type) {
        return 1;
    } else if (TNN_NS::RESERVED_BFP16_TEST == mat_type || TNN_NS::RESERVED_FP16_TEST == mat_type) {
        return 2;
    } else if (TNN_NS::RESERVED_INT8_TEST == mat_type) {
        return 1;
    } else {
        return 0;
    }
}

bool AlignTNNModel::IsDimsCanBeExtend(std::vector<int> src_dims, std::vector<int> dst_dims) {
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

TNN_NS::Status AlignTNNModel::ExtendMatMap(const TNN_NS::BlobMap& blobs_map,
                                           std::map<std::string, std::shared_ptr<TNN_NS::Mat>>& mat_map) {
    for (auto item : blobs_map) {
        auto blob_name = item.first;
        if (mat_map.count(blob_name) <= 0) {
            LOGE("mat map don't has blob data (name: %s)\n", blob_name.c_str());
            return TNN_NS::Status(TNN_NS::TNNERR_COMMON_ERROR, "extend falied: mat map is not match with blobs map");
        }

        auto mat      = mat_map[blob_name];
        auto src_dims = mat->GetDims();
        auto dst_dims = item.second->GetBlobDesc().dims;

        if (TNN_NS::DimsVectorUtils::Equal(src_dims, dst_dims)) {
            continue;
        }

        printf("Warning: mat map (name: %s) will try to be extended due to dims not match\n", blob_name.c_str());
        if (!IsDimsCanBeExtend(src_dims, dst_dims)) {
            return TNN_NS::Status(TNN_NS::TNNERR_COMMON_ERROR, "extend falied: dims can't be extend");
        }

        int bytesize_perbatch = TNN_NS::DimsVectorUtils::Count(src_dims, 1) * GetMatElementSize(mat.get());
        int src_batch_size    = src_dims[0];
        int dst_batch_size    = dst_dims[0];
        int src_bytesize      = bytesize_perbatch * src_batch_size;

        printf("batch extrend form %d to %d\n", src_batch_size, dst_batch_size);
        std::shared_ptr<TNN_NS::Mat> mat_new(new TNN_NS::Mat(mat->GetDeviceType(), mat->GetMatType(), dst_dims));
        int batch_idx = 0;
        for (; batch_idx < dst_batch_size - src_batch_size; batch_idx += src_batch_size) {
            memcpy((char*)mat_new->GetData() + batch_idx * bytesize_perbatch, mat->GetData(), src_bytesize);
        }
        int batch_left = dst_batch_size - batch_idx;
        memcpy((char*)mat_new->GetData() + batch_idx * bytesize_perbatch, mat->GetData(),
               batch_left * bytesize_perbatch);

        mat_map[blob_name] = mat_new;
    }

    return TNN_NS::TNN_OK;
}

TNN_NS::Status AlignTNNModel::FeedInputData() {
    std::vector<float> bias  = {0, 0, 0, 0};
    std::vector<float> scale = {1.0f, 1.0f, 1.0f, 1.0f};
    const auto file_format   = TNN_NS::TEXT;

    TNN_NS::BlobMap input_blobs_cpu;
    std::map<std::string, std::shared_ptr<TNN_NS::Mat>> input_mat_map;
    auto status = instance_cpu_->GetAllInputBlobs(input_blobs_cpu);
    RETURN_ON_NEQ(status, TNN_NS::TNN_OK);

    // feed cpu instance input
    if (input_file_path_ == "") {
        LOGE("input file path is empty!\n");
        return TNN_NS::Status(TNN_NS::TNNERR_INVALID_INPUT, "input file path is empty");
    }
    TNN_NS::FileReader file_reader;
    file_reader.SetBiasValue(bias);
    file_reader.SetScaleValue(scale);
    status = file_reader.Read(input_mat_map, input_file_path_, file_format);
    if (status != TNN_NS::TNN_OK) {
        LOGE("read input file (%s) falied!\n", input_file_path_.c_str());
        return TNN_NS::Status(TNN_NS::TNNERR_COMMON_ERROR, "read input failed");
    }
    status = ExtendMatMap(input_blobs_cpu, input_mat_map);
    RETURN_ON_NEQ(status, TNN_NS::TNN_OK);

    for (auto item : input_blobs_cpu) {
        if (input_mat_map.count(item.first) == 0) {
            LOGE("input mat map not found blob data (name: %s)\n", item.first.c_str());
            return TNN_NS::Status(TNN_NS::TNNERR_COMMON_ERROR, "input mat not match with blobs");
        }
        TNN_NS::MatConvertParam param;
        status = instance_cpu_->SetInputMat(input_mat_map[item.first], param, item.first);
        RETURN_ON_NEQ(status, TNN_NS::TNN_OK);
    }

    return TNN_NS::TNN_OK;
}

TNN_NS::Status AlignTNNModel::GetDumpData(const std::string& file_path, std::vector<float>& data) {
    int num_out;
    std::ifstream f_stream(file_path);
    f_stream >> num_out;
    if (num_out != 1) {
        LOGE("invalid dump file (%s)!  Please make sure the dump file format right\n", file_path.c_str());
        return TNN_NS::Status(TNN_NS::TNNERR_COMMON_ERROR, "invalid output ref file, the wrong file formate!");
    }

    int dims_size = 0;
    int dim       = 1;
    int dim_cnt   = 1;
    int data_type = 0;
    std::string name;
    for (int index = 0; index < num_out; index++) {
        f_stream >> name;
        f_stream >> dims_size;
        for (int i = 0; i < dims_size; i++) {
            f_stream >> dim;
            dim_cnt *= dim;
        }
        f_stream >> data_type;
        data.resize(dim_cnt);
        for (int line = 0; line < dim_cnt; line++) {
            f_stream >> data[line];
        }
    }

    f_stream.close();

    return TNN_NS::TNN_OK;
}

bool AlignTNNModel::CompareData(float* src_data, float* tnn_data, TNN_NS::DimsVector blob_dims, CompareType type) {
    int data_count = TNN_NS::DimsVectorUtils::Count(blob_dims);

    if (DEFAULT == type) {
        float ep = 0.005;
        for (unsigned long long i = 0; i < data_count; i++) {
            float diff = static_cast<float>(fabs(src_data[i] - tnn_data[i]));
            float sum  = static_cast<float>(fabs(src_data[i]) + fabs(tnn_data[i]));
            if (fabs(diff / sum) > ep && fabs(diff) > 1e-3f) {
                LOGE("ERROR AT %llu result %.6f ref %.6f  diff/sum %f  diff %f\n", i, src_data[i], tnn_data[i],
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
            float diff = static_cast<float>(fabs(src_data[i] - tnn_data[i]));
            if (diff > max_diff) {
                max_diff     = diff;
                max_diff_idx = i;
            }
            cpu_device_mul += src_data[i] * tnn_data[i];
            cpu_sum2 += tnn_data[i] * tnn_data[i];
            device_sum2 += src_data[i] * src_data[i];
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

TNN_NS::Status AlignTNNModel::AlignModelPerLayer() {
    TNN_NS::BlobStatisticCallback cpu_func_after = [&](std::vector<TNN_NS::Blob*>& blobs, TNN_NS::LayerInfo* info) {
        if (!is_align_) {
            return;
        }

        for (auto blob : blobs) {
            auto blob_name = blob->GetBlobDesc().name;
            if (dump_blob_map_.find(blob_name) == dump_blob_map_.end()) {
                continue;
            }

            std::vector<float> src_data;
            std::replace(blob_name.begin(), blob_name.end(), '/', '_');
            const auto dump_data_path = dump_dir_path_ + blob_name + ".txt";
            auto status               = GetDumpData(dump_data_path, src_data);
            if (status != TNN_NS::TNN_OK) {
                LOGE("load %s failed\n", dump_data_path.c_str());
                return;
            }

            const auto blob_desc = blob->GetBlobDesc();
            int tnn_data_count   = TNN_NS::DimsVectorUtils::Count(blob_desc.dims);
            std::vector<float> tnn_data(tnn_data_count, 0);
            auto* tnn_data_ptr = tnn_data.data();
            auto* blob_data    = blob->GetHandle().base;
            if (blob_desc.data_type == TNN_NS::DATA_TYPE_FLOAT) {
                memcpy(tnn_data_ptr, blob_data, tnn_data_count * sizeof(float));
            } else if (blob_desc.data_type == TNN_NS::DATA_TYPE_INT32) {
                auto* blob_data_ptr = (int*)blob_data;
                for (int i = 0; i < tnn_data_count; i++) {
                    tnn_data_ptr[i] = static_cast<float>(blob_data_ptr[i]);
                }
            } else {
                LOGE("unsupport data type!\n");
                return;
            }

            auto* src_data_ptr = src_data.data();
            is_align_ &= CompareData(src_data_ptr, tnn_data_ptr, blob_desc.dims);
            if (!is_align_ && not_align_tnn_data_.size() == 0) {
                not_align_tnn_data_      = tnn_data;
                not_align_tnn_blob_decs_ = blob_desc;
            }
        }
    };

    return instance_cpu_->ForwardWithCallback(nullptr, cpu_func_after);
}

void AlignTNNModel::DumpBlobData(void* blob_data, TNN_NS::DimsVector blob_dims, std::string output_name) {
    std::ofstream f_out(output_name.c_str());

    int count       = TNN_NS::DimsVectorUtils::Count(blob_dims);
    float* data_ptr = reinterpret_cast<float*>(blob_data);
    for (int index = 0; index < count; ++index) {
        f_out << data_ptr[index] << std::endl;
    }

    f_out.close();
}

TNN_NS::Status AlignTNNModel::RunAlignTNNModel() {
    auto status = GetDumpBlobMap();
    RETURN_ON_NEQ(status, TNN_NS::TNN_OK);

    status = FeedInputData();
    RETURN_ON_NEQ(status, TNN_NS::TNN_OK);

    status = AlignModelPerLayer();
    RETURN_ON_NEQ(status, TNN_NS::TNN_OK);

    if (!is_align_) {
        auto blob_name = not_align_tnn_blob_decs_.name;
        std::replace(blob_name.begin(), blob_name.end(), '/', ' ');
        const auto src_dump_path = dump_dir_path_ + blob_name + ".txt";
        const auto dump_path     = dump_dir_path_ + "tnn-" + blob_name + ".txt";
        auto* dump_data_ptr      = not_align_tnn_data_.data();
        DumpBlobData(dump_data_ptr, not_align_tnn_blob_decs_.dims, dump_path);

        printf("TNN model and src model not aligned at %s\n", blob_name.c_str());
        printf("You can find the output of %s of TNN at %s\n", blob_name.c_str(), dump_path.c_str());
        printf("You can find the output of %s of src model at %s\n", blob_name.c_str(), src_dump_path.c_str());

        return TNN_NS::Status(TNN_NS::TNNERR_COMMON_ERROR, "align model failed");
    }

    return TNN_NS::TNN_OK;
}
