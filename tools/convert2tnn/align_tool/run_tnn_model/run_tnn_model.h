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

#ifndef TNN_TOOLS_CONVERT2TNN_ALIGN_TOOL_RUN_TNN_MODEL_H
#define TNN_TOOLS_CONVERT2TNN_ALIGN_TOOL_RUN_TNN_MODEL_H

#include <memory>

#include "file_reader.h"
#include "tnn/core/blob.h"
#include "tnn/core/instance.h"
#include "tnn/core/layer_type.h"
#include "tnn/core/status.h"
#include "tnn/core/tnn.h"

enum CompareType { DEFAULT = 0, COSINE = 1 };

class AlignTNNModel {
public:
    AlignTNNModel(std::string proto_file, std::string model_file, std::string dump_dir_path);
    TNN_NS::Status Init();
    TNN_NS::Status RunAlignTNNModel();

    virtual ~AlignTNNModel();

private:
    TNN_NS::ModelConfig GetModelConfig();
    TNN_NS::NetworkConfig GetNetworkConfig();
    TNN_NS::Status GetDumpBlobMap();
    TNN_NS::Status FeedInputData();
    TNN_NS::Status AlignModelPerLayer();
    bool CompareData(float* src_data, float* tnn_data, TNN_NS::DimsVector blob_dims, CompareType type = DEFAULT);
    void DumpBlobData(void* blob_data, TNN_NS::DimsVector blob_dims, std::string output_name);
    TNN_NS::Status GetDumpData(const std::string& file_path, std::vector<float>& data);

    bool is_align_                                  = true;
    std::shared_ptr<TNN_NS::TNN> tnn_cpu_           = nullptr;
    std::shared_ptr<TNN_NS::Instance> instance_cpu_ = nullptr;

    std::map<std::string, int> dump_blob_map_;
    std::vector<float> not_align_tnn_data_;
    TNN_NS::BlobDesc not_align_tnn_blob_decs_;
    std::string proto_file_path_;
    std::string model_file_path_;
    std::string dump_dir_path_;
    std::string input_file_path_;
    std::string dump_file_list_;
};

#endif  // TNN_TOOLS_CONVERT2TNN_ALIGN_TOOL_RUN_TNN_MODEL_H
