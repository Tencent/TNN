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

#include "generate_model.h"

#include "tnn/interpreter/tnn/model_packer.h"
namespace TNN_CONVERTER {


#define PROTO_SUFFIX std::string(".tnnproto")
#define MODEL_SUFFIX std::string(".tnnmodel");

std::string GetFileName(std::string& file_path) {
    auto pos_s = file_path.rfind('/');
    auto pos_e = file_path.rfind('.');
    pos_s == std::string::npos ? (pos_s = 0) : (pos_s++);

    if (pos_e == std::string::npos) {
        pos_e = file_path.length();
    }
    auto len = pos_e - pos_s;
    return file_path.substr(pos_s, len);
}

TNN_NS::Status GenerateModel(TNN_NS::NetStructure& net_structure, TNN_NS::NetResource& net_resource,
                             std::string& output_dir, std::string& file_name) {
    std::string proto_path = output_dir + file_name + PROTO_SUFFIX;
    std::string model_path = output_dir + file_name + MODEL_SUFFIX;
    printf("TNN Converter generate TNN proto path %s\n", proto_path.c_str());
    printf("TNN Converter generate TNN model path %s\n", model_path.c_str());
    TNN_NS::ModelPacker model_packer(&net_structure, &net_resource);
    Status status = model_packer.Pack(proto_path, model_path);
    if (status != TNN_OK) {
        LOGE("generate tnn model failed!\n");
        return TNN_NS::TNNERR_CONVERT_GENERATE_MODEL;
    }
    return TNN_NS::TNN_CONVERT_OK;
}

}  // namespace TNN_CONVERTER
