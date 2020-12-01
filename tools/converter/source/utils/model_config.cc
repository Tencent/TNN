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

#include "model_config.h"
namespace TNN_CONVERTER {

ModelConfig::~ModelConfig(){};

ModelConfig::ModelConfig(std::string model_type, std::string proto_path, std::string model_path, std::string tnn_path) {
    // TODO
    proto_path_ = proto_path;
    model_path_ = model_path;
    output_dir_ = tnn_path;
}
ModelConfig::ModelConfig(std::string model_type, std::string model_path, std::string output_dir) {
    // TODO
    if (model_type == "TFLITE") {
        model_type_ = MODEL_TYPE_TF_LITE;
    } else if (model_type == "ONNX") {
        model_type_ = MODEL_TYPE_ONNX;
    }
    model_path_ = model_path;
    output_dir_ = output_dir;
}

bool ModelConfig::CheckPath(std::string path) {
    auto filepath = path.c_str();
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (fs.is_open()) {
        fs.close();
        return true;
    }

    fprintf(stderr, "open failed %s\n", filepath);

    return false;
}
bool ModelConfig::CheckDir(std::string dir) {
    auto dirpath = dir.c_str();
    DIR* root    = opendir(dirpath);

    if (root != NULL) {
        closedir(root);
        return true;
    }

    fprintf(stderr, "open failed %s\n", dirpath);
    return false;
}
}  // namespace TNN_CONVERTER
