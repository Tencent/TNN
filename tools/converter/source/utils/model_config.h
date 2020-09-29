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

#ifndef TNNCONVERTER_SRC_MODEL_CONFIG_H_
#define TNNCONVERTER_SRC_MODEL_CONFIG_H_

#include <dirent.h>
#include <sys/stat.h>

#include <fstream>
#include <string>

namespace TNN_CONVERTER {

typedef enum { MODEL_TYPE_CAFFE = 0, MODEL_TYPE_TF = 1, MODEL_TYPE_TF_LITE = 2, MODEL_TYPE_ONNX = 3 } ModelType;

class ModelConfig {
public:
    ModelConfig(std::string model_type, std::string proto_path, std::string model_path, std::string onnx_path);
    ModelConfig(std::string model_type, std::string model_path, std::string onnx_path);
    ~ModelConfig();

    std::string proto_path_;
    std::string model_path_;
    std::string output_dir_;
    ModelType model_type_;

private:
    bool CheckPath(std::string path);
    bool CheckDir(std::string dir);
};

}  // namespace TNN_CONVERTER

#endif  // TNNCONVERTER_SRC_MODEL_CONFIG_H_
