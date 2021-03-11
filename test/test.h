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

#ifndef TNN_TEST_TEST_H_
#define TNN_TEST_TEST_H_

#include "tnn/core/blob.h"
#include "tnn/core/common.h"
#include "tnn/core/instance.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/utils/blob_converter.h"

namespace TNN_NS {

namespace test {

    int Run(int argc, char* argv[]);

    bool ParseAndCheckCommandLine(int argc, char* argv[]);

    void ShowUsage();

    void SetCpuAffinity();

    InputShapesMap GetInputShapesMap();

    ModelConfig GetModelConfig();

    NetworkConfig GetNetworkConfig();

    bool CheckResult(std::string desc, Status result);

    MatMap CreateBlobMatMap(BlobMap& blob_map, int mat_type);

    void InitInputMatMap(MatMap& mat_map);

    std::map<std::string, std::shared_ptr<BlobConverter>> CreateBlobConverterMap(BlobMap& blob_map);

    std::map<std::string, MatConvertParam> CreateConvertParamMap(MatMap& mat_map, bool is_input);

    void WriteOutput(MatMap& outputs);

    void FreeMatMapMemory(MatMap& mat_map);

}  // namespace test

}  // namespace TNN_NS

#endif  // TNN_TEST_TEST_H_
