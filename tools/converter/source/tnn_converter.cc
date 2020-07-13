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

#include "utils/command.h"
#include "utils/model_config.h"
#include "utils/flags.h"
#include "tflite/tf_lite_converter.h"

namespace TNN_CONVERTER {
int Run(int argc, char* argv[]) {
    ParseCommandLine(argc, argv);

    ModelConfig model_config(FLAGS_mt, FLAGS_mp, FLAGS_op);
    if (model_config.model_type_ == TNN_CONVERTER::MODEL_TYPE_TF_LITE) {
        TFLite2Tnn tf_lite_2_tnn(model_config.model_path_);
        tf_lite_2_tnn.Convert2Tnn();
    }

    return 0;
}

}  // namespace TNN_CONVERTER
int main(int argc, char* argv[]) {
    return TNN_CONVERTER::Run(argc, argv);
}
