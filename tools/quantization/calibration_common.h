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

#ifndef TNN_TOOLS_QUANTIZATION_CALIBRATION_COMMON_H_
#define TNN_TOOLS_QUANTIZATION_CALIBRATION_COMMON_H_

#include <map>
#include <string>
#include <vector>

#include "file_reader.h"

namespace TNN_NS {

typedef enum {
    /* min max method */
    MIN_MAX = 0,
    /* ADMM method */
    ADMM = 1,
    /* kl divergence method */
    KL_DIVERGENCE = 2,
    /* Asymmetric min max method */
    ASY_MIN_MAX = 3,
    /* ACIQ gaus */
    ACIQ_GAUS = 4,
    /* ACIQ laplace */
    ACIQ_LAPLACE = 5,
} CalibrationMethod;

struct DataSet {
    /* list of input file path and format */
    std::vector<std::pair<std::string, FileFormat>> file_list;

    /* input shape of the input files* */
    InputShapesMap input_shape;
};

struct CalibrationParam {
    CalibrationMethod blob_quantize_method    = MIN_MAX;
    CalibrationMethod weights_quantize_method = MIN_MAX;
    bool merge_blob_channel                   = false;
    bool merge_weights_channel                = false;
    std::vector<float> input_bias             = {0, 0, 0, 0};
    std::vector<float> input_scale            = {1.0f, 1.0f, 1.0f, 1.0f};
    bool reverse_channel                      = false;
};

}  // namespace TNN_NS

#endif  // TNN_TOOLS_QUANTIZATION_CALIBRATION_COMMON_H_
