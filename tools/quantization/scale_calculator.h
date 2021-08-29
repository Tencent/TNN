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

#ifndef TNN_TOOLS_QUANTIZATION_SCALE_CALCULATOR_H_
#define TNN_TOOLS_QUANTIZATION_SCALE_CALCULATOR_H_

#include <map>
#include <string>
#include <vector>

#include "tnn/core/blob.h"
#include "tnn/core/status.h"

#include "calibration_common.h"

namespace TNN_NS {

class ScaleCalculator {
public:
    // @brief ScaleCalculator constructor
    ScaleCalculator();

    // @brief ScaleCalculator virtual Destructor
    virtual ~ScaleCalculator();

public:
    // @brief: init with a blob and a method.
    // param 0 : input_blob, can be a device blob
    // param 1 : flag to merge_channel
    // param 2 : method, kl, minmax or admm
    int Init(Blob* blob, bool merge_channel = true, CalibrationMethod method = MIN_MAX);

    // @brief: set the quantize method
    // param 0 : method, the method to set
    int SetQuantizeMethod(CalibrationMethod method);

    // @brief: set merge channel param
    // param 0 : method, the method to set
    void SetMergeChannel(bool merge);

    // @brief: clear range_done_flag_.
    void ClearRangeFlag();

    // @brief: clear distribute_done_flag_.
    void ClearDistributeFlag();

    // @brief: update the blob data.
    int UpdateRange();

    // @brief: reset distribute according range.
    int ResetDistribute();

    // @brief: update distribute.
    int UpdateDistribute();

    // @brief: get the per-channel scale of the given blob
    int CalculateScale(std::vector<float>& val);
    int CalculateScale(std::vector<float>& val, std::vector<int8_t>& bias);

private:
    // @brief: statistical-based methods
    int CalculateScalePerDis(std::vector<float>& distribute, float interval, float& output);
    // @brief: analytical-based methods
    int CalculateScaleAnalysis(int channel_index, float& blob_scale, int8_t& bias);

    Blob* origin_blob_;
    bool merge_channel_;
    CalibrationMethod cali_method_;
    int bin_nums_;
    bool range_done_flag_;
    bool distribute_done_flag_;
    std::vector<std::pair<float, float>> range_per_channel_;
    std::vector<float> interval_per_channel_;
    std::vector<float> mean_per_channel_;
    std::vector<float> mean_abs_per_channel_;
    std::vector<int> index_image_per_channel_;
    std::vector<bool> valid_channel_;
    std::vector<std::vector<float>> distribute_per_channel_;
};

}  // namespace TNN_NS

#endif  // TNN_TOOLS_QUANTIZATION_SCALE_CALCULATOR_H_
