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

#include "hand_tracking.h"
#include <sys/time.h>
#include <cmath>
#include <fstream>
#include <cstring>
#include <tuple>

namespace TNN_NS {

Status HandTracking::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<HandTrackingOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));

    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);

    auto input_dims = GetInputShape();
    option->input_height = input_dims[2];
    option->input_width  = input_dims[3];

    return status;
}

MatConvertParam HandTracking::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_convert_param;
    input_convert_param.scale = {1.0 / 128.0, 1.0 / 128.0, 1.0 / 128.0, 0.0};
    input_convert_param.bias  = {-1.0, -1.0, -1.0, 0.0};
    return input_convert_param;
}

std::shared_ptr<TNNSDKOutput> HandTracking::CreateSDKOutput() {
    return std::make_shared<HandTrackingOutput>();
}

std::shared_ptr<Mat> HandTracking::ProcessSDKInputMat(std::shared_ptr<Mat> input_image,
                                                                   std::string name) {
    auto target_dims = GetInputShape(name);
    auto input_height = input_image->GetHeight();
    auto input_width = input_image->GetWidth();

    // crop face region accorfing
    auto start_x = x1_;
    auto start_y = y1_;
    auto end_x   = x2_;
    auto end_y   = y2_;
    if (valid_hand_in_prev_frame_) {
        auto center_x = (x1_ + x2_) / 2;
        auto center_y = (y1_ + y2_) / 2;
        start_x = x1_ - (center_x - x1_) * 0.5;
        start_y = y1_ - (center_x - y1_) * 0.5;
        end_x   = x2_ + (x2_ - center_x) * 0.5;
        end_y   = y2_ + (y2_ - center_y) * 0.5;

        start_x = fmax(0, start_x);
        start_y = fmax(0, start_y);
        end_x   = fmin(end_x, input_width  - 2);
        end_y   = fmin(end_y, input_height - 2);
    }
    auto croped_dims = input_image->GetDims();
    croped_dims[2] = end_y - start_y + 1;
    croped_dims[3] = end_x - start_x + 1;
    auto croped_mat = std::make_shared<TNN_NS::Mat>(input_image->GetDeviceType(), input_image->GetMatType(), croped_dims);
    auto status = Crop(input_image, croped_mat, start_x, start_y);
    if (status != TNN_OK) {
        LOGI("%s\n", status.description().c_str());
        return nullptr;
    }

    std::shared_ptr<TNN_NS::Mat> input_mat = nullptr;
    if (target_dims[2] != croped_dims[2] || target_dims[3] != croped_dims[3]) {
        input_mat = std::make_shared<TNN_NS::Mat>(input_image->GetDeviceType(),
                                                        input_image->GetMatType(), target_dims);
        auto status = Resize(croped_mat, input_mat, TNNInterpLinear);
        if (status != TNN_OK) {
            LOGI("%s\n", status.description().c_str());
            return nullptr;
        }
    } else {
        input_mat = croped_mat;
    }

    return input_mat;
}

Status HandTracking::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto option = dynamic_cast<HandTrackingOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                           Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    auto output = dynamic_cast<HandTrackingOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
    Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));

    auto tracking_result  = output->GetMat("output0");
    auto prob             = output->GetMat("output1");
    // get score
    float score = static_cast<float*>(prob->GetData())[0];
    float *region_ptr = static_cast<float*>(tracking_result->GetData());
    ObjectInfo hand;
    auto x1 = region_ptr[0];
    auto y1 = region_ptr[1];
    auto x2 = region_ptr[2];
    auto y2 = region_ptr[3];
    hand.x1 = x1;
    hand.y1 = y1;
    hand.x2 = x2;
    hand.y2 = y2;
    output->hand_list.push_back(std::move(hand));
    if (score >= option->hand_presence_threshold) {
        valid_hand_in_prev_frame_ = true;
        // update hand region
        SetHandRegion(x1, y1, x2, y2);
    } else {
        valid_hand_in_prev_frame_ = false;
    }
    return status;
}

}

