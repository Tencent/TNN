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

#include "hand_tracker.h"
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
    // hand tracker requires input in BGR
    input_convert_param.reverse_channel = true;
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

    this->input_shape = input_image->GetDims();

    // crop face region accorfing
    auto start_x = x1_;
    auto start_y = y1_;
    auto end_x   = x2_;
    auto end_y   = y2_;
    if (valid_hand_in_prev_frame_) {
        auto center_x = (x1_ + x2_) / 2;
        auto center_y = (y1_ + y2_) / 2;
        start_x = x1_ - (center_x - x1_) * 0.5;
        start_y = y1_ - (center_y - y1_) * 0.5;
        end_x   = x2_ + (x2_ - center_x) * 0.5;
        end_y   = y2_ + (y2_ - center_y) * 0.5;

        start_x = fmax(0, start_x);
        start_y = fmax(0, start_y);
        end_x   = fmin(end_x, input_width  - 2);
        end_y   = fmin(end_y, input_height - 2);

        x1_ = start_x;
        x2_ = end_x;
        y1_ = start_y;
        y2_ = end_y;
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

    auto heat_map         = output->GetMat("output0");
    auto prob             = output->GetMat("output1");
    // get score
    float* score_ptr = static_cast<float*>(prob->GetData());
    float score = score_ptr[1];

    std::vector<float> hand_locations;
    status = GetHandRegion(heat_map, hand_locations);
    RETURN_ON_NEQ(status, TNN_OK);

    ObjectInfo hand;
    hand.x1 = hand_locations[0];
    hand.y1 = hand_locations[1];
    hand.x2 = hand_locations[2];
    hand.y2 = hand_locations[3];
    hand.image_width  = this->input_shape[3];
    hand.image_height = this->input_shape[2];
    hand.score = score;
    output->hand_list.push_back(std::move(hand));
    if (score >= option->hand_presence_threshold) {
        valid_hand_in_prev_frame_ = true;
        // next frame will use tracking result, update hand region
        SetHandRegion(hand.x1, hand.y1, hand.x2, hand.y2);
    } else {
        // next frame will use hand detector
        SetHandRegion(0, 0, 0, 0);
        valid_hand_in_prev_frame_ = false;
    }
    return status;
}

Status HandTracking::GetHandRegion(std::shared_ptr<Mat> mat, std::vector<float>& locations) {
    // get the max value location
    auto arg_max_hw = [](const float* data, size_t size, size_t w) {
        auto idx = std::distance(data, std::max_element(data, data+size));
        return std::make_pair(idx % w, idx / w);
    };
    locations.clear();

    auto heat_map_data = static_cast<float *>(mat->GetData());
    auto dims = mat->GetDims();
    auto hw   = dims[2] * dims[3];
    RETURN_VALUE_ON_NEQ(dims[1], 2, Status(TNNERR_PARAM_ERR, "heat_map mat should have 2 channels!"));

    HandTrackingOption* option = dynamic_cast<HandTrackingOption *>(option_.get());
    auto target_height = option->input_height;
    auto target_width  = option->input_width;
    assert(target_height == 64);
    assert(target_width == 64);
    for(int c=0; c<dims[1]; ++c) {
        auto xy = arg_max_hw(heat_map_data + c*hw, hw, dims[3]);
        float x = xy.first  * (x2_ - x1_) / target_width  + x1_;
        float y = xy.second * (y2_ - y1_) / target_height + y1_;
        locations.push_back(x);
        locations.push_back(y);
    }

    return TNN_OK;
}

}

