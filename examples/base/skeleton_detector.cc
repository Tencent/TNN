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

#include "skeleton_detector.h"
#include <cmath>
#include <fstream>
#include <cstring>

namespace TNN_NS {

Status SkeletonDetector::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<SkeletonDetectorOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    
    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto input_dims = GetInputShape();
    option->input_height = input_dims[2];
    option->input_width  = input_dims[3];
    
    landmark_filter = std::make_shared<VelocityFilter>(this->window_size,
                                                       this->velocity_scale,
                                                       this->min_allowed_object_scale,
                                                       option->fps);

    return status;
}

std::shared_ptr<Mat> SkeletonDetector::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat, std::string name) {
    this->orig_input_height = input_mat->GetHeight();
    this->orig_input_width  = input_mat->GetWidth();
    return TNNSDKSample::ResizeToInputShape(input_mat, name);
}

MatConvertParam SkeletonDetector::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_convert_param;
    // rgb input required
    input_convert_param.scale = {0.01712475,   0.017507,     0.01742919,  0.0};
    input_convert_param.bias  = {-2.11790393,  -2.03571429,  -1.80444444, 0.0};

    return input_convert_param;
}

std::shared_ptr<TNNSDKOutput> SkeletonDetector::CreateSDKOutput() {
    return std::make_shared<SkeletonDetectorOutput>();
}

Status SkeletonDetector::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto option = dynamic_cast<SkeletonDetectorOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                           Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    auto output = dynamic_cast<SkeletonDetectorOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
    Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));
    
    auto heatmap = output->GetMat("heatmap");
    RETURN_VALUE_ON_NEQ(!heatmap, false,
                           Status(TNNERR_PARAM_ERR, "heatmap mat is nil"));
    
    //decode keypoints
    GenerateSkeleton(output, heatmap, option->min_threshold);
    SmoothingLandmarks(output);
    DeNormalize(output);
    
    return status;
}

void SkeletonDetector::GenerateSkeleton(SkeletonDetectorOutput* output,
                                        std::shared_ptr<TNN_NS::Mat> heatmap, float threshold) {
    SkeletonInfo& skeleton = output->keypoints;
    std::vector<float>& confidence_list = output->confidence_list;
    std::vector<bool>& detected = output->detected;

    const int heatmap_channels = heatmap->GetChannel();
    const int heatmap_height   = heatmap->GetHeight();
    const int heatmap_width    = heatmap->GetWidth();
    
    const int src_height = this->orig_input_height;
    const int src_width  = this->orig_input_width;

    float* heatmap_data = static_cast<float *>(heatmap->GetData());
    int idx = 0;
    skeleton.key_points.resize(heatmap_channels);
    confidence_list.resize(heatmap_channels);
    detected.resize(heatmap_channels);

    for(int c=0; c<heatmap_channels; ++c) {
        float* data_c = heatmap_data + c * heatmap_height * heatmap_width;
        // locate the max value inside a channel
        float max_pos_h = -1;
        float max_pos_w = -1;
        float max_val = -FLT_MAX;
        idx = 0;
        for(int h=0; h<heatmap_height; ++h) {
            for(int w=0; w<heatmap_width; ++w) {
                auto val = data_c[idx++];
                if ( val > max_val) {
                    max_val = val;
                    max_pos_h = h;
                    max_pos_w = w;
                }
            }
        }
        if (max_val < threshold) {
            skeleton.key_points[c] = std::make_pair(-1, -1);
            detected[c] = false;
        } else {
            skeleton.key_points[c] = std::make_pair(max_pos_w / heatmap_width,
                                                    max_pos_h / heatmap_height);
            detected[c] = true;
        }
        confidence_list[c] = max_val;
    }
    for(const auto& line:this->lines) {
        if (detected[line.first] && detected[line.second])
            skeleton.lines.push_back(line);
    }
    skeleton.image_width  = src_width;
    skeleton.image_height = src_height;
}

void SkeletonDetector::SmoothingLandmarks(SkeletonDetectorOutput* output) {
    std::vector<std::pair<float, float>> out_landmarks;
    landmark_filter->Apply2D(output->keypoints.key_points,
                           std::make_pair(orig_input_height, orig_input_width),
                           Now(),
                           &out_landmarks);
    if (out_landmarks.size() > 0) {
        output->keypoints.key_points = out_landmarks;
    }
}

void SkeletonDetector::DeNormalize(SkeletonDetectorOutput* output) {
    const int src_height = this->orig_input_height;
    const int src_width  = this->orig_input_width;

    SkeletonInfo& skeleton = output->keypoints;
    for(auto& lm2d: skeleton.key_points) {
        float x = lm2d.first * src_width;
        float y = lm2d.second * src_height;
        lm2d = std::make_pair(x, y);
    }
    skeleton.image_height = src_height;
    skeleton.image_width  = src_width;
}


}

