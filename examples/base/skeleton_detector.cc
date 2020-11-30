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
#include <sys/time.h>
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

    return status;
}

std::shared_ptr<Mat> SkeletonDetector::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat, std::string name) {
    this->orig_input_height = input_mat->GetHeight();
    this->orig_input_width  = input_mat->GetWidth();
    return TNNSDKSample::ResizeToInputShape(input_mat, name);
}

MatConvertParam SkeletonDetector::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_convert_param;
    // for reduce model, rgb input
    input_convert_param.scale = {0.01712475,  0.017507  ,  0.01742919, 0.0};
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
    
    std::vector<SkeletonInfo> keypoints;
    //decode keypoints
    GenerateSkeleton(keypoints, heatmap, option->min_threshold);
    output->keypoint_list = keypoints;
    
    return status;
}

void SkeletonDetector::GenerateSkeleton(std::vector<SkeletonInfo> &skeleton, std::shared_ptr<TNN_NS::Mat> heatmap,
                                        float threshold) {
    const int heatmap_channels = heatmap->GetChannel();
    const int heatmap_height   = heatmap->GetHeight();
    const int heatmap_width    = heatmap->GetWidth();

    float* heatmap_data = static_cast<float *>(heatmap->GetData());
    int idx = 0;

    for(int c=0; c<heatmap_channels; ++c) {
        float* data_c = heatmap_data + c * heatmap_height * heatmap_width;
        // locate the max value inside a channel
        int max_pos_h = -1;
        int max_pos_w = -1;
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
            continue;
        }
        SkeletonInfo info;
        info.score = max_val;
        info.image_width  = this->orig_input_width;
        info.image_height = this->orig_input_height;
        info.key_points.emplace_back(max_pos_w, max_pos_h);
        
        skeleton.push_back(std::move(info));
    }
    
}


}

