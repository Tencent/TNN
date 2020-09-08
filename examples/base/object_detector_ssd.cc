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

#include "object_detector_ssd.h"
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <sys/time.h>

namespace TNN_NS {

ObjectDetectorSSDOutput::~ObjectDetectorSSDOutput() {}

ObjectDetectorSSD::~ObjectDetectorSSD() {}

std::shared_ptr<Mat> ObjectDetectorSSD::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
                                                                   std::string name) {
    auto target_dims = GetInputShape(name);
    auto input_height = input_mat->GetHeight();
    auto input_width = input_mat->GetWidth();
    if (target_dims.size() >= 4 &&
        (input_height != target_dims[2] || input_width != target_dims[3])) {
        auto target_mat = std::make_shared<TNN_NS::Mat>(input_mat->GetDeviceType(),
                                                        input_mat->GetMatType(), target_dims);
        auto status = Resize(input_mat, target_mat, TNNInterpLinear);
        if (status == TNN_OK) {
            return target_mat;
        } else {
            LOGE("%s\n", status.description().c_str());
            return nullptr;
        }
    }
    return input_mat;
}

MatConvertParam ObjectDetectorSSD::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_convert_param;
    input_convert_param.scale = {1.0 / (255 * 0.229), 1.0 / (255 * 0.224), 1.0 / (255 * 0.225), 0.0};
    input_convert_param.bias  = {-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225, 0.0};
    return input_convert_param;
}

std::shared_ptr<TNNSDKOutput> ObjectDetectorSSD::CreateSDKOutput() {
    return std::make_shared<ObjectDetectorSSDOutput>();
}

Status ObjectDetectorSSD::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    
    auto output = dynamic_cast<ObjectDetectorSSDOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));
    
    auto output_mat = output->GetMat();
    RETURN_VALUE_ON_NEQ(!output_mat, false,
                        Status(TNNERR_PARAM_ERR, "GetMat is invalid"));
    
    auto input_shape = GetInputShape();
    RETURN_VALUE_ON_NEQ(input_shape.size() == 4, true,
                        Status(TNNERR_PARAM_ERR, "GetInputShape is invalid"));
    
    auto num_detections = output_mat->GetHeight();
    
    std::vector<ObjectInfo> object_list;
    GenerateDetectResult(output_mat, object_list, num_detections, input_shape[3], input_shape[2]);
    output->object_list = object_list;
    return status;
}

void ObjectDetectorSSD::GenerateDetectResult(std::shared_ptr<TNN_NS::Mat> output, std::vector<ObjectInfo>& detecs,
                                             int num_detections, int image_width, int image_height) {
    float* data = reinterpret_cast<float*>(output->GetData());
    auto clip = [](float v){
        return std::min(v>0.0?v:0.0, 1.0);
    };
    
    for(int i=0; i<num_detections; ++i) {
        ObjectInfo info;
        info.image_width = image_width;
        info.image_height = image_height;
        
        info.class_id = data[i*7+1];
        if(info.class_id < 0 || info.class_id >= sizeof(voc_classes)/sizeof(*voc_classes)) {
            //LOGE("invalid object classid:%d\n", info.class_id);
            continue;
        }
        info.score = data[i*7+2];
        info.x1 = clip(data[i*7+3])*image_width;
        info.y1 = clip(data[i*7+4])*image_height;
        info.x2 = clip(data[i*7+5])*image_width;
        info.y2 = clip(data[i*7+6])*image_height;
        
        detecs.push_back(std::move(info));
    }
}

}
