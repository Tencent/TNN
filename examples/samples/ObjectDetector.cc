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

#include "ObjectDetector.h"
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <sys/time.h>


ObjectDetector::ObjectDetector(int input_width, int input_length, int num_thread_) {
    num_thread = num_thread_;
    in_w  = input_width;
    in_h = input_length;
}

ObjectDetector::~ObjectDetector() {}

int ObjectDetector::Detect(std::shared_ptr<tnn::Mat> image, int image_height, int image_width, std::vector<ObjInfo> &obj_list) {
    if(!image || !image->GetData()) {
        std::cout << "image is empty, please check!" << std::endl;
        return -1;
    }
#if TNN_SDK_ENABLE_BENCHMARK
    bench_result_.Reset();
    for(int fcount = 0; fcount < bench_option_.forward_count; fcount++) {
        timeval tv_begin, tv_end;
        gettimeofday(&tv_begin, NULL);
#endif
        obj_list.clear();
        
        TNN_NS::MatConvertParam input_convert_param;
        // from ImageNet
        input_convert_param.scale = {1.0 / (255 * 0.229), 1.0 / (255 * 0.224), 1.0 / (255 * 0.225), 0.0};
        input_convert_param.bias  = {-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225, 0.0};
        auto status = instance_->SetInputMat(image, input_convert_param);
        if (status != TNN_NS::TNN_OK) {
            LOGE("input_convert.ConvertFromMatAsync Error: %s\n", status.description().c_str());
            return status;
        }
        
        status = instance_->ForwardAsync(nullptr);
        if (status != TNN_NS::TNN_OK) {
            LOGE("instance.Forward Error: %s\n", status.description().c_str());
            return status;
        }
        
        TNN_NS::MatConvertParam output_convert_param;
        std::shared_ptr<TNN_NS::Mat> output_mat = nullptr;
        status = instance_->GetOutputMat(output_mat, output_convert_param, detection_output_name_);
        if (status != TNN_NS::TNN_OK) {
            LOGE("GetOutputMat Error: %s\n", status.description().c_str());
            return status;
        }
        num_detections_ = output_mat->GetHeight();
        
#if TNN_SDK_ENABLE_BENCHMARK
        gettimeofday(&tv_end, NULL);
        double time_elapsed = (tv_end.tv_sec - tv_begin.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_begin.tv_usec) / 1000.0;
        bench_result_.AddTime(time_elapsed);
#endif
        GenerateDetectResult(output_mat, obj_list);
#if TNN_SDK_ENABLE_BENCHMARK
    }
#endif
    return 0;
}

void ObjectDetector::GenerateDetectResult(std::shared_ptr<TNN_NS::Mat> output, std::vector<ObjInfo>& detecs) {
    float* data = reinterpret_cast<float*>(output->GetData());
    auto clip = [](float v){
        return std::min(v>0.0?v:0.0, 1.0);
    };
    
    for(int i=0; i<num_detections_; ++i) {
        ObjInfo info;
        
        info.classid = data[i*7+1];
        if(info.classid < 0 || info.classid >= sizeof(voc_classes)/sizeof(*voc_classes)) {
            LOGE("invalid object classid:%dn", info.classid);
            continue;
        }
        info.score = data[i*7+2];
        info.x1 = clip(data[i*7+3])*in_w;
        info.y1 = clip(data[i*7+4])*in_h;
        info.x2 = clip(data[i*7+5])*in_w;
        info.y2 = clip(data[i*7+6])*in_h;
        
        detecs.push_back(std::move(info));
    }
    // TODO(fix detection results)
    // As the detection results seem not to be right, we manually only keep one detect for each class
    std::unordered_set<int>distinct_class;
    std::vector<ObjInfo> filtered;
    for(auto& obj:detecs){
        if(distinct_class.find(obj.classid) == distinct_class.end()){
            filtered.push_back(std::move(obj));
            distinct_class.insert(obj.classid);
        }
    }
    detecs = filtered;
}
