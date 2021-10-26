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

#include "object_detector_nanodet.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <unordered_set>

namespace TNN_NS {

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

template <typename T>
void fast_softmax(const T *src, T *dst, const int length) {
    const T alpha = *std::max_element(src, src+length);
    T denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }
}

ObjectDetectorNanodet::~ObjectDetectorNanodet() {}

Status ObjectDetectorNanodet::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<ObjectDetectorNanodetOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));

    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);
    
    score_threshold = option->score_threshold;
    iou_threshold   = option->iou_threshold;
    if (option->model_cfg == "m" || option->model_cfg == "t") {
        reg_max = 7;
    } else if (option->model_cfg == "e1") {
        reg_max = 10;
    } else {
        return Status(TNNERR_PARAM_ERR, "Invalid Nanodet model_cfg!");
    }
    return status;
}

MatConvertParam ObjectDetectorNanodet::GetConvertParamForInput(std::string name) {
    // nanodet requires input in BGR
    MatConvertParam input_convert_param;
    input_convert_param.scale = {0.017429f,       0.017507f,      0.017125f,        0.0};
    input_convert_param.bias  = {103.53*0.017429, 116.28*0.017507, 123.675*0.017125, 0.0};
    input_convert_param.reverse_channel = true;
    return input_convert_param;
}

std::shared_ptr<Mat> ObjectDetectorNanodet::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
                                                                   std::string name) {
    auto target_dims   = GetInputShape(name);
    auto target_height = target_dims[2];
    auto target_width  = target_dims[3];

    auto input_height  = input_mat->GetHeight();
    auto input_width   = input_mat->GetWidth();
    auto option = dynamic_cast<ObjectDetectorNanodetOption *>(option_.get());
    option->input_height = input_height;
    option->input_width  = input_width;

    if (input_height != target_height || input_width !=target_width) {
        const float scale = std::min(static_cast<float>(target_width) / input_width,
                                     static_cast<float>(target_height) / input_height);
        const int resized_width  = std::round(input_width * scale);
        const int resized_height = std::round(input_height * scale);

        DimsVector intermediate_shape = input_mat->GetDims();
        intermediate_shape[2] = resized_height;
        intermediate_shape[3] = resized_width;
        auto intermediate_mat = std::make_shared<Mat>(input_mat->GetDeviceType(), input_mat->GetMatType(), intermediate_shape);
        auto status = Resize(input_mat, intermediate_mat, TNNInterpLinear);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);

        // top, bottom, left, right
        pads[0] = (target_height - resized_height) / 2;
        pads[1] = (target_height - resized_height) - pads[0];
        pads[2] = (target_width  - resized_width) / 2;
        pads[3]  = (target_width  - resized_width) - pads[2];

        auto target_mat = std::make_shared<Mat>(intermediate_mat->GetDeviceType(), input_mat->GetMatType(), target_dims);
        status = CopyMakeBorder(intermediate_mat, target_mat, pads[0], pads[1], pads[2], pads[3], TNNBorderConstant);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);

        return target_mat;
    }
    return input_mat;
}

std::shared_ptr<TNNSDKOutput> ObjectDetectorNanodet::CreateSDKOutput() {
    return std::make_shared<ObjectDetectorNanodetOutput>();
}

Status ObjectDetectorNanodet::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    
    auto output = dynamic_cast<ObjectDetectorNanodetOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));

    std::vector<ObjectInfo> object_list;
    for(const auto& head : heads) {
        auto cls_mat = output->GetMat(head.cls_output);
        RETURN_VALUE_ON_NEQ(!cls_mat, false,
                            Status(TNNERR_PARAM_ERR, "GetMat is invalid"));
        auto dis_mat = output->GetMat(head.dis_output);
        RETURN_VALUE_ON_NEQ(!dis_mat, false,
                            Status(TNNERR_PARAM_ERR, "GetMat is invalid"));

        DecodeDetectionResult(cls_mat.get(), dis_mat.get(), head.stride, object_list);
    }
    NMS(object_list, output->object_list);

    return status;
}

void ObjectDetectorNanodet::NMS(std::vector<ObjectInfo>& objs, std::vector<ObjectInfo>& results) {
    ::TNN_NS::NMS(objs, results, iou_threshold, TNNHardNMS);
}

void ObjectDetectorNanodet::DecodeDetectionResult(Mat *cls_mat, Mat *dis_mat, const int stride, std::vector<ObjectInfo>& detecs) {
    const auto option = dynamic_cast<ObjectDetectorNanodetOption *>(option_.get());
    const float input_img_height = static_cast<float>(option->input_height);
    const float input_img_width  = static_cast<float>(option->input_width);
    const auto model_input_shape = GetInputShape();
    const int model_input_height = model_input_shape[2];
    const int model_input_width  = model_input_shape[3];
    const float scale_x = input_img_width / model_input_width;
    const float scale_y = input_img_height / model_input_height;
    const float scale = std::max(scale_x, scale_y);

    int feature_height = model_input_shape[2] / stride;
    int feature_width  = model_input_shape[3] / stride;

    // cls_mat shape: [1, feature_height * feature_width, num_class]
    // dis_mat shape: [1, feature_height * feature_width, 4*(reg_max+1)]
    const float *cls_ptr = static_cast<float *>(cls_mat->GetData());
    const float *box_dis = static_cast<float *>(dis_mat->GetData());
    const int dis_size   = 4 * (reg_max + 1);

    for(int idx=0; idx< feature_width * feature_height; ++idx) {
        int x = idx % feature_width;
        int y = idx / feature_width;
        auto max_idx = std::max_element(cls_ptr, cls_ptr+num_class);
        auto class_id  = std::distance(cls_ptr, max_idx);
        float score = *max_idx;
        if (score > score_threshold) {
            float center_x = (x + 0.5f) * stride;
            float center_y = (y + 0.5f) * stride;
            float dis[4] = {0.f, 0.f, 0.f, 0.f};
            for(int i=0; i<4; ++i) {
                float dis_activated[reg_max + 1];
                fast_softmax<float>(box_dis + i*(reg_max+1), dis_activated, reg_max+1);
                for(int j=0; j<reg_max+1; ++j) {
                    dis[i] += j * dis_activated[j];
                }
                dis[i] *= stride;
            }
            ObjectInfo info;
            info.image_height = input_img_height;
            info.image_width  = input_img_width;
            info.score = score;
            info.class_id = static_cast<int>(class_id);
            info.x1 = center_x - dis[0];
            info.y1 = center_y - dis[1];
            info.x2 = center_x + dis[2];
            info.y2 = center_y + dis[3];
            // rescale to input img size
            info.x1 = std::max((info.x1 - pads[2]) * scale, 0.f);
            info.x2 = std::min((info.x2 - pads[2]) * scale, input_img_width);
            info.y1 = std::max((info.y1 - pads[0]) * scale, 0.f);
            info.y2 = std::min((info.y2 - pads[0]) * scale, input_img_height);

            detecs.push_back(info);
        }
        cls_ptr += num_class;
        box_dis += dis_size;
    }
}

}

