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

#include "blazepose_detector.h"
#include <cmath>
#include <fstream>
#include <cstring>

namespace TNN_NS {

float CalculateScale(float min_scale, float max_scale, int stride_index,
                     int num_strides) {
    if (num_strides == 1) {
        return (min_scale + max_scale) * 0.5f;
    } else {
        return min_scale + (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0f);
    }
}

Status BlazePoseDetector::Init(std::shared_ptr<TNNSDKOption> option_i) {
    Status status = TNN_OK;
    auto option = dynamic_cast<BlazePoseDetectorOption *>(option_i.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                        Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));

    status = TNNSDKSample::Init(option_i);
    RETURN_ON_NEQ(status, TNN_OK);

    auto input_dims = GetInputShape();
    option->input_height = input_dims[2];
    option->input_width  = input_dims[3];
    //init anchors
    GenerateAnchor(&anchors);
    return status;
}

std::shared_ptr<Mat> BlazePoseDetector::ProcessSDKInputMat(std::shared_ptr<Mat> mat,
                                                                   std::string name) {
    auto target_dims   = GetInputShape(name);
    auto target_height = target_dims[2];
    auto target_width  = target_dims[3];

    auto input_height  = mat->GetHeight();
    auto input_width   = mat->GetWidth();

    letterbox_pads.fill(0.f);
    const float input_aspect_ratio  = static_cast<float>(input_width) / input_height;
    const float output_aspect_ratio = static_cast<float>(target_width) / target_height;
    if (input_aspect_ratio < output_aspect_ratio) {
        // compute left and right padding.
        letterbox_pads[0] = (1.f - input_aspect_ratio / output_aspect_ratio) / 2.f;
        letterbox_pads[2] = letterbox_pads[0];
    } else if (output_aspect_ratio < input_aspect_ratio) {
        // compute top and bottom padding.
        letterbox_pads[1] = (1.f - output_aspect_ratio / input_aspect_ratio) / 2.f;
        letterbox_pads[3] = letterbox_pads[1];
    }

    if (input_height != target_height || input_width !=target_width) {
        const float scale = std::min(static_cast<float>(target_width) / input_width,
                                     static_cast<float>(target_height) / input_height);
        const int resized_width  = std::round(input_width * scale);
        const int resized_height = std::round(input_height * scale);

        // TODO: we should use INTER_AREA when scale<1.0
        auto interp_mode = scale < 1.0f ? TNNInterpLinear : TNNInterpLinear;
        DimsVector intermediate_shape = mat->GetDims();
        intermediate_shape[2] = resized_height;
        intermediate_shape[3] = resized_width;
        auto intermediate_mat = std::make_shared<Mat>(mat->GetDeviceType(), mat->GetMatType(), intermediate_shape);
        auto status = Resize(mat, intermediate_mat, interp_mode);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);

        const int top    = (target_height - resized_height) / 2;
        const int bottom = (target_height - resized_height) - top;
        const int left   = (target_width  - resized_width) / 2;
        const int right  = (target_width  - resized_width) - left;

        auto input_mat = std::make_shared<Mat>(intermediate_mat->GetDeviceType(), mat->GetMatType(), target_dims);
        status = CopyMakeBorder(intermediate_mat, input_mat, top, bottom, left, right, TNNBorderConstant);
        RETURN_VALUE_ON_NEQ(status, TNN_OK, nullptr);

        return input_mat;
    }
    return mat;
}

MatConvertParam BlazePoseDetector::GetConvertParamForInput(std::string tag) {
    MatConvertParam param;
    param.scale = {2.0 / 255.0, 2.0 / 255.0, 2.0 / 255.0, 0.0};
    param.bias  = {-1.0,        -1.0,        -1.0,        0.0};
    return param;
}

std::shared_ptr<TNNSDKOutput> BlazePoseDetector::CreateSDKOutput() {
    return std::make_shared<BlazePoseDetectorOutput>();
}

Status BlazePoseDetector::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    auto option = dynamic_cast<BlazePoseDetectorOption *>(option_.get());
    RETURN_VALUE_ON_NEQ(!option, false,
                           Status(TNNERR_PARAM_ERR, "TNNSDKOption is invalid"));
    auto output = dynamic_cast<BlazePoseDetectorOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false,
    Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));

    //TODO: tnn's output shape mismatches with tflite
    auto scores = output->GetMat("classificators"); //(1, 896, 1)
    auto boxes  = output->GetMat("regressors"); //(1, 896, 12)
    RETURN_VALUE_ON_NEQ(!scores, false,
                           Status(TNNERR_PARAM_ERR, "scores mat is nil"));
    RETURN_VALUE_ON_NEQ(!boxes, false,
                           Status(TNNERR_PARAM_ERR, "boxes mat is nil"));

    std::vector<BlazePoseInfo> poses;
    GenerateBBox(poses, *(scores.get()), *(boxes.get()), option->min_score_threshold);
    NMS(poses, output->body_list, option->min_suppression_threshold, TNNWeightedNMS);
    RemoveLetterBox(output->body_list);

    return status;
}

/*
 mediapipe ssd_anchors_calculator
 */
void BlazePoseDetector::GenerateAnchor(std::vector<Anchor>* anchors) {
    const int stride_size = static_cast<int>(anchor_options.strides.size());
    const int ar_size     = static_cast<int>(anchor_options.aspect_ratios.size());
    int layer_id = 0;
    while (layer_id < anchor_options.num_layers) {
        std::vector<float> anchor_height;
        std::vector<float> anchor_width;
        std::vector<float> aspect_ratios;
        std::vector<float> scales;

        // For same strides, we merge the anchors in the same order.
        int last_same_stride_layer = layer_id;
        while (last_same_stride_layer < stride_size &&
               anchor_options.strides[last_same_stride_layer] ==
               anchor_options.strides[layer_id]) {
            const float scale =
            CalculateScale(anchor_options.min_scale, anchor_options.max_scale,
                           last_same_stride_layer, stride_size);
            for (int aspect_ratio_id = 0; aspect_ratio_id < ar_size; ++aspect_ratio_id) {
                aspect_ratios.push_back(anchor_options.aspect_ratios[aspect_ratio_id]);
                scales.push_back(scale);
            }
            if (anchor_options.interpolated_scale_aspect_ratio > 0.0) {
                const float scale_next =
                last_same_stride_layer == stride_size - 1 ? 1.0f
                : CalculateScale(anchor_options.min_scale, anchor_options.max_scale,
                                 last_same_stride_layer + 1,
                                 stride_size);
                scales.push_back(std::sqrt(scale * scale_next));
                aspect_ratios.push_back(anchor_options.interpolated_scale_aspect_ratio);
            }
            last_same_stride_layer++;
        }

        for (int i = 0; i < aspect_ratios.size(); ++i) {
            const float ratio_sqrts = std::sqrt(aspect_ratios[i]);
            anchor_height.push_back(scales[i] / ratio_sqrts);
            anchor_width.push_back(scales[i] * ratio_sqrts);
        }

        const int stride = anchor_options.strides[layer_id];
        auto input_shape = GetInputShape();
        const int input_height = input_shape[2];
        const int input_wdith  = input_shape[3];
        int feature_map_height = std::ceil(1.0f * input_height / stride);
        int feature_map_width = std::ceil(1.0f * input_wdith / stride);

        for (int y = 0; y < feature_map_height; ++y) {
            for (int x = 0; x < feature_map_width; ++x) {
                for (int anchor_id = 0; anchor_id < anchor_height.size(); ++anchor_id) {
                    const float x_center = (x + anchor_options.anchor_offset_x) * 1.0f / feature_map_width;
                    const float y_center = (y + anchor_options.anchor_offset_y) * 1.0f / feature_map_height;

                    Anchor new_anchor;
                    new_anchor.x_center = x_center;
                    new_anchor.y_center = y_center;
                    new_anchor.w = 1.0f;
                    new_anchor.h = 1.0f;

                    anchors->push_back(new_anchor);
                }
            }
        }
        layer_id = last_same_stride_layer;
    }
}

void BlazePoseDetector::GenerateBBox(std::vector<BlazePoseInfo> &detects, Mat &scoreMat, Mat &boxMat,
                                     float min_score_threshold) {
    detects.clear();
    // check shape
    auto box_dims = boxMat.GetDims();
    auto score_dims = scoreMat.GetDims();
    if (box_dims[1] != decode_options.num_boxes ||
        box_dims[2] != decode_options.num_coords) {
        return;
    }
    if (score_dims[1] != decode_options.num_boxes ||
        score_dims[2] != decode_options.num_classes) {
        return;
    }

    const float* raw_boxes = static_cast<float *>(boxMat.GetData());
    const float* raw_scores = static_cast<float *>(scoreMat.GetData());

    std::vector<float> boxes(decode_options.num_boxes * decode_options.num_coords);
    // decode box
    DecodeBoxes(boxes, raw_boxes);
    // decode score
    std::vector<float> scores(decode_options.num_boxes);
    std::vector<int> classes(decode_options.num_boxes);
    DecodeScore(scores, classes, raw_scores);

    // generate output
    for (int i = 0; i < decode_options.num_boxes; ++i) {
        if (scores[i] < min_score_threshold) {
            continue;
        }
        BlazePoseInfo info;

        const int box_offset = i * decode_options.num_coords;
        float box_ymin = boxes[box_offset + 0];
        float box_xmin = boxes[box_offset + 1];
        float box_ymax = boxes[box_offset + 2];
        float box_xmax = boxes[box_offset + 3];
        float score = scores[i];
        int class_id = classes[i];
        info.score = score;
        info.class_id = class_id;
        info.x1 = box_xmin;
        info.y1 = box_ymin;
        info.x2 = box_xmax;
        info.y2 = box_ymax;

        // add keypoints.
        int keypoint_index = box_offset + decode_options.keypoint_coord_offset;
        for (int kp_id = 0; kp_id < decode_options.num_keypoints; kp_id += 1) {
            float kpx = boxes[keypoint_index + 0];
            float kpy = boxes[keypoint_index + 1];
            info.key_points.push_back(std::make_pair(kpx, kpy));

            keypoint_index += decode_options.num_values_per_keypoint;
        }
        detects.push_back(std::move(info));
    }
}

void BlazePoseDetector::DecodeBoxes(std::vector<float>& boxes, const float* raw_boxes) {
    auto input_shape = GetInputShape();
    const float input_h = input_shape[2];
    const float input_w = input_shape[3];

    for (int i = 0; i < decode_options.num_boxes; ++i) {
        const int box_offset = i * decode_options.num_coords;

        float y_center = raw_boxes[box_offset];
        float x_center = raw_boxes[box_offset + 1];
        float h = raw_boxes[box_offset + 2];
        float w = raw_boxes[box_offset + 3];
        if (decode_options.reverse_output_order) {
            x_center = raw_boxes[box_offset];
            y_center = raw_boxes[box_offset + 1];
            w = raw_boxes[box_offset + 2];
            h = raw_boxes[box_offset + 3];
        }

        x_center = x_center / input_w * anchors[i].w + anchors[i].x_center;
        y_center = y_center / input_h * anchors[i].h + anchors[i].y_center;
        h = h / input_h * anchors[i].h;
        w = w / input_w * anchors[i].w;

        const float ymin = y_center - h / 2.f;
        const float xmin = x_center - w / 2.f;
        const float ymax = y_center + h / 2.f;
        const float xmax = x_center + w / 2.f;

        boxes[i * decode_options.num_coords + 0] = ymin;
        boxes[i * decode_options.num_coords + 1] = xmin;
        boxes[i * decode_options.num_coords + 2] = ymax;
        boxes[i * decode_options.num_coords + 3] = xmax;

        int kp_offset = i * decode_options.num_coords + decode_options.keypoint_coord_offset;
        for (int k = 0; k < decode_options.num_keypoints; ++k) {
            float keypoint_y = raw_boxes[kp_offset];
            float keypoint_x = raw_boxes[kp_offset + 1];
            if (decode_options.reverse_output_order) {
                keypoint_x = raw_boxes[kp_offset];
                keypoint_y = raw_boxes[kp_offset + 1];
            }
            boxes[kp_offset] = keypoint_x / input_w * anchors[i].w + anchors[i].x_center;
            boxes[kp_offset + 1] = keypoint_y / input_h * anchors[i].h + anchors[i].y_center;

            kp_offset += decode_options.num_values_per_keypoint;
        }
    }
}

void BlazePoseDetector::DecodeScore(std::vector<float>& scores, std::vector<int>& classes, const float* raw_scores) {
    scores.resize(decode_options.num_boxes, 0);
    classes.resize(decode_options.num_boxes, -1);
    // Filter classes by scores.
    for (int i = 0; i < decode_options.num_boxes; ++i) {
        int class_id = -1;
        float max_score = -std::numeric_limits<float>::max();
        // Find the top score for box i.
        for (int score_idx = 0; score_idx < decode_options.num_classes; ++score_idx) {
            auto score = raw_scores[i * decode_options.num_classes + score_idx];
            if (decode_options.sigmoid_score) {
                if (decode_options.score_clipping_thresh != 0) {
                    score = score < -decode_options.score_clipping_thresh ?
                            -decode_options.score_clipping_thresh :
                            score;
                    score = score >  decode_options.score_clipping_thresh ?
                             decode_options.score_clipping_thresh :
                             score;
                }
                score = 1.0f / (1.0f + std::exp(-score));
            }
            if (max_score < score) {
                max_score = score;
                class_id = score_idx;
            }
        }
        scores[i] = max_score;
        classes[i] = class_id;
    }
}

void BlazePoseDetector::RemoveLetterBox(std::vector<BlazePoseInfo>& detects) {
    const float left = letterbox_pads[0];
    const float top  = letterbox_pads[1];
    const float left_and_right = letterbox_pads[0] + letterbox_pads[2];
    const float top_and_bottom = letterbox_pads[1] + letterbox_pads[3];
    for (auto& pose : detects) {
        pose.x1 = (pose.x1 - left) / (1.0f - left_and_right);
        pose.y1 = (pose.y1 - top)  / (1.0f - top_and_bottom);
        pose.x2 = (pose.x2 - left) / (1.0f - left_and_right);
        pose.y2 = (pose.y2 - top)  / (1.0f - top_and_bottom);
        for (int i = 0; i<pose.key_points.size(); ++i) {
            auto kp = pose.key_points[i];
            const float new_x = (kp.first  - left) / (1.0f - left_and_right);
            const float new_y = (kp.second - top)  / (1.0f - top_and_bottom);
            pose.key_points[i] = std::make_pair(new_x, new_y);
        }
    }
}

}
