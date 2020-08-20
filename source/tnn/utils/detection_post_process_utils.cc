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

#include "detection_post_process_utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
namespace TNN_NS {

void DecodeBoxes(DetectionPostProcessLayerParam* param, DetectionPostProcessLayerResource* resource,
                 Blob* boxes_encoding_blob, const CenterSizeEncoding& scale_values, Blob* decode_boxes_blob) {
    const int num_boxes         = boxes_encoding_blob->GetBlobDesc().dims[1];
    const int box_coord_num     = boxes_encoding_blob->GetBlobDesc().dims[3];
    const int num_anchors       = param->num_anchors;
    const int anchors_coord_num = param->anchors_coord_num;
    assert(num_boxes == num_anchors);
    assert(box_coord_num >= 4);
    assert(anchors_coord_num == 4);
    const auto boxes_ptr   = (float*)(boxes_encoding_blob->GetHandle().base);
    const auto anchors_ptr = reinterpret_cast<const CenterSizeEncoding*>(resource->anchors_handle.force_to<void*>());
    auto decode_boxes_ptr  = reinterpret_cast<BoxCornerEncoding*>(decode_boxes_blob->GetHandle().base);
    CenterSizeEncoding box_center_size_encoding;
    CenterSizeEncoding anchor_center_size_encoding;
    for (int i = 0; i < num_boxes; ++i) {
        const int box_index         = i * box_coord_num;
        box_center_size_encoding    = *reinterpret_cast<const CenterSizeEncoding*>(boxes_ptr + box_index);
        anchor_center_size_encoding = anchors_ptr[i];
        float ycenter =
            box_center_size_encoding.y / scale_values.y * anchor_center_size_encoding.h + anchor_center_size_encoding.y;
        float xcenter =
            box_center_size_encoding.x / scale_values.x * anchor_center_size_encoding.w + anchor_center_size_encoding.x;
        float halfh =
            0.5f * static_cast<float>(exp(box_center_size_encoding.h / scale_values.h)) * anchor_center_size_encoding.h;
        float halfw =
            0.5f * static_cast<float>(exp(box_center_size_encoding.w / scale_values.w)) * anchor_center_size_encoding.w;
        auto& cur_box = decode_boxes_ptr[i];
        cur_box.ymin  = ycenter - halfh;
        cur_box.xmin  = xcenter - halfw;
        cur_box.ymax  = ycenter + halfh;
        cur_box.xmax  = xcenter + halfw;
    }
}

void NonMaxSuppressionMultiClassFastImpl(DetectionPostProcessLayerParam* param,
                                         DetectionPostProcessLayerResource* resource, Blob* decoded_boxes,
                                         Blob* class_predictions, Blob* detection_boxes, Blob* detection_class,
                                         Blob* detection_scores, Blob* num_detections) {
    const int num_boxes                 = decoded_boxes->GetBlobDesc().dims[0];
    const int num_classes               = param->num_classes;
    const int max_classes_per_anchor    = param->max_classes_per_detection;
    const int num_class_with_background = class_predictions->GetBlobDesc().dims[3];

    const int label_offset = num_class_with_background - num_classes;
    assert(max_classes_per_anchor > 0);
    const int num_categories_per_anchor = std::min(max_classes_per_anchor, num_classes);
    std::vector<float> max_scores;
    max_scores.resize(num_boxes);
    std::vector<int> sorted_class_indices;
    sorted_class_indices.resize(num_boxes * num_classes);
    const auto scores_start_ptr = static_cast<float*>(class_predictions->GetHandle().base);

    // sort scores on every anchor
    for (int idx = 0; idx < num_boxes; ++idx) {
        const auto box_scores = scores_start_ptr + idx * num_class_with_background + label_offset;
        auto class_indices    = sorted_class_indices.data() + idx * num_classes;

        std::iota(class_indices, class_indices + num_classes, 0);
        std::partial_sort(class_indices, class_indices + num_categories_per_anchor, class_indices + num_classes,
                          [&box_scores](const int i, const int j) { return box_scores[i] > box_scores[j]; });
        max_scores[idx] = box_scores[class_indices[0]];
    }

    std::vector<int> seleted;
    NonMaxSuppressionSingleClasssImpl(decoded_boxes, max_scores.data(), param->max_detections, param->nms_iou_threshold,
                                      param->nms_score_threshold, &seleted);

    const auto decodedBoxesPtr = reinterpret_cast<const BoxCornerEncoding*>(decoded_boxes->GetHandle().base);
    auto detection_boxes_ptr   = reinterpret_cast<BoxCornerEncoding*>(detection_boxes->GetHandle().base);
    auto detection_classes_ptr = static_cast<float*>(detection_class->GetHandle().base);
    auto detection_scores_ptr  = static_cast<float*>(detection_scores->GetHandle().base);
    auto num_detections_ptr    = static_cast<float*>(num_detections->GetHandle().base);

    int output_box_index = 0;
    for (const auto& selected_index : seleted) {
        const float* box_scores  = scores_start_ptr + selected_index * num_class_with_background + label_offset;
        const int* class_indices = sorted_class_indices.data() + selected_index * num_classes;
        for (int col = 0; col < num_categories_per_anchor; ++col) {
            int boxOffset                    = num_categories_per_anchor * output_box_index + col;
            detection_boxes_ptr[boxOffset]   = decodedBoxesPtr[selected_index];
            detection_classes_ptr[boxOffset] = class_indices[col];
            detection_scores_ptr[boxOffset]  = box_scores[class_indices[col]];
            output_box_index++;
        }
    }
    *num_detections_ptr = output_box_index;
}
void NonMaxSuppressionSingleClasssImpl(Blob* decoded_boxes, const float* scores, int max_detections,
                                       float iou_threshold, float score_threshold, std::vector<int32_t>* selected) {
    ASSERT(iou_threshold >= 0.0f && iou_threshold <= 1.0f);
    ASSERT(decoded_boxes->GetBlobDesc().dims.size() == 4);
    const int numBoxes = decoded_boxes->GetBlobDesc().dims[0];
    ASSERT(decoded_boxes->GetBlobDesc().dims[1] == 4);

    const int outputNum = std::min(max_detections, numBoxes);
    std::vector<float> scoresData(numBoxes);
    std::copy_n(scores, numBoxes, scoresData.begin());

    struct Candidate {
        int boxIndex;
        float score;
    };

    auto cmp = [](const Candidate bsI, const Candidate bsJ) { return bsI.score < bsJ.score; };

    std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)> candidatePriorityQueue(cmp);

    for (int i = 0; i < scoresData.size(); ++i) {
        if (scoresData[i] > score_threshold) {
            candidatePriorityQueue.emplace(Candidate({i, scoresData[i]}));
        }
    }

    // std::vector<float> selectedScores;
    Candidate nextCandidate;
    float iou, originalScore;

    const auto boxesPtr = static_cast<float*>(decoded_boxes->GetHandle().base);
    while (selected->size() < outputNum && !candidatePriorityQueue.empty()) {
        nextCandidate = candidatePriorityQueue.top();
        originalScore = nextCandidate.score;
        candidatePriorityQueue.pop();

        // Overlapping boxes are likely to have similar scores,
        // therefore we iterate through the previously selected boxes backwards
        // in order to see if `next_candidate` should be suppressed.
        bool should_select = true;
        for (int j = (int)selected->size() - 1; j >= 0; --j) {
            iou = IOU(boxesPtr, nextCandidate.boxIndex, selected->at(j));
            if (iou == 0.0) {
                continue;
            }
            if (iou > iou_threshold) {
                should_select = false;
            }
        }

        if (should_select) {
            selected->push_back(nextCandidate.boxIndex);
            // selectedScores.push_back(nextCandidate.score);
        }
    }
}
static inline float IOU(const float* boxes, int i, int j) {
    const float yMinI = std::min<float>(boxes[i * 4 + 0], boxes[i * 4 + 2]);
    const float xMinI = std::min<float>(boxes[i * 4 + 1], boxes[i * 4 + 3]);
    const float yMaxI = std::max<float>(boxes[i * 4 + 0], boxes[i * 4 + 2]);
    const float xMaxI = std::max<float>(boxes[i * 4 + 1], boxes[i * 4 + 3]);
    const float yMinJ = std::min<float>(boxes[j * 4 + 0], boxes[j * 4 + 2]);
    const float xMinJ = std::min<float>(boxes[j * 4 + 1], boxes[j * 4 + 3]);
    const float yMaxJ = std::max<float>(boxes[j * 4 + 0], boxes[j * 4 + 2]);
    const float xMaxJ = std::max<float>(boxes[j * 4 + 1], boxes[j * 4 + 3]);
    const float areaI = (yMaxI - yMinI) * (xMaxI - xMinI);
    const float areaJ = (yMaxJ - yMinJ) * (xMaxJ - xMinJ);
    if (areaI <= 0 || areaJ <= 0)
        return 0.0;
    const float intersectionYMin = std::max<float>(yMinI, yMinJ);
    const float intersectionXMin = std::max<float>(xMinI, xMinJ);
    const float intersectionYMax = std::min<float>(yMaxI, yMaxJ);
    const float intersectionXMax = std::min<float>(xMaxI, xMaxJ);
    const float intersectionArea = std::max<float>(intersectionYMax - intersectionYMin, 0.0) *
                                   std::max<float>(intersectionXMax - intersectionXMin, 0.0);
    return intersectionArea / (areaI + areaJ - intersectionArea);
}

}  // namespace TNN_NS