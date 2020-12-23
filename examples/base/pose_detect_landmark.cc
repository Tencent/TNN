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

#include "tnn_sdk_sample.h"
#include "pose_detect_landmark.h"
#include "blazepose_landmark.h"
#include "blazeface_detector.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
Status PoseDetectLandmark::Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks) {
    if (sdks.size() < 2) {
        return Status(TNNERR_INST_ERR, "FaceDetectAligner::Init has invalid sdks, its size < 2");
    }

    predictor_detect_ = sdks[0];
    predictor_landmark_ = sdks[1];
    auto predictor_landmark_cast = dynamic_cast<BlazePoseLandmark *>(predictor_landmark_.get());
    if (predictor_landmark_cast->isFullBody()) {
        this->detect2roi_option.keypoints_start_idx = 0;
        this->detect2roi_option.keypoints_end_idx = 1;
    } else {
        this->detect2roi_option.keypoints_start_idx = 2;
        this->detect2roi_option.keypoints_end_idx = 3;
    }
    return TNNSDKComposeSample::Init(sdks);
}

Status PoseDetectLandmark::Predict(std::shared_ptr<TNNSDKInput> sdk_input,
                                  std::shared_ptr<TNNSDKOutput> &sdk_output) {
    Status status = TNN_OK;

    if (!sdk_input || sdk_input->IsEmpty()) {
        status = Status(TNNERR_PARAM_ERR, "input image is empty ,please check!");
        LOGE("input image is empty ,please check!\n");
        return status;
    }
    auto predictor_detect_async = predictor_detect_;
    auto predictor_landmark_async = predictor_landmark_;
    auto predictor_detect_cast = dynamic_cast<BlazePoseDetector *>(predictor_detect_async.get());
    auto predictor_landmark_cast = dynamic_cast<BlazePoseLandmark *>(predictor_landmark_async.get());
    const unsigned int input_height = sdk_input->GetMat()->GetHeight();
    const unsigned int input_width  = sdk_input->GetMat()->GetWidth();

    // output of each model
    std::shared_ptr<TNNSDKOutput> sdk_output_detect = nullptr;
    std::shared_ptr<TNNSDKOutput> sdk_output_landmark = nullptr;

    // phase1: blazepose detect
    if (predictor_landmark_cast->NeedPoseDetection()) {
        status = predictor_detect_cast->Predict(sdk_input, sdk_output_detect);
        RETURN_ON_NEQ(status, TNN_OK);

        std::vector<BlazePoseInfo>* detects = nullptr;
        if (sdk_output_detect && dynamic_cast<BlazePoseDetectorOutput *>(sdk_output_detect.get())) {
            auto output = dynamic_cast<BlazePoseDetectorOutput *>(sdk_output_detect.get());
            detects = &(output->body_list);
        }
        if (!detects || detects->size() <= 0) {
            // no detects, return
            return status;
        }
        // set the original input shape
        predictor_landmark_cast->SetOrigianlInputShape(input_height, input_width);
        // only use the first detect
        predictor_landmark_cast->Detection2RoI((*detects)[0], this->detect2roi_option);
    }
    // phase2: blazepose landmark
    {
        // set the original input shape
        predictor_landmark_cast->SetOrigianlInputShape(input_height, input_width);
        status = predictor_landmark_cast->Predict(sdk_input, sdk_output_landmark);
        RETURN_ON_NEQ(status, TNN_OK);
    }

    //get output
    {
        sdk_output = sdk_output_landmark;
    }
    return TNN_OK;
}

}
