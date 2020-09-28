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

#include "hand_detect_tracker.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn_sdk_sample.h"
#include "hand_detector.h"
#include "hand_tracker.h"

namespace TNN_NS {
Status HandDetectTracker::Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks) {
    if (sdks.size() < 2) {
        return Status(TNNERR_INST_ERR, "HandDetectTracker::Init has invalid sdks, its size < 2");
    }

    predictor_detect_ = sdks[0];
    predictor_track_  = sdks[1];
    this->need_hand_detect = true;
    return TNNSDKComposeSample::Init(sdks);
}

Status HandDetectTracker::Predict(std::shared_ptr<TNNSDKInput> sdk_input,
                                  std::shared_ptr<TNNSDKOutput> &sdk_output) {
    Status status = TNN_OK;

    if (!sdk_input || sdk_input->IsEmpty()) {
        status = Status(TNNERR_PARAM_ERR, "input image is empty ,please check!");
        LOGE("input image is empty ,please check!\n");
        return status;
    }
    auto predictor_detect_async = predictor_detect_;
    auto predictor_track_async  = predictor_track_;
    auto predictor_detect_cast = dynamic_cast<HandDetector *>(predictor_detect_async.get());
    auto predictor_track_cast = dynamic_cast<HandTracking *>(predictor_track_async.get());

    auto image_mat = sdk_input->GetMat();
    const int image_orig_height = image_mat->GetHeight();
    const int image_orig_width = image_mat->GetWidth();

    // output of each model
    std::shared_ptr<TNNSDKOutput> sdk_output_detect = nullptr;
    std::shared_ptr<TNNSDKOutput> sdk_output_track  = nullptr;

    //phase1 model
    if (need_hand_detect) {
        status = predictor_detect_cast->Predict(std::make_shared<HandDetectorInput>(image_mat), sdk_output_detect);
        RETURN_ON_NEQ(status, TNN_OK);

        std::vector<ObjectInfo> hands;
        if (sdk_output_detect && dynamic_cast<HandDetectorOutput *>(sdk_output_detect.get())) {
            auto hand_output = dynamic_cast<HandDetectorOutput *>(sdk_output_detect.get());
            hands = hand_output->hands;
        }
        if (hands.size() <= 0) {
            // no hands, return
            LOGD("Error no hands found!\n");
            return status;
        }
        auto hand = hands[0];
        auto hand_orig = hand.AdjustToViewSize(image_orig_height, image_orig_width, 2);
        // set hands to the phase2 model
        predictor_track_cast->SetHandRegion(hand_orig.x1, hand_orig.y1, hand_orig.x2, hand_orig.y2);
    }

    //phase2 model
    {
        status = predictor_track_cast->Predict(std::make_shared<HandTrackingInput>(image_mat), sdk_output_track);
        RETURN_ON_NEQ(status, TNN_OK);
        need_hand_detect = predictor_track_cast->NeedHandDetect();
    }

    // generate results
    {
        sdk_output = std::make_shared<HandTrackingOutput>();
        auto output = dynamic_cast<HandTrackingOutput *>(sdk_output.get());

        auto phase2_output = dynamic_cast<HandTrackingOutput *>(sdk_output_track.get());
        auto hand = phase2_output->hand_list[0];
        auto hand_orig = hand.AdjustToViewSize(image_orig_height, image_orig_width, 2);

        output->hand_list.push_back(hand_orig);
    }
    return TNN_OK;
}
}

