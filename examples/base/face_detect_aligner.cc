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

#include "face_detect_aligner.h"
#include "tnn/utils/dims_vector_utils.h"
#include "blazeface_detector.h"
#include "tnn_sdk_sample.h"
#include "youtu_face_align.h"

namespace TNN_NS {
Status FaceDetectAligner::Init(std::vector<std::shared_ptr<TNNSDKSample>> sdks) {
    if (sdks.size() < 3) {
        return Status(TNNERR_INST_ERR, "FaceDetectAligner::Init has invalid sdks, its size < 3");
    }
    
    predictor_detect_ = sdks[0];
    predictor_align_phase1_ = sdks[1];
    predictor_align_phase2_ = sdks[2];
    return TNNSDKComposeSample::Init(sdks);
}

Status FaceDetectAligner::Predict(std::shared_ptr<TNNSDKInput> sdk_input,
                                  std::shared_ptr<TNNSDKOutput> &sdk_output) {
    Status status = TNN_OK;
    
    if (!sdk_input || sdk_input->IsEmpty()) {
        status = Status(TNNERR_PARAM_ERR, "input image is empty ,please check!");
        LOGE("input image is empty ,please check!\n");
        return status;
    }
    auto predictor_detect_async = predictor_detect_;
    auto predictor_align_phase1_async = predictor_align_phase1_;
    auto predictor_align_phase2_async = predictor_align_phase2_;
    auto predictor_align1_cast = dynamic_cast<YoutuFaceAlign *>(predictor_align_phase1_async.get());
    auto predictor_align2_cast = dynamic_cast<YoutuFaceAlign *>(predictor_align_phase2_async.get());
    
    auto image_mat = sdk_input->GetMat();
    const int image_orig_height = image_mat->GetHeight();
    const int image_orig_width = image_mat->GetWidth();
    
    // output of each model
    std::shared_ptr<TNNSDKOutput> sdk_output_face = nullptr;
    std::shared_ptr<TNNSDKOutput> sdk_output1 = nullptr;
    std::shared_ptr<TNNSDKOutput> sdk_output2 = nullptr;
    
    std::shared_ptr<TNN_NS::Mat> phase1_pts = nullptr;
    
    //phase1 model
    {
        // 1) prepare input for phase1 model
        if(!has_prev_face_) {
            // i) get face from detector
            auto facedetector_input_dims = predictor_detect_->GetInputShape();
            
            //preprocess
            auto input_mat = std::make_shared<TNN_NS::Mat>(image_mat->GetDeviceType(), image_mat->GetMatType(), facedetector_input_dims);
            
            status = predictor_detect_async->Resize(image_mat, input_mat, TNNInterpLinear);
            RETURN_ON_NEQ(status, TNN_OK);
            
            status = predictor_detect_async->Predict(std::make_shared<BlazeFaceDetectorInput>(input_mat), sdk_output_face);
            RETURN_ON_NEQ(status, TNN_OK);
            
            std::vector<BlazeFaceInfo> face_info;
            if (sdk_output_face && dynamic_cast<BlazeFaceDetectorOutput *>(sdk_output_face.get()))
            {
                auto face_output = dynamic_cast<BlazeFaceDetectorOutput *>(sdk_output_face.get());
                face_info = face_output->face_list;
            }
            if(face_info.size() <= 0) {
                //no faces, return
                LOGD("Error no faces found!\n");
                return status;
            }
            auto face = face_info[0];
            // scale the face point according to the original image size
            auto face_orig = face.AdjustToViewSize(image_orig_height, image_orig_width, 2);
            LOGD("face_origin:(%f,%f,%f,%f), conf=%.4f\n", face_orig.x1, face_orig.y1, face_orig.x2, face_orig.y2, face_orig.score);
            
            // set face region for phase1 model
            if (!(predictor_align1_cast &&
                  predictor_align1_cast->SetFaceRegion(face_orig.x1, face_orig.y1, face_orig.x2, face_orig.y2))) {
                //no invalid faces, return
                LOGD("Error no valid faces found!\n");
                return status;
            }
        }
        
        // 2) predict
        status = predictor_align1_cast->Predict(std::make_shared<YoutuFaceAlignInput>(image_mat), sdk_output1);
        RETURN_ON_NEQ(status, TNN_OK);
        
        // update prev_face
        has_prev_face_ = predictor_align1_cast->GetPrevFace();
        if(!has_prev_face_) {
            LOGD("Next frame will use face detector!\n");
        }
        phase1_pts = predictor_align1_cast->GetPrePts();
    }
    
    //phase 2
    std::shared_ptr<TNN_NS::Mat> phase2_pts = nullptr;
    //phase2 model
    {
        // 1) prepare phase1 pts
        predictor_align2_cast->SetPrePts(phase1_pts, true);
        // 2) predict
        status = predictor_align2_cast->Predict(std::make_shared<YoutuFaceAlignInput>(image_mat), sdk_output2);
        RETURN_ON_NEQ(status, TNN_OK);
        phase2_pts = predictor_align2_cast->GetPrePts();
    }
    

    {
        sdk_output = std::make_shared<YoutuFaceAlignOutput>();
        auto phase1_output = dynamic_cast<YoutuFaceAlignOutput *>(sdk_output1.get());
        auto phase2_output = dynamic_cast<YoutuFaceAlignOutput *>(sdk_output2.get());

        auto& points        = phase1_output->face.key_points;
        auto& points_phase2 = phase2_output->face.key_points;

        points.insert(points.end(), points_phase2.begin(), points_phase2.end());

        auto output = dynamic_cast<YoutuFaceAlignOutput *>(sdk_output.get());
        output->face.key_points = points;
        output->face.image_height = image_orig_height;
        output->face.image_width  = image_orig_width;
    }
    return TNN_OK;
}
}
