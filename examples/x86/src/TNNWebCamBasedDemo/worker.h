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

#ifndef TNN_EXAMPLES_X86_SRC_WORKER_H_
#define TNN_EXAMPLES_X86_SRC_WORKER_H_

#include <string>
#include <memory>

#include <opencv2/core.hpp>

#include "tnn/core/status.h"
#include "ultra_face_detector.h"
#include "blazeface_detector.h"
#include "youtu_face_align.h"
#include "face_detect_aligner.h"
#include "tnn_fps_counter.h"

using TNN_NS::Status;

class Worker {
public:
    Status Init(std::string model_path); 
    Status FaceDetectWithPaint(cv::Mat &frame, cv::Mat &frame_paint);
    Status BlazeFaceDetectWithPaint(cv::Mat &frame, cv::Mat &frame_paint);
    Status AlignWithPaint(cv::Mat &frame, cv::Mat &frame_paint);
    Status FrocessFrame(cv::Mat &frame, cv::Mat &frame_paint);
private:
    Status DrawUI(cv::Mat &frame_paint);
    std::shared_ptr<TNN_NS::UltraFaceDetector> detecotr_;
    std::shared_ptr<TNNFPSCounter> fps_counter_;

    std::shared_ptr<TNN_NS::BlazeFaceDetector> blaze_detecotr_;
    std::shared_ptr<TNN_NS::FaceDetectAligner> aligner_;

};

#endif // TNN_EXAMPLES_X86_SRC_WORKER_H_M

