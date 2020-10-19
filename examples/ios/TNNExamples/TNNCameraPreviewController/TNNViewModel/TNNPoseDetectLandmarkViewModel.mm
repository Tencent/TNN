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

#import "TNNPoseDetectLandmarkViewModel.h"
#import "blazepose_detector.h"
#import "blazepose_landmark.h"
#import "pose_detect_landmark.h"

using namespace std;

@implementation TNNPoseDetectLandmarkViewModel

- (std::shared_ptr<BlazePoseDetector>) loadPoseDetector:(TNNComputeUnits)units {
    std::shared_ptr<BlazePoseDetector> predictor = nullptr;

    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path   = [[NSBundle mainBundle] pathForResource:@"model/blazepose/pose_detection.tnnmodel"
                                                      ofType:nil];
    auto proto_path   = [[NSBundle mainBundle] pathForResource:@"model/blazepose/pose_detection.tnnproto"
                                                      ofType:nil];
    if (proto_path.length <= 0 || model_path.length <= 0) {
        LOGE("Error: proto or model or anchor path is invalid\n");
        return predictor;
    }

    string proto_content =
        [NSString stringWithContentsOfFile:proto_path encoding:NSUTF8StringEncoding error:nil].UTF8String;
    NSData *data_mode    = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data_mode length] > 0 ? string((const char *)[data_mode bytes], [data_mode length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        LOGE("Error: proto or model path is invalid\n");
        return predictor;
    }

    auto option = std::make_shared<BlazePoseDetectorOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;

        option->min_score_threshold = 0.5;
        option->min_suppression_threshold = 0.3;
    }

    predictor = std::make_shared<BlazePoseDetector>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        LOGE("Error: %s\n", status.description().c_str());
        return nullptr;
    }

    return predictor;
}

- (std::shared_ptr<BlazePoseLandmark>) loadPoseLandmark:(TNNComputeUnits)units {
    std::shared_ptr<BlazePoseLandmark> predictor = nullptr;

    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path   = [[NSBundle mainBundle] pathForResource:@"model/blazepose/pose_landmark_upper_body.tnnmodel"
                                                      ofType:nil];
    auto proto_path   = [[NSBundle mainBundle] pathForResource:@"model/blazepose/pose_landmark_upper_body.tnnproto"
                                                      ofType:nil];
    if (proto_path.length <= 0 || model_path.length <= 0) {
        LOGE("Error: proto or model or anchor path is invalid\n");
        return predictor;
    }

    string proto_content =
        [NSString stringWithContentsOfFile:proto_path encoding:NSUTF8StringEncoding error:nil].UTF8String;
    NSData *data_mode    = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data_mode length] > 0 ? string((const char *)[data_mode bytes], [data_mode length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        LOGE("Error: proto or model path is invalid\n");
        return predictor;
    }

    auto option = std::make_shared<BlazePoseLandmarkOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;

        option->pose_presence_threshold = 0.5;
    }

    predictor = std::make_shared<BlazePoseLandmark>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        LOGE("Error: %s\n", status.description().c_str());
        return nullptr;
    }

    return predictor;
}

-(Status)loadNeuralNetworkModel:(TNNComputeUnits)units {
    Status status = TNN_OK;
    auto pose_detector = [self loadPoseDetector:units];
    auto pose_landmark = [self loadPoseLandmark:units];

    if (!pose_detector) {
        return Status(TNNERR_MODEL_ERR, "loadPoseDetector failed: pls make sure the pose detect model is downloaded");
    }

    if (!pose_landmark) {
        return Status(TNNERR_MODEL_ERR, "loadPoseLandmark failed: pls make sure the pose landmark model is downloaded");
    }
    
    
    auto predictor = std::make_shared<PoseDetectLandmark>();
    status = predictor->Init({pose_detector, pose_landmark});
    
    self.predictor = predictor;

    return status;
}


-(std::vector<std::shared_ptr<ObjectInfo> >)getObjectList:(std::shared_ptr<TNNSDKOutput>)sdk_output {
    std::vector<std::shared_ptr<ObjectInfo> > body_list;
    if (sdk_output && dynamic_cast<BlazePoseLandmarkOutput *>(sdk_output.get())) {
        auto body_output = dynamic_cast<BlazePoseLandmarkOutput *>(sdk_output.get());
        for (auto item : body_output->body_list) {
            auto body = std::make_shared<BlazePoseInfo>();
            for(const auto& kp3d: item.key_points_3d) {
                item.key_points.push_back(std::make_pair(std::get<0>(kp3d), std::get<1>(kp3d)));
            }
            *body = item;
            body_list.push_back(body);
        }
    }
    return body_list;
}

-(NSString*)labelForObject:(std::shared_ptr<ObjectInfo>)object {
    if (object) {
        return [NSString stringWithUTF8String:"body"];
    }
    return nil;
}

@end

