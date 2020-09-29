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

#import "TNNHandDetectTrackerViewModel.h"
#import "hand_detect_tracker.h"
#import "hand_detector.h"
#import "hand_tracker.h"
#import "UIImage+Utility.h"

#import <Metal/Metal.h>
#import <memory>

using namespace std;

@implementation TNNHandDetectTrackerViewModel

- (std::shared_ptr<HandDetector>) loadHandDetector:(TNNComputeUnits)units {
    std::shared_ptr<HandDetector> predictor = nullptr;

    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/hand_detector/detector.opt.tnnmodel"
                                                      ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/hand_detector/detector.opt.tnnproto"
                                                      ofType:nil];
    if (proto_path.length <= 0 || model_path.length <= 0) {
        LOGE("Error: proto or model path is invalid\n");
        return predictor;
    }

    NSString *protoFormat = [NSString stringWithContentsOfFile:proto_path
                                                   encoding:NSUTF8StringEncoding
                                                      error:nil];
    string proto_content = protoFormat.UTF8String;
    NSData *data_mode    = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data_mode length] > 0 ? string((const char *)[data_mode bytes], [data_mode length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        LOGE("Error: proto or model path is invalid\n");
        return predictor;
    }

    auto option = std::make_shared<HandDetectorOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;

        option->conf_threshold = 0.5;
        option->nms_threshold = 0.4;
    }

    predictor = std::make_shared<HandDetector>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        LOGE("Error: %s\n", status.description().c_str());
        return nullptr;
    }

    return predictor;
}

- (std::shared_ptr<HandTracking>) loadHandTracker:(TNNComputeUnits)units {
    std::shared_ptr<HandTracking> predictor = nullptr;

    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    NSString *model_path = nil;
    NSString *proto_path = nil;

    model_path = [[NSBundle mainBundle] pathForResource:@"model/hand_tracking/hand_tracking.opt.tnnmodel"
                                                 ofType:nil];
    proto_path = [[NSBundle mainBundle] pathForResource:@"model/hand_tracking/hand_tracking.opt.tnnproto"
                                                 ofType:nil];

    if (proto_path.length <= 0 || model_path.length <= 0) {
        LOGE("Error: proto or model path is invalid\n");
        return predictor;
    }

    NSString *protoFormat = [NSString stringWithContentsOfFile:proto_path
                                                   encoding:NSUTF8StringEncoding
                                                      error:nil];
    string proto_content = protoFormat.UTF8String;
    NSData *data_mode    = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data_mode length] > 0 ? string((const char *)[data_mode bytes], [data_mode length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        LOGE("Error: proto or model path is invalid\n");
        return predictor;
    }

    auto option = std::make_shared<HandTrackingOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;

        option->hand_presence_threshold = 0.4;
    }

    predictor = std::make_shared<HandTracking>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        LOGE("Error: %s\n", status.description().c_str());
        return nullptr;
    }

    return predictor;
}

-(Status)loadNeuralNetworkModel:(TNNComputeUnits)units {
    Status status = TNN_OK;
    auto hand_detector = [self loadHandDetector:units];
    auto hand_tracker  = [self loadHandTracker:units];

    if (!hand_detector) {
        return Status(TNNERR_MODEL_ERR, "loadHandDetector failed: pls make sure the hand detect model is downloaded");
    }

    if (!hand_tracker) {
        return Status(TNNERR_MODEL_ERR, "loadHandTracker failed: pls make sure the hand tracking model is downloaded");
    }

    auto predictor = std::make_shared<HandDetectTracker>();
    status = predictor->Init({hand_detector, hand_tracker});

    self.predictor = predictor;

    return status;
}

-(std::vector<std::shared_ptr<ObjectInfo> >)getObjectList:(std::shared_ptr<TNNSDKOutput>)sdk_output {
    std::vector<std::shared_ptr<ObjectInfo> > hands;
    if (sdk_output && dynamic_cast<HandTrackingOutput *>(sdk_output.get())) {
        auto hand_output = dynamic_cast<HandTrackingOutput *>(sdk_output.get());

        auto hand = std::make_shared<ObjectInfo>(hand_output->hand_list[0]);
        hands.push_back(hand);
    }
    return hands;
}

-(NSString*)labelForObject:(std::shared_ptr<ObjectInfo>)object {
    return nil;
}

@end


