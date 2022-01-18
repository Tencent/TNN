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

@interface TNNPoseDetectLandmarkViewModel ()
@property (nonatomic, strong) UILabel *label;
@property (nonatomic, strong) NSArray<UIButton *> *modelButtons;
@property (nonatomic, strong) NSArray<UILabel *> *modelLabels;
@property (nonatomic, assign) NSInteger activeModel;  // 0: upper_body, 1: full_body
@property (nonatomic, assign) std::array<std::shared_ptr<TNNSDKSample>, 2> landmarkPredictors;

-(Status) switchLandmarkModel:(NSInteger)modelIndex;
@end

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
        option->cache_path = NSTemporaryDirectory().UTF8String;

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

- (std::shared_ptr<BlazePoseLandmark>) loadPoseLandmark:(TNNComputeUnits)units full_body:(bool)full_body {
    std::shared_ptr<BlazePoseLandmark> predictor = nullptr;

    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path   = [[NSBundle mainBundle] pathForResource:@"model/blazepose/pose_landmark_upper_body.tnnmodel"
                                                      ofType:nil];
    auto proto_path   = [[NSBundle mainBundle] pathForResource:@"model/blazepose/pose_landmark_upper_body.tnnproto"
                                                      ofType:nil];
    if (full_body) {
        model_path = [[NSBundle mainBundle] pathForResource:@"model/blazepose/pose_landmark_full_body.tnnmodel"
                                                    ofType:nil];
        proto_path = [[NSBundle mainBundle] pathForResource:@"model/blazepose/pose_landmark_full_body.tnnproto"
                                                    ofType:nil];
    }
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
        option->cache_path = NSTemporaryDirectory().UTF8String;

        option->pose_presence_threshold = 0.5;
        option->landmark_visibility_threshold = 0.1;
        option->full_body = full_body;
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
    RETURN_VALUE_ON_NEQ(!pose_detector,
                        false,
                        Status(TNNERR_MODEL_ERR,
                               "loadPoseDetector failed: pls make sure the pose detect model is downloaded"));
    
    auto pose_landmark_ub = [self loadPoseLandmark:units full_body:false];
    RETURN_VALUE_ON_NEQ(!pose_landmark_ub,
                            false,
                            Status(TNNERR_MODEL_ERR,
                                   "loadPoseLandmark failed: pls make sure the pose landmark model is downloaded"));
    auto pose_landmark_fb = [self loadPoseLandmark:units full_body:true];
    RETURN_VALUE_ON_NEQ(!pose_landmark_fb,
                         false,
                         Status(TNNERR_MODEL_ERR,
                                "loadPoseLandmark failed: pls make sure the pose landmark model is downloaded"));

    self.landmarkPredictors = {pose_landmark_ub, pose_landmark_fb};
    
    auto active_landmarkmodel = self.landmarkPredictors[self.activeModel];
    auto predictor = std::make_shared<PoseDetectLandmark>();
    status = predictor->Init({pose_detector, active_landmarkmodel});
    RETURN_ON_NEQ(status, TNN_OK);
    
    self.predictor = predictor;

    return status;
}

-(Status)switchLandmarkModel:(NSInteger)modelIndex {
    auto activeLandmarkModel = self.landmarkPredictors[modelIndex];
    // update detect_landmark
    auto predictor_cast = dynamic_cast<PoseDetectLandmark *>(self.predictor.get());
    RETURN_VALUE_ON_NEQ(!predictor_cast, false, Status(TNNERR_PARAM_ERR, "invalid sdk!"));

    auto status = predictor_cast->SwitchLandmarkModel(self.landmarkPredictors[modelIndex], modelIndex==1);
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

#pragma mark - UI control
- (void)setupCustomView:(UIView *)view
           layoutHeight:(NSLayoutConstraint *)viewLayoutHeight {
    viewLayoutHeight.constant = 75;

    // initialize buttons
    auto label = [[UILabel alloc] initWithFrame:CGRectMake(0, 10, 80, viewLayoutHeight.constant)];
    label.font = [UIFont systemFontOfSize:14];
    label.text = @"选择模型：";
    label.textColor = [UIColor whiteColor];
    [view addSubview:label];

    const int modelCnt = 2;
    std::array<std::string, modelCnt> assetNames = {"blazepose_upper_body", "blazepose_full_body"};
    std::array<std::string, modelCnt> labelTexts = {"上半身", "全身"};

    auto modeButtons = [NSMutableArray new];
    auto modeLabels = [NSMutableArray new];
    NSString* iconDir = @"assets/";
    for(int i=0; i<modelCnt; ++i) {
        // set button
        NSString *iconName = [NSString stringWithUTF8String:assetNames[i].c_str()];
        NSMutableString *iconPath = [NSMutableString stringWithString:iconDir];
        [iconPath appendString: iconName];
        NSString *imagePath = [[NSBundle mainBundle] pathForResource:iconPath
                                                              ofType:@"png"];
        UIImage *btnImage = [UIImage imageNamed:imagePath];
        UIButton *btn = [UIButton buttonWithType:UIButtonTypeCustom];
        [btn setBackgroundImage:btnImage forState:UIControlStateNormal];
        btn.tag = i;
        [btn addTarget:self action:@selector(onButtonClick:) forControlEvents:UIControlEventTouchUpInside];
        auto btnFrame = CGRectMake(15 + 80 + i*(45 + 20),
                                    12, 45, 45);
        btn.frame = btnFrame;
        btn.selected = i == 0;
        if (btn.selected) {
            [btn setBackgroundColor:[UIColor redColor]];
        }
        [modeButtons addObject:btn];

        // set label
        auto labelFrame = CGRectMake(btnFrame.origin.x, btnFrame.origin.y+50, btnFrame.size.width, 10);
        UILabel *label = [[UILabel alloc] initWithFrame:labelFrame];
        label.font = [UIFont systemFontOfSize:12];
        label.text = [NSString stringWithUTF8String:labelTexts[i].c_str()];
        label.textColor = [UIColor whiteColor];
        label.textAlignment = NSTextAlignmentCenter;
        [modeLabels addObject:label];

        [view addSubview:btn];
        [view addSubview:label];
    }

    self.modelButtons = modeButtons;
    self.modelLabels = modeLabels;
    self.label = label;
    // use big mode for default
    self.activeModel = 0;
}

- (void) onButtonClick:(UIButton *)button {
    auto selected = button.selected;
    if (selected) {
        for(UIButton *item in self.modelButtons) {
            item.selected = NO;
        }
    } else {
        for(UIButton *item in self.modelButtons) {
            item.selected = NO;
            [item setBackgroundColor:[UIColor clearColor]];
        }
        button.selected = YES;
        [button setBackgroundColor:[UIColor redColor]];
    }

    if (button.selected) {
        self.activeModel = button.tag;
        [self switchLandmarkModel:self.activeModel];
    }
}

@end

