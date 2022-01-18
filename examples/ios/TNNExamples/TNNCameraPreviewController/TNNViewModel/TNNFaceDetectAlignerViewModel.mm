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

#import "TNNFaceDetectAlignerViewModel.h"
#import "face_detect_aligner.h"
#import "youtu_face_align.h"
#import "blazeface_detector.h"
#import "UIImage+Utility.h"

#import <Metal/Metal.h>
#import <memory>

using namespace std;

@implementation TNNFaceDetectAlignerViewModel

- (std::shared_ptr<BlazeFaceDetector>) loadFaceDetector:(TNNComputeUnits)units {
    std::shared_ptr<BlazeFaceDetector> predictor = nullptr;
    
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/blazeface/blazeface.tnnmodel"
                                                      ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/blazeface/blazeface.tnnproto"
                                                      ofType:nil];
    auto anchor_path = [[NSBundle mainBundle] pathForResource:@"model/blazeface/blazeface_anchors.txt"
                                                          ofType:nil];
    if (proto_path.length <= 0 || model_path.length <= 0 || anchor_path.length <= 0) {
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
    //blazeface requires input with shape 128*128
    const int target_height = 128;
    const int target_width  = 128;
    DimsVector target_dims  = {1, 3, target_height, target_width};
    
    auto option = std::make_shared<BlazeFaceDetectorOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        option->cache_path = NSTemporaryDirectory().UTF8String;
        
        option->input_width = target_width;
        option->input_height = target_height;
        //min_score_thresh
        option->min_score_threshold = 0.75;
        //min_suppression_thresh
        option->min_suppression_threshold = 0.3;
        //predefined anchor file path
        option->anchor_path = string(anchor_path.UTF8String);
    }
    
    predictor = std::make_shared<BlazeFaceDetector>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        LOGE("Error: %s\n", status.description().c_str());
        return nullptr;
    }
    
    return predictor;
}

- (std::shared_ptr<YoutuFaceAlign>) loadYoutuFaceAlign:(TNNComputeUnits)units : (int) phase {
    std::shared_ptr<YoutuFaceAlign> predictor = nullptr;
    
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    NSString *model_path = nil;
    NSString *proto_path = nil;
    NSString *mean_pts_path = nil;
    
    if(1 == phase) {
        model_path = [[NSBundle mainBundle] pathForResource:@"model/youtu_face_alignment/youtu_face_alignment_phase1.tnnmodel"
                                                     ofType:nil];
        proto_path = [[NSBundle mainBundle] pathForResource:@"model/youtu_face_alignment/youtu_face_alignment_phase1.tnnproto"
                                                     ofType:nil];
        mean_pts_path = [[NSBundle mainBundle] pathForResource:@"model/youtu_face_alignment/youtu_mean_pts_phase1.txt"
                                                     ofType:nil];
    } else if(2 == phase) {
        model_path = [[NSBundle mainBundle] pathForResource:@"model/youtu_face_alignment/youtu_face_alignment_phase2.tnnmodel"
                                                     ofType:nil];
        proto_path = [[NSBundle mainBundle] pathForResource:@"model/youtu_face_alignment/youtu_face_alignment_phase2.tnnproto"
                                                     ofType:nil];
        mean_pts_path = [[NSBundle mainBundle] pathForResource:@"model/youtu_face_alignment/youtu_mean_pts_phase2.txt"
                                                     ofType:nil];
    } else{
        LOGE("Error: facealign model phase is invalid\n");
        return nullptr;
    }
    
    if (proto_path.length <= 0 || model_path.length <= 0) {
        LOGE("Error: proto or model path is invalid\n");
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
    //youtu facealign models require input with shape 128*128
    const int target_height = 128;
    const int target_width  = 128;
    DimsVector target_dims  = {1, 1, target_height, target_width};
    
    auto option = std::make_shared<YoutuFaceAlignOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        option->cache_path = NSTemporaryDirectory().UTF8String;
        
        option->input_width = target_width;
        option->input_height = target_height;
        //face threshold
        option->face_threshold = 0.5;
        option->min_face_size = 20;
        //model phase
        option->phase = phase;
        //net_scale
        option->net_scale = phase == 1? 1.2 : 1.3;
        //mean pts path
        option->mean_pts_path = mean_pts_path ? string(mean_pts_path.UTF8String) : "";
    }
    
    predictor = std::make_shared<YoutuFaceAlign>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        LOGE("Error: %s\n", status.description().c_str());
        return nullptr;
    }
    
    return predictor;
}

-(Status)loadNeuralNetworkModel:(TNNComputeUnits)units {
    Status status = TNN_OK;
    auto face_detector = [self loadFaceDetector:units];
    auto predictor_phase1 = [self loadYoutuFaceAlign:units :1];
    auto predictor_phase2 = [self loadYoutuFaceAlign:units :2];
    
    if (!face_detector) {
        return Status(TNNERR_MODEL_ERR, "loadFaceDetector failed: pls make sure the face detect model is downloaded");
    }
    
    if (!predictor_phase1 || !predictor_phase2) {
        return Status(TNNERR_MODEL_ERR, "loadYoutuFaceAlign failed: pls make sure the face alignment model is downloaded");
    }
    
    
    auto predictor = std::make_shared<FaceDetectAligner>();
    status = predictor->Init({face_detector, predictor_phase1, predictor_phase2});
    
    self.predictor = predictor;
    
    //TODO: we need to set it to false when change camera
    self.prev_face = false;
    
    return status;
}

-(std::vector<std::shared_ptr<ObjectInfo> >)getObjectList:(std::shared_ptr<TNNSDKOutput>)sdk_output {
    std::vector<std::shared_ptr<ObjectInfo> > object_list;
    if (sdk_output && dynamic_cast<YoutuFaceAlignOutput *>(sdk_output.get())) {
        auto face_output = dynamic_cast<YoutuFaceAlignOutput *>(sdk_output.get());
        auto face = std::make_shared<YoutuFaceAlignInfo>();
        *face = face_output->face;
        object_list.push_back(face);
    }
    return object_list;
}

-(NSString*)labelForObject:(std::shared_ptr<ObjectInfo>)object {
    return nil;
}

@end
