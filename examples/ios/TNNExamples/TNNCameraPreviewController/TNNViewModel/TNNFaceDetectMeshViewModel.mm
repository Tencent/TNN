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

#import "TNNFaceDetectMeshViewModel.h"
#import "face_detect_mesh.h"
#import "face_mesh.h"
#import "blazeface_detector.h"
#import "UIImage+Utility.h"

#import <Metal/Metal.h>
#import <memory>

using namespace std;

@implementation TNNFaceDetectMeshViewModel

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
        NSLog(@"Error: proto or model or anchor path is invalid");
        return predictor;
    }

    string proto_content =
    [NSString stringWithContentsOfFile:proto_path encoding:NSUTF8StringEncoding error:nil].UTF8String;
    NSData *data_mode    = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data_mode length] > 0 ? string((const char *)[data_mode bytes], [data_mode length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        NSLog(@"Error: proto or model path is invalid");
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
        NSLog(@"Error: %s", status.description().c_str());
        return nullptr;
    }

    return predictor;
}

- (std::shared_ptr<Facemesh>) loadFacemesh:(TNNComputeUnits)units {
    std::shared_ptr<Facemesh> predictor = nullptr;

    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/face_mesh/face_mesh.tnnmodel"
                                                          ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/face_mesh/face_mesh.tnnproto"
                                                          ofType:nil];

    if (proto_path.length <= 0 || model_path.length <= 0) {
        NSLog(@"Error: proto or model path is invalid");
        return predictor;
    }

    string proto_content =
    [NSString stringWithContentsOfFile:proto_path encoding:NSUTF8StringEncoding error:nil].UTF8String;
    NSData *data_mode    = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data_mode length] > 0 ? string((const char *)[data_mode bytes], [data_mode length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        NSLog(@"Error: proto or model path is invalid");
        return predictor;
    }

    auto option = std::make_shared<FacemeshOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        option->cache_path = NSTemporaryDirectory().UTF8String;

        option->face_presence_threshold = 0.1;
        option->flip_vertically = false;
        option->flip_horizontally = false;
        option->norm_z = 1.0f;
        option->ignore_rotation = false;
    }

    predictor = std::make_shared<Facemesh>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        NSLog(@"Error: %s", status.description().c_str());
        return nullptr;
    }

    return predictor;
}

-(Status)loadNeuralNetworkModel:(TNNComputeUnits)units {
    Status status = TNN_OK;
    auto face_detector = [self loadFaceDetector:units];
    auto face_mesher = [self loadFacemesh:units];

    auto predictor = std::make_shared<FaceDetectMesh>();
    status = predictor->Init({face_detector, face_mesher});

    self.predictor = predictor;

    return status;
}

-(std::vector<std::shared_ptr<ObjectInfo> >)getObjectList:(std::shared_ptr<TNNSDKOutput>)sdk_output {
    std::vector<std::shared_ptr<ObjectInfo> > object_list;
    if (sdk_output && dynamic_cast<FacemeshOutput *>(sdk_output.get())) {
        auto face_output = dynamic_cast<FacemeshOutput *>(sdk_output.get());
        auto face = std::make_shared<ObjectInfo>();
        *face = face_output->face_list[0];
        object_list.push_back(face);
    }
    return object_list;
}

-(NSString*)labelForObject:(std::shared_ptr<ObjectInfo>)object {
    return nil;
}

@end
