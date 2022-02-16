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

#import "TNNFaceDetectorViewModel.h"
#import "ultra_face_detector.h"

using namespace std;

const int target_height = 640;
const int target_width = 480;

@implementation TNNFaceDetectorViewModel

-(Status)loadNeuralNetworkModel:(TNNComputeUnits)units {
    Status status = TNN_OK;
    
    // check release mode at Product->Scheme when running
    //运行时请在Product->Scheme中确认已经调整到release模式

    // Get metallib path from app bundle
    // PS：A script(Build Phases -> Run Script) is added to copy the metallib
    // file from tnn framework project to TNNExamples app
    //注意：此工程添加了脚本将tnn工程生成的tnn.metallib自动复制到app内
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib"
    ofType:nil];
#if TNN_SDK_USE_NCNN_MODEL
        auto model_path = [[NSBundle mainBundle] pathForResource:@"model/face_detector/version-slim-320_simplified.bin"
                                                           ofType:nil];
        auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/face_detector/version-slim-320_simplified.param"
                                                           ofType:nil];
#else
        auto model_path = [[NSBundle mainBundle] pathForResource:@"model/face_detector/version-slim-320_simplified.tnnmodel"
                                                           ofType:@""];
        auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/face_detector/version-slim-320_simplified.tnnproto"
                                                           ofType:@""];
#endif
    if (model_path.length <= 0 || proto_path.length <= 0) {
        status = Status(TNNERR_NET_ERR, "Error: proto or model path is invalid");
        NSLog(@"Error: proto or model path is invalid");
        return status;
    }

    NSString *protoFormat = [NSString stringWithContentsOfFile:proto_path
                                                   encoding:NSUTF8StringEncoding
                                                      error:nil];
    string proto_content = protoFormat.UTF8String;
    NSData *data = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data length] > 0 ? string((const char *)[data bytes], [data length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <=0) {
        status = Status(TNNERR_NET_ERR, "Error: proto or model path is invalid");
        NSLog(@"Error: proto or model path is invalid");
        return status;
    }
    
    auto option = std::make_shared<UltraFaceDetectorOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        option->cache_path = NSTemporaryDirectory().UTF8String;
        
        //only one input, ignore the input name
        option->input_shapes = {{"ignore", {1,3,target_height,target_width}}};
        
        option->input_width = target_width;
        option->input_height = target_height;
        option->score_threshold = 0.975;
        option->iou_threshold = 0.23;
        option->topk = 1;
    }
    
    auto predictor = std::make_shared<UltraFaceDetector>();
    status = predictor->Init(option);
    
    BenchOption bench_option;
    bench_option.forward_count = 1;
    predictor->SetBenchOption(bench_option);
    
    //考虑多线程安全，最好初始化完全没问题后再赋值给成员变量
    //for muti-thread safety, copy to member variable after allocate
    self.predictor = predictor;
    return status;
}


-(std::vector<std::shared_ptr<ObjectInfo> >)getObjectList:(std::shared_ptr<TNNSDKOutput>)sdk_output {
    std::vector<std::shared_ptr<ObjectInfo> > object_list;
    if (sdk_output && dynamic_cast<UltraFaceDetectorOutput *>(sdk_output.get())) {
        auto face_output = dynamic_cast<UltraFaceDetectorOutput *>(sdk_output.get());
        for (auto item : face_output->face_list) {
            auto face = std::make_shared<FaceInfo>();
            *face = item;
            object_list.push_back(face);
        }
    }
    return object_list;
}

-(NSString*)labelForObject:(std::shared_ptr<ObjectInfo>)object {
    if (object) {
        return [NSString stringWithFormat:@"%.2f", object->score];
    }
    return nil;
}

@end
