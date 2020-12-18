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

#import "TNNSkeletonDetectorViewModel.h"
#import "skeleton_detector.h"

using namespace std;

@interface TNNSkeletonDetectorViewModel ()
@property (nonatomic, strong) UISwitch *modelSwitch;
@property (nonatomic, strong) UILabel *modelLabel;
@property (nonatomic, assign) bool useThinModel;
@property (nonatomic, assign) std::shared_ptr<TNNSDKSample> bigPredictor;
@property (nonatomic, assign) std::shared_ptr<TNNSDKSample> thinPredictor;
@end

@implementation TNNSkeletonDetectorViewModel

-(Status)loadNeuralNetworkModel:(TNNComputeUnits)units {
    Status status = TNN_OK;
    
    // check release mode at Product->Scheme when running
    //运行时请在Product->Scheme中确认意见调整到release模式

    // Get metallib path from app bundle
    // PS：A script(Build Phases -> Run Script) is added to copy the metallib
    // file from tnn framework project to TNNExamples app
    //注意：此工程添加了脚本将tnn工程生成的tnn.metallib自动复制到app内
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    // fused model, no post-processing required, gaussian blur->2 conv layers
    //auto model_path = [[NSBundle mainBundle] pathForResource:@"model/skeleton/skeleton.tnnmodel"
    //                                                  ofType:nil];
    //auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/skeleton/skeleton.tnnproto"
    //                                                  ofType:nil];
    //auto model_path = [[NSBundle mainBundle] pathForResource:@"model/skeleton/skeleton.tnnmodel"
    //                                                  ofType:nil];
    //auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/skeleton/skeleton_noresize.tnnproto"
    //                                                  ofType:nil];

    // new efficient model
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/weishi_2dpose/pose2d_thin.tnnproto"
                                                      ofType:nil];
    auto thin_proto_path = [[NSBundle mainBundle] pathForResource:@"model/weishi_2dpose/pose2d_thin_noresize.tnnproto"
                                                      ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/weishi_2dpose/pose2d.tnnmodel"
                                                      ofType:nil];

    if (model_path.length <= 0 || proto_path.length <= 0 || thin_proto_path.length <= 0) {
        status = Status(TNNERR_NET_ERR, "Error: proto or model path is invalid");
        NSLog(@"Error: proto or model path is invalid");
        return status;
    }

    NSString *protoFormat = [NSString stringWithContentsOfFile:proto_path
                                                   encoding:NSUTF8StringEncoding
                                                      error:nil];
    NSString *thinProtoFormat = [NSString stringWithContentsOfFile:thin_proto_path
                                                          encoding:NSUTF8StringEncoding
                                                             error:nil];
    string proto_content = protoFormat.UTF8String;
    string thin_proto_content = thinProtoFormat.UTF8String;
    NSData *data = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data length] > 0 ? string((const char *)[data bytes], [data length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <=0) {
        status = Status(TNNERR_NET_ERR, "Error: proto or model path is invalid");
        NSLog(@"Error: proto or model path is invalid");
        return status;
    }
    
    auto option = std::make_shared<SkeletonDetectorOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        
        option->min_threshold = 0.15f;
    }
    
    auto bigPredictor = std::make_shared<SkeletonDetector>();
    status = bigPredictor->Init(option);
    RETURN_ON_NEQ(status, TNN_OK);

    auto thinOption = std::make_shared<SkeletonDetectorOption>();
    {
        thinOption->proto_content = thin_proto_content;
        thinOption->model_content = model_content;
        thinOption->library_path = library_path.UTF8String;
        thinOption->compute_units = units;

        thinOption->min_threshold = 0.15f;
    }
    auto thinPredictor = std::make_shared<SkeletonDetector>();
    status = thinPredictor->Init(thinOption);
    
    BenchOption bench_option;
    bench_option.forward_count = 1;
    bigPredictor->SetBenchOption(bench_option);
    thinPredictor->SetBenchOption(bench_option);

    self.bigPredictor = bigPredictor;
    self.thinPredictor = thinPredictor;

    //考虑多线程安全，最好初始化完全没问题后再赋值给成员变量
    //for muti-thread safety, copy to member variable after allocate
    auto useThinModel = self.useThinModel;
    self.predictor =  useThinModel? thinPredictor : bigPredictor;

    return status;
}


-(std::vector<std::shared_ptr<ObjectInfo> >)getObjectList:(std::shared_ptr<TNNSDKOutput>)sdk_output {
    std::vector<std::shared_ptr<ObjectInfo> > object_list;
    if (sdk_output && dynamic_cast<SkeletonDetectorOutput *>(sdk_output.get())) {
        auto skeleton_output = dynamic_cast<SkeletonDetectorOutput *>(sdk_output.get());
        auto skeleton = std::make_shared<SkeletonInfo>();
        *skeleton = skeleton_output->keypoints;
        object_list.push_back(skeleton);
    }
    return object_list;
}

-(NSString*)labelForObject:(std::shared_ptr<ObjectInfo>)object {
    if (object) {
        return [NSString stringWithFormat:@"%.2f", object->score];
    }
    return nil;
}

#pragma mark - UI control
- (void)setupCustomView:(UIView *)view
           layoutHeight:(NSLayoutConstraint *)viewLayoutHeight {
    viewLayoutHeight.constant = 60;

    auto mySwitch = [[UISwitch alloc] initWithFrame:CGRectMake(55 + 200,
                                                            12, 36, 36)];
    [mySwitch addTarget:self action:@selector(onSwitchModel:) forControlEvents:UIControlEventValueChanged];
    [view addSubview:mySwitch];
    [mySwitch setOn:NO];

    auto myLabel = [[UILabel alloc] initWithFrame:CGRectMake(55 + 200,
                                                            40, 100, 45)];
    [myLabel setText:@"thin model"];
    [view addSubview:myLabel];

    self.modelSwitch = mySwitch;
    self.modelLabel = myLabel;
}

- (void)onSwitchModel:(UISwitch *)switchModel {
    auto useThinModel = switchModel.on;
    self.useThinModel = useThinModel;
    if (useThinModel)
        self.predictor = self.thinPredictor;
    else
        self.predictor = self.bigPredictor;
}

@end

