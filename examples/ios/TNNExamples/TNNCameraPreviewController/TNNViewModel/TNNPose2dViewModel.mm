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

#import "TNNPose2dViewModel.h"
#import "pose2d_detector.h"

#import <array>

#define ENABLE_GAUSSIAN_BLUR 0

using namespace std;

@interface TNNPose2dViewModel ()
@property (nonatomic, strong) UILabel *label;
@property (nonatomic, strong) NSArray<UIButton *> *modelButtons;
@property (nonatomic, strong) NSArray<UILabel *> *modelLabels;
@property (nonatomic, assign) NSInteger activeModel;

@property (nonatomic, assign) std::array<std::shared_ptr<TNNSDKSample>, 3> predictors;
@property (nonatomic, assign) std::shared_ptr<TNNSDKSample> bigPredictor;
@property (nonatomic, assign) std::shared_ptr<TNNSDKSample> middlePredictor;
@property (nonatomic, assign) std::shared_ptr<TNNSDKSample> smallPredictor;
@end

@implementation TNNPose2dViewModel

-(Status)loadNeuralNetworkModel:(TNNComputeUnits)units {
    Status status = TNN_OK;
    
    // check release mode at Product->Scheme when running
    //运行时请在Product->Scheme中确认意见调整到release模式

    // Get metallib path from app bundle
    // PS：A script(Build Phases -> Run Script) is added to copy the metallib
    // file from tnn framework project to TNNExamples app
    //注意：此工程添加了脚本将tnn工程生成的tnn.metallib自动复制到app内
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];

    // new efficient model
#if ENABLE_GAUSSIAN_BLUR
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/weishi_2dpose/middle_wb2.tnnproto"
                                                      ofType:nil];
    auto middle_proto_path = [[NSBundle mainBundle] pathForResource:@"model/weishi_2dpose/middle.tnnproto"
                                                             ofType:nil];
    auto small_proto_path = [[NSBundle mainBundle] pathForResource:@"model/weishi_2dpose/small_wb2.tnnproto"
                                                            ofType:nil];

    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/weishi_2dpose/pose2d_wb.tnnmodel"
                                                      ofType:nil];
#else
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/weishi_2dpose/big.tnnproto"
                                                      ofType:nil];
    auto middle_proto_path = [[NSBundle mainBundle] pathForResource:@"model/weishi_2dpose/middle.tnnproto"
                                                             ofType:nil];
    auto small_proto_path = [[NSBundle mainBundle] pathForResource:@"model/weishi_2dpose/small.tnnproto"
                                                            ofType:nil];

    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/weishi_2dpose/pose2d.tnnmodel"
                                                      ofType:nil];
#endif

    if (proto_path.length <= 0 || middle_proto_path.length <= 0 || small_proto_path.length <= 0) {
        status = Status(TNNERR_NET_ERR, "Error: proto path is invalid");
        NSLog(@"Error: proto path is invalid");
        return status;
    }
    if (model_path.length <= 0) {
        status = Status(TNNERR_NET_ERR, "Error: model path is invalid");
        NSLog(@"Error: model path is invalid");
        return status;
    }

    NSString *protoFormat = [NSString stringWithContentsOfFile:proto_path
                                                   encoding:NSUTF8StringEncoding
                                                      error:nil];
    NSString *middleProtoFormat = [NSString stringWithContentsOfFile:middle_proto_path
                                                          encoding:NSUTF8StringEncoding
                                                             error:nil];
    NSString *smallProtoFormat = [NSString stringWithContentsOfFile:small_proto_path
                                                          encoding:NSUTF8StringEncoding
                                                             error:nil];
    string proto_content = protoFormat.UTF8String;
    string middle_proto_content = middleProtoFormat.UTF8String;
    string small_proto_content = smallProtoFormat.UTF8String;
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
    
    auto bigPredictor = std::make_shared<Pose2dDetector>();
    status = bigPredictor->Init(option);
    RETURN_ON_NEQ(status, TNN_OK);
    
    auto middleOption = std::make_shared<SkeletonDetectorOption>();
    {
        middleOption->proto_content = middle_proto_content;
        middleOption->model_content = model_content;
        middleOption->library_path = library_path.UTF8String;
        middleOption->compute_units = units;

        middleOption->min_threshold = 0.15f;
    }
    auto middlePredictor = std::make_shared<Pose2dDetector>();
    status = middlePredictor->Init(middleOption);
    RETURN_ON_NEQ(status, TNN_OK);

    auto smallOption = std::make_shared<SkeletonDetectorOption>();
    {
        smallOption->proto_content = small_proto_content;
        smallOption->model_content = model_content;
        smallOption->library_path = library_path.UTF8String;
        smallOption->compute_units = units;

        smallOption->min_threshold = 0.15f;
    }
    auto smallPredictor = std::make_shared<Pose2dDetector>();
    status = smallPredictor->Init(smallOption);
    
    BenchOption bench_option;
    bench_option.forward_count = 1;
    bigPredictor->SetBenchOption(bench_option);
    middlePredictor->SetBenchOption(bench_option);
    smallPredictor->SetBenchOption(bench_option);

    self.bigPredictor = bigPredictor;
    self.middlePredictor = middlePredictor;
    self.smallPredictor = smallPredictor;

    //考虑多线程安全，最好初始化完全没问题后再赋值给成员变量
    //for muti-thread safety, copy to member variable after allocate
    self.predictors = {bigPredictor, middlePredictor, smallPredictor};
    self.predictor = self.predictors[self.activeModel];

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
    viewLayoutHeight.constant = 75;
    
    // initialize buttons
    auto label = [[UILabel alloc] initWithFrame:CGRectMake(0, 10, 80, viewLayoutHeight.constant)];
    label.font = [UIFont systemFontOfSize:14];
    label.text = @"选择模式：";
    label.textColor = [UIColor whiteColor];
    [view addSubview:label];
    
    const int modeCnt = 3;
    std::array<std::string, modeCnt> modeNames = {"full", "balance", "lite"};
    
    auto modeButtons = [NSMutableArray new];
    auto modeLabels = [NSMutableArray new];
    for(int i=0; i<modeCnt; ++i) {
        // set button
        NSString* imagePath = [[NSBundle mainBundle] pathForResource:[NSString stringWithUTF8String:modeNames[i].c_str()]
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
        label.text = [NSString stringWithUTF8String:modeNames[i].c_str()];
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
        self.predictor = self.predictors[button.tag];
        self.activeModel = button.tag;
    }
    
}

@end


