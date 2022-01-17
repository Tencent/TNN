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

#import "TNNMonoDepthViewModel.h"
#import "monodepth.h"

#import "DYFlatButton.h"
#import "UIColor+Utility.h"

using namespace std;

@interface TNNMonoDepthViewModel ()
@property (nonatomic, assign) vector<RGBA> colors;
@property (nonatomic, strong) NSArray<DYFlatButton *> *colorButtons;
@property (nonatomic, assign) RGBA active_color;
@end

@implementation TNNMonoDepthViewModel

-(Status)loadNeuralNetworkModel:(TNNComputeUnits)units {
    Status status = TNN_OK;
    
    // check release mode at Product->Scheme when running
    //运行时请在Product->Scheme中确认已经调整到release模式
    
    // Get metallib path from app bundle
    // PS：A script(Build Phases -> Run Script) is added to copy the metallib
    // file from tnn framework project to TNNExamples app
    //注意：此工程添加了脚本将tnn工程生成的tnn.metallib自动复制到app内
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/monodepth_pydnet/monodepth_pydnet.tnnmodel"
                                                          ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/monodepth_pydnet/monodepth_pydnet.tnnproto"
                                                          ofType:nil];
    if (proto_path.length <= 0 || model_path.length <= 0) {
        status = Status(TNNERR_NET_ERR, "Error: proto or model path is invalid");
        NSLog(@"Error: proto or model path is invalid");
        return status;
    }

    NSString *protoFormat = [NSString stringWithContentsOfFile:proto_path
    encoding:NSUTF8StringEncoding
       error:nil];
    string proto_content =
        protoFormat.UTF8String;
    NSData *data = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data length] > 0 ? string((const char *)[data bytes], [data length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        status = Status(TNNERR_NET_ERR, "Error: proto or model path is invalid");
        NSLog(@"Error: proto or model path is invalid");
        return status;
    }

    auto option = std::make_shared<MonoDepthOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        option->cache_path = NSTemporaryDirectory().UTF8String;
    }
        
    auto predictor = std::make_shared<MonoDepth>();
    status = predictor->Init(option);
    
    BenchOption bench_option;
    bench_option.forward_count = 1;
    predictor->SetBenchOption(bench_option);
    
    //考虑多线程安全，最好初始化完全没问题后再赋值给成员变量
    //for muti-thread safety, copy to member variable after allocate
    self.predictor = predictor;
    
    // color blue
    [self setHairSegmentationRGBA:self.active_color];
    return status;
}


-(std::vector<std::shared_ptr<ObjectInfo> >)getObjectList:(std::shared_ptr<TNNSDKOutput>)sdk_output {
    return {};
}

-(ImageInfo)getImage:(std::shared_ptr<TNNSDKOutput>)sdk_output {
    //std::shared_ptr<char> image_data = nullptr;
    ImageInfo image;
    if (sdk_output && dynamic_cast<MonoDepthOutput *>(sdk_output.get())) {
        auto output = dynamic_cast<MonoDepthOutput *>(sdk_output.get());
        //auto merged_image = output->merged_image;
        image = output->depth;
    }
    return image;
}

-(NSString*)labelForObject:(std::shared_ptr<ObjectInfo>)object {
    return nil;
}

-(void) setHairSegmentationRGBA:(RGBA) color{
    if (self.predictor) {
//        auto* predictor_cast = dynamic_cast<HairSegmentation *>(self.predictor.get());
//        predictor_cast->SetHairColor(color);
    }
}

#pragma mark - UI control
- (BOOL)showImageAtMinorPreview {
    return YES;
}

- (void)setupCustomView:(UIView *)view
           layoutHeight:(NSLayoutConstraint *)viewLayoutHeight {
    viewLayoutHeight.constant = 0;
}

- (void)adajustStackPrevieView:(UIStackView *)stackView {
    for (UIView *subview in stackView.arrangedSubviews) {
        subview.hidden = NO;
    }
}

@end

