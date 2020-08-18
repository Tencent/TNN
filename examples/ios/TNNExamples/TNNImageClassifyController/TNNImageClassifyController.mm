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

#import "TNNImageClassifyController.h"
#import <Metal/Metal.h>
#include <fstream>
#include <iostream>
#import <tnn/tnn.h>

#import "image_classifier.h"
#import "UIImage+Utility.h"

using namespace std;
using namespace TNN_NS;

@interface TNNImageClassifyController ()
@property(nonatomic, weak) IBOutlet UIButton *btnTNNExamples;
@property(nonatomic, weak) IBOutlet UIImageView *imageView;
@property(nonatomic, weak) IBOutlet UILabel *labelResult;
@property(nonatomic, weak) IBOutlet UISwitch *switchGPU;

@property(nonatomic, strong) UIImage *image_orig;

@property(nonatomic, strong) NSArray<NSString *> *allClasses;
@end

@implementation TNNImageClassifyController

- (void)viewDidLoad {
    [super viewDidLoad];

}

- (void)viewWillAppear:(BOOL)animated
{
    [super viewWillAppear:animated];
    self.image_orig      = [UIImage imageWithContentsOfFile:[[NSBundle mainBundle] pathForResource:@"tiger_cat.jpg"
                                                                                       ofType:nil]];
    self.imageView.image = self.image_orig;

    self.allClasses = [self getAllClasses];
    auto view = self.labelResult.superview;
    [self.imageView removeFromSuperview];
    [self.labelResult removeFromSuperview];
    int screenWidth = view.frame.size.width;
    int screenHeight = view.frame.size.height;
    int width = self.imageView.frame.size.width;
    int height = self.imageView.frame.size.height;
    int widthOffset = (screenWidth - width) / 2;
    self.imageView.frame = CGRectMake(widthOffset, (screenHeight - height) / 10, width, height);
    [view addSubview:self.imageView];
    self.labelResult.frame = CGRectMake(self.imageView.frame.origin.x, self.imageView.frame.origin.y + height + 5 - self.labelResult.frame.size.height / 2, self.labelResult.frame.size.width, self.labelResult.frame.size.height);
    [view addSubview:self.labelResult];
}

- (NSArray<NSString *> *)getAllClasses {
    NSMutableArray *classes = [NSMutableArray new];

    auto path_class = [[NSBundle mainBundle] pathForResource:@"synset.txt" ofType:nil];
    ifstream fin(path_class.UTF8String);
    string s;
    while (getline(fin, s)) {
        [classes addObject:[NSString stringWithFormat:@"%s", s.c_str()]];
    }

    return classes;
}

- (IBAction)onSwichChanged:(id)sender {
    self.imageView.image  = self.image_orig;
    self.labelResult.text = nil;
}

- (IBAction)onBtnTNNExamples:(id)sender {
    // check release mode at Product->Scheme when running
    //运行时请在Product->Scheme中确认意见调整到release模式

    // Get metallib path from app bundle
    // PS：A script(Build Phases -> Run Script) is added to copy the metallib
    // file from tnn framework project to TNNExamples app
    //注意：此工程添加了脚本将tnn工程生成的tnn.metallib自动复制到app内
    auto path_library = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
#if TNN_SDK_USE_NCNN_MODEL
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/SqueezeNet/squeezenet_v1.1.bin" ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/SqueezeNet/squeezenet_v1.1.param" ofType:nil];
#else
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/SqueezeNet/squeezenet_v1.1.tnnmodel" ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/SqueezeNet/squeezenet_v1.1.tnnproto" ofType:nil];
#endif
    if (model_path.length <= 0 || proto_path.length <= 0) {
        self.labelResult.text = @"proto or model path is invalid";
        NSLog(@"Error: proto or model path is invalid");
        return;
    }
    string proto_content =
        [NSString stringWithContentsOfFile:proto_path encoding:NSUTF8StringEncoding error:nil].UTF8String;
    NSData *data         = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data length] > 0 ? string((const char *)[data bytes], [data length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        self.labelResult.text = @"proto or model path is invalid";
        NSLog(@"Error: proto or model path is invalid");
        return;
    }

    const int target_height = 224;
    const int target_width  = 224;
    DimsVector target_dims = {1, 3, target_height, target_width};

    auto image_data = utility::UIImageGetData(self.image_orig, target_height, target_width);

    TNNComputeUnits units = self.switchGPU.isOn ? TNNComputeUnitsGPU : TNNComputeUnitsCPU;

    ImageClassifier classifier;
    auto status = classifier.Init(proto_content, model_content, path_library.UTF8String, units);
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithUTF8String:status.description().c_str()];
        NSLog(@"Error: %s", status.description().c_str());
        return;
    }

    BenchOption bench_option;
    bench_option.forward_count = 20;
    classifier.SetBenchOption(bench_option);

    int class_id       = -1;
    auto compute_units = classifier.GetComputeUnits();
    if (compute_units == TNNComputeUnitsGPU) {
        auto image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_METAL, TNN_NS::N8UC4, target_dims);

        id<MTLTexture> texture_rgba = (__bridge id<MTLTexture>)image_mat->GetData();
        if (!texture_rgba) {
            NSLog(@"Error texture input rgba is nil");
            return;
        }

        [texture_rgba replaceRegion:MTLRegionMake2D(0, 0, target_width, target_height)
                        mipmapLevel:0
                          withBytes:image_data.get()
                        bytesPerRow:target_width * 4];
        status = classifier.Classify(image_mat, target_height, target_width, class_id);
    } else if (compute_units == TNNComputeUnitsCPU) {
        auto image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_ARM, TNN_NS::N8UC4, target_dims, image_data.get());
        status = classifier.Classify(image_mat, target_height, target_width, class_id);
    }
    if (status != TNN_OK) {
        NSLog(@"Error: %s", status.description().c_str());
        return;
    }

    string class_result = "";
    if (class_id < _allClasses.count) {
        class_result = _allClasses[class_id].UTF8String;
    }

    auto bench_result     = classifier.GetBenchResult();
    self.labelResult.text = [NSString stringWithFormat:@"device: %@\nclass:%s\ntime:\n%s",
                                                       compute_units == TNNComputeUnitsGPU ? @"gpu" : @"arm",
                                                       class_result.c_str(), bench_result.Description().c_str()];
}

@end
