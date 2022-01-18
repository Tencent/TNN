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

- (void)onSwitchChanged:(id)sender {
    self.imageView.image  = self.image_orig;
    self.labelResult.text = nil;
}

- (IBAction)onBtnTNNExamples:(id)sender {
    // check release mode at Product->Scheme when running
    //运行时请在Product->Scheme中确认已经调整到release模式

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

    auto units = [self getComputeUnitsForIndex:self.switchDevice.selectedSegmentIndex];
    auto option = std::make_shared<TNNSDKOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = path_library.UTF8String;
        option->compute_units = units;
        option->cache_path = NSTemporaryDirectory().UTF8String;
    }
    auto predictor = std::make_shared<ImageClassifier>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithFormat:@"%s", status.description().c_str()];
        NSLog(@"Error: %s", status.description().c_str());
        return;
    }
    
    BenchOption bench_option;
    bench_option.forward_count = 20;
    predictor->SetBenchOption(bench_option);
    
    auto image_data = utility::UIImageGetData(self.image_orig);
    //preprocess
    const int origin_height = (int)CGImageGetHeight(self.image_orig.CGImage);
    const int origin_width  = (int)CGImageGetWidth(self.image_orig.CGImage);
    DimsVector image_dims = {1, 3, origin_height, origin_width};
    std::shared_ptr<TNN_NS::Mat> image_mat = nullptr;
    
    auto actual_units = predictor->GetComputeUnits();
    if(actual_units == TNNComputeUnitsCPU || actual_units == TNNComputeUnitsAppleNPU) {
        image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_ARM, TNN_NS::N8UC4, image_dims, image_data.get());
    } else {
        image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_METAL, TNN_NS::N8UC4, image_dims);
        id<MTLTexture> texture_rgba = (__bridge id<MTLTexture>)image_mat->GetData();
        if (!texture_rgba) {
            self.labelResult.text = @"Error texture input rgba is nil";
            NSLog(@"Error texture input rgba is nil");
            return;
        }
        
        [texture_rgba replaceRegion:MTLRegionMake2D(0, 0, image_dims[3], image_dims[2])
                        mipmapLevel:0
                          withBytes:image_data.get()
                        bytesPerRow:image_dims[3] * 4];
    }
    
    auto target_dims = predictor->GetInputShape();
    auto input_mat = std::make_shared<TNN_NS::Mat>(image_mat->GetDeviceType(), TNN_NS::N8UC4, target_dims);
    status = predictor->Resize(image_mat, input_mat, TNNInterpLinear);
    if (status != TNN_OK) {
        NSLog(@"Error: %s", status.description().c_str());
        return;
    }

    std::shared_ptr<TNNSDKOutput> sdk_output = nullptr;
    status = predictor->Predict(std::make_shared<TNNSDKInput>(input_mat), sdk_output);
   
    if (status != TNN_OK) {
        NSLog(@"Error: %s", status.description().c_str());
        return;
    }

    int class_id = -1;
    if (sdk_output && dynamic_cast<ImageClassifierOutput *>(sdk_output.get())) {
        auto classfy_output = dynamic_cast<ImageClassifierOutput *>(sdk_output.get());
        class_id = classfy_output->class_id;
    }
    
    NSString *class_result = @"";
    if (class_id < _allClasses.count) {
        class_result = _allClasses[class_id];
        auto results = [class_result componentsSeparatedByString:@","];
        if (results.count > 0) {
            class_result = results[0];
        }
    }
    
    auto bench_result     = predictor->GetBenchResult();
    self.labelResult.text = [NSString stringWithFormat:@"device: %@\nclass:%@\ntime:\n%s",
                                                       [self getNSSTringForComputeUnits:actual_units],
                                                       class_result, bench_result.Description().c_str()];
}

@end
