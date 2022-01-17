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

#import "TNNBlazefaceDetectorController.h"
#import "blazeface_detector.h"
#import "UIImage+Utility.h"
#import <Metal/Metal.h>
#import <cstdlib>
#import <sstream>
#import <string>
#import <tnn/tnn.h>

using namespace std;
using namespace TNN_NS;

@interface TNNBlazefaceDetectorController ()

@property (weak, nonatomic) IBOutlet UIButton *btnExample;
@property (weak, nonatomic) IBOutlet UILabel *labelResult;
@property (weak, nonatomic) IBOutlet UILabel *labelGPU;
@property (weak, nonatomic) IBOutlet UIImageView *imageView;

@property(nonatomic, strong) UIImage* image_orig;

@end

@implementation TNNBlazefaceDetectorController
;

- (void)viewDidLoad {
    [super viewDidLoad];
}

- (void)viewWillAppear:(BOOL) animated {
    [super viewWillAppear:animated];
    
    self.image_orig = [UIImage imageWithContentsOfFile:[[NSBundle mainBundle] pathForResource:@"test_blazeface.jpg" ofType:nil]];
    
    self.imageView.image = self.image_orig;
    
    auto view = self.labelResult.superview;
    [self.imageView removeFromSuperview];
    [self.labelResult removeFromSuperview];
    int screenWidth      = view.frame.size.width;
    int screenHeight     = view.frame.size.height;
    int width            = self.imageView.frame.size.width;
    int height           = self.imageView.frame.size.height;
    int widthOffset      = (screenWidth - width) / 2;
    self.imageView.frame = CGRectMake(widthOffset, (screenHeight - height) / 10, width, height);
    [view addSubview:self.imageView];
    self.labelResult.frame =
        CGRectMake(self.imageView.frame.origin.x,
                   self.imageView.frame.origin.y + height + 5 - self.labelResult.frame.size.height / 2,
                   self.labelResult.frame.size.width, self.labelResult.frame.size.height);
    [view addSubview:self.labelResult];
    
}
- (IBAction)onSwitchChanged:(id)sender {
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
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/blazeface/blazeface.tnnmodel"
                                                          ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/blazeface/blazeface.tnnproto"
                                                          ofType:nil];
    auto anchor_path = [[NSBundle mainBundle] pathForResource:@"model/blazeface/blazeface_anchors.txt"
                                                          ofType:nil];
    if (proto_path.length <= 0 || model_path.length <= 0 || anchor_path.length <= 0) {
        self.labelResult.text = @"proto or model or anchor path is invalid";
        NSLog(@"Error: proto or model or anchor path is invalid");
        return;
    }

    string proto_content =
        [NSString stringWithContentsOfFile:proto_path encoding:NSUTF8StringEncoding error:nil].UTF8String;
    NSData *data_mode    = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data_mode length] > 0 ? string((const char *)[data_mode bytes], [data_mode length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        self.labelResult.text = @"proto or model path is invalid";
        NSLog(@"Error: proto or model path is invalid");
        return;
    }

    auto image_data = utility::UIImageGetData(self.image_orig, 128, 128, 1);

    auto units = [self getComputeUnitsForIndex:self.switchDevice.selectedSegmentIndex];
    auto option = std::make_shared<BlazeFaceDetectorOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        option->cache_path = NSTemporaryDirectory().UTF8String;

        //min_score_thresh
        option->min_score_threshold = 0.75;
        //min_suppression_thresh
        option->min_suppression_threshold = 0.3;
        //predefined anchor file path
        option->anchor_path = string(anchor_path.UTF8String);
    }
        
    auto predictor = std::make_shared<BlazeFaceDetector>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithFormat:@"%s", status.description().c_str()];
        NSLog(@"Error: %s", status.description().c_str());
            return;
    }

    BenchOption bench_option;
    bench_option.forward_count = 20;
    predictor->SetBenchOption(bench_option);

    DimsVector image_dims = {1, 3, 128, 128};
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
    //preprocess
    auto target_dims = predictor->GetInputShape();
    auto input_mat = std::make_shared<TNN_NS::Mat>(image_mat->GetDeviceType(), TNN_NS::N8UC4, target_dims);
    status = predictor->Resize(image_mat, input_mat, TNNInterpNearest);
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithUTF8String:status.description().c_str()];
        NSLog(@"Error: %s", status.description().c_str());
            return;
    }

    std::shared_ptr<TNNSDKOutput> sdk_output = nullptr;
    status = predictor->Predict(std::make_shared<BlazeFaceDetectorInput>(input_mat), sdk_output);

    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithUTF8String:status.description().c_str()];
        NSLog(@"Error: %s", status.description().c_str());
            return;
    }
    
    std::vector<BlazeFaceInfo> face_info;
    if (sdk_output && dynamic_cast<BlazeFaceDetectorOutput *>(sdk_output.get()))
    {
        auto face_output = dynamic_cast<BlazeFaceDetectorOutput *>(sdk_output.get());
        face_info = face_output->face_list;
    }
    
    auto bench_result     = predictor->GetBenchResult();
    self.labelResult.text = [NSString stringWithFormat:@"device: %@      face count:%d\ntime:\n%s", [self getNSSTringForComputeUnits:actual_units], (int)face_info.size(), bench_result.Description().c_str()];

    const int image_orig_height = (int)CGImageGetHeight(self.image_orig.CGImage);
    const int image_orig_width  = (int)CGImageGetWidth(self.image_orig.CGImage);
    auto image_orig_data        = utility::UIImageGetData(self.image_orig, image_orig_height, image_orig_width);
    for (int i = 0; i < face_info.size(); i++) {
        auto face = face_info[i];
        auto face_orig = face.AdjustToViewSize(image_orig_height, image_orig_width, 2);
        Rectangle((void *)image_orig_data.get(), image_orig_height, image_orig_width, face_orig.x1, face_orig.y1, face_orig.x2,
                      face_orig.y2);
        for(auto& p:face_orig.key_points) {
            TNN_NS::Point((void*)image_orig_data.get(), image_orig_height, image_orig_width, p.first, p.second, 1);
        }
    }
    
    UIImage *output_image =
        utility::UIImageWithDataRGBA((void *)image_orig_data.get(), image_orig_height, image_orig_width);
    self.imageView.image = output_image;
}

@end
