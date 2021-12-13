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

#import "TNNImageColourController.h"
#import <Metal/Metal.h>
#include <fstream>
#include <iostream>
#import <tnn/tnn.h>

#import "face_gray_transfer.h"
#import "UIImage+Utility.h"

using namespace std;
using namespace TNN_NS;

@interface TNNImageColourController ()
@property(nonatomic, weak) IBOutlet UIButton *btnTNNExamples;
@property(nonatomic, weak) IBOutlet UIImageView *imageView;
@property(nonatomic, weak) IBOutlet UILabel *labelResult;
@property(nonatomic, strong) UIImage *image_orig;
@property(nonatomic, strong) UIImage *image_color;
@end

@implementation TNNImageColourController

- (void)viewDidLoad {
    [super viewDidLoad];

}

- (void)viewWillAppear:(BOOL)animated
{
    [super viewWillAppear:animated];
    self.image_orig      = [UIImage imageWithContentsOfFile:[[NSBundle mainBundle] pathForResource:@"gray_00001.jpg"
                                                                                       ofType:nil]];
    self.imageView.image = self.image_orig;

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
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/gray_transfer/G_8_GRAY2RGB_256.tnnmodel" ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/gray_transfer/G_8_GRAY2RGB_256.tnnproto" ofType:nil];
    
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

    const int target_height = 256;
    const int target_width  = 256;
    DimsVector target_dims = {1, 3, target_height, target_width};

    auto image_data = utility::UIImageGetData(self.image_orig, target_height, target_width);

    auto units = [self getComputeUnitsForIndex:self.switchDevice.selectedSegmentIndex];
    auto option = std::make_shared<TNNSDKOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = path_library.UTF8String;
        option->compute_units = units;
        option->cache_path = NSTemporaryDirectory().UTF8String;
    }
    
    auto predictor = std::make_shared<FaceGrayTransfer>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithFormat:@"%s", status.description().c_str()];
        NSLog(@"Error: %s", status.description().c_str());
        return;
    }
    
    BenchOption bench_option;
    bench_option.forward_count = 1;
    predictor->SetBenchOption(bench_option);
    
    std::shared_ptr<TNNSDKOutput> sdk_output = nullptr;
    auto actual_units = predictor->GetComputeUnits();
    if (actual_units == TNNComputeUnitsGPU) {
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
        status = predictor->Predict(std::make_shared<TNNSDKInput>(image_mat), sdk_output);
    } else if (actual_units == TNNComputeUnitsCPU || actual_units == TNNComputeUnitsAppleNPU) {
        auto image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_ARM, TNN_NS::N8UC4, target_dims, image_data.get());
        status = predictor->Predict(std::make_shared<TNNSDKInput>(image_mat), sdk_output);
    }
    if (status != TNN_OK) {
        NSLog(@"Error: %s", status.description().c_str());
        return;
    }
    
    std::shared_ptr<TNN_NS::Mat> output_mat = nullptr;
    if (sdk_output) {
        output_mat = sdk_output->GetMat();
    }
    
    const int output_height = output_mat->GetHeight();
    const int output_width = output_mat->GetWidth();
    const float *output_data_0 = (const float *)output_mat->GetData();
    const float *output_data_1 = output_data_0 + output_height*output_width;
    const float *output_data_2 = output_data_1 + output_height*output_width;
    
    auto output_data_rgba = new RGBA[output_height*output_width];
    
    for (int i=0; i<output_height*output_width; i++) {
        auto r = (unsigned char)(output_data_0[i]*255.f/2.f + 255.f/2.f);
        auto g = (unsigned char)(output_data_1[i]*255.f/2.f + 255.f/2.f);
        auto b = (unsigned char)(output_data_2[i]*255.f/2.f + 255.f/2.f);
        output_data_rgba[i] = {r, g , b, 255};
    }
    
    auto output_image = utility::UIImageWithDataRGBA(output_data_rgba, output_height, output_width);
    self.imageView.image = output_image;
    delete [] output_data_rgba;

    auto bench_result   = predictor->GetBenchResult();
    self.labelResult.text = [NSString stringWithFormat:@"device: %@\ntime:\n%s",
                             [self getNSSTringForComputeUnits:actual_units],
                             bench_result.Description().c_str()];
}

@end
