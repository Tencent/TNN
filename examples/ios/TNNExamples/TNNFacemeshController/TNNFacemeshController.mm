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

#import "TNNFacemeshController.h"
#import <Metal/Metal.h>
#import <cstdlib>
#import <sstream>
#import <string>
#import <tuple>
#import <tnn/tnn.h>

#import "BlazeFaceDetector.h"
#import "Facemesh.h"
#import "UIImage+Utility.h"

using namespace std;
using namespace TNN_NS;

@interface TNNFacemeshController ()

@property (weak, nonatomic) IBOutlet UIButton *btnExample;
@property (weak, nonatomic) IBOutlet UILabel *labelResult;
@property (weak, nonatomic) IBOutlet UILabel *labelGPU;
@property (weak, nonatomic) IBOutlet UISwitch *switchGPU;
@property (weak, nonatomic) IBOutlet UIImageView *imageView;


@property(nonatomic, strong) UIImage* image_orig;

@end

@implementation TNNFacemeshController
;

- (void)viewDidLoad {
    [super viewDidLoad];
}

- (void)viewWillAppear:(BOOL) animated {
    [super viewWillAppear:animated];
    //TODO: test image
    self.image_orig = [UIImage imageWithContentsOfFile:[[NSBundle mainBundle] pathForResource:@"test_facemesh.jpg" ofType:nil]];
    
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

- (std::shared_ptr<BlazeFaceDetector>)loadBalzeFace {
    std::shared_ptr<BlazeFaceDetector> predictor = nullptr;
    
    // check release mode at Product->Scheme when running
    //运行时请在Product->Scheme中确认意见调整到release模式
    
    // Get metallib path from app bundle
    // PS：A script(Build Phases -> Run Script) is added to copy the metallib
    // file from tnn framework project to TNNExamples app
    //注意：此工程添加了脚本将tnn工程生成的tnn.metallib自动复制到app内
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/blazeface/blazeface.tnnmodel"
                                                          ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/blazeface/blazeface.tnnproto"
                                                          ofType:nil];
    if (proto_path.length <= 0 || model_path.length <= 0) {
        self.labelResult.text = @"proto or model path is invalid";
        NSLog(@"Error: proto or model path is invalid");
        return predictor;
    }

    string proto_content =
        [NSString stringWithContentsOfFile:proto_path encoding:NSUTF8StringEncoding error:nil].UTF8String;
    NSData *data_mode    = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data_mode length] > 0 ? string((const char *)[data_mode bytes], [data_mode length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        self.labelResult.text = @"proto or model path is invalid";
        NSLog(@"Error: proto or model path is invalid");
        return predictor;
    }
    //blazeface requires input with shape 128*128
    const int target_height = 128;
    const int target_width  = 128;
    DimsVector target_dims  = {1, 3, target_height, target_width};

    auto image_data = utility::UIImageGetData(self.image_orig, target_height, target_width);

    TNNComputeUnits units = self.switchGPU.isOn ? TNNComputeUnitsGPU : TNNComputeUnitsCPU;
    auto option = std::make_shared<BlazeFaceDetectorOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;

        //min_score_thresh
        option->min_score_threshold = 0.75;
        //min_suppression_thresh
        option->min_suppression_threshold = 0.3;
        //predefined anchor file path
        option->anchor_path = string([[[NSBundle mainBundle] pathForResource:@"blazeface_anchors.txt" ofType:nil] UTF8String]);
    }
        
    predictor = std::make_shared<BlazeFaceDetector>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithFormat:@"%s", status.description().c_str()];
        NSLog(@"Error: %s", status.description().c_str());
        return nullptr;
    }
    
    return predictor;
}

- (std::shared_ptr<Facemesh>)loadFaceMesh {
    std::shared_ptr<Facemesh> predictor = nullptr;
    
    // check release mode at Product->Scheme when running
    //运行时请在Product->Scheme中确认意见调整到release模式
    
    // Get metallib path from app bundle
    // PS：A script(Build Phases -> Run Script) is added to copy the metallib
    // file from tnn framework project to TNNExamples app
    //注意：此工程添加了脚本将tnn工程生成的tnn.metallib自动复制到app内
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    auto model_path = [[NSBundle mainBundle] pathForResource:@"model/face_mesh/face_mesh.tnnmodel"
                                                          ofType:nil];
    auto proto_path = [[NSBundle mainBundle] pathForResource:@"model/face_mesh/face_mesh.tnnproto"
                                                          ofType:nil];
    if (proto_path.length <= 0 || model_path.length <= 0) {
        self.labelResult.text = @"proto or model path is invalid";
        NSLog(@"Error: proto or model path is invalid");
        return predictor;
    }

    string proto_content =
        [NSString stringWithContentsOfFile:proto_path encoding:NSUTF8StringEncoding error:nil].UTF8String;
    NSData *data_mode    = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data_mode length] > 0 ? string((const char *)[data_mode bytes], [data_mode length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <= 0) {
        self.labelResult.text = @"proto or model path is invalid";
        NSLog(@"Error: proto or model path is invalid");
        return predictor;
    }

    TNNComputeUnits units = self.switchGPU.isOn ? TNNComputeUnitsGPU : TNNComputeUnitsCPU;
    auto option = std::make_shared<FacemeshOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;

        //TODO: set parameters
        option->face_presence_threshold = 0.1;
        option->flip_vertically = false;
        option->flip_horizontally = false;
        option->norm_z = 1.0f;
        option->ignore_rotation = false;
    }
        
    predictor = std::make_shared<Facemesh>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithFormat:@"%s", status.description().c_str()];
        NSLog(@"Error: %s", status.description().c_str());
        return nullptr;
    }

    return predictor;
}

- (IBAction)onBtnTNNExamples:(id)sender {
    Status status = TNN_OK;
    
    auto predictor_face_detector = [self loadBalzeFace];
    DimsVector target_face_detector_dims = predictor_face_detector->GetInputShape();
    
    auto predictor_face_mesh = [self loadFaceMesh];
    DimsVector target_face_mesh_dims = predictor_face_mesh->GetInputShape();

    auto units = self.switchGPU.isOn ? TNNComputeUnitsGPU : TNNComputeUnitsCPU;

    const int image_orig_height = (int)CGImageGetHeight(self.image_orig.CGImage);
    const int image_orig_width  = (int)CGImageGetWidth(self.image_orig.CGImage);
    DimsVector image_dims = {1, 3, image_orig_height, image_orig_width};
    
    auto image_data_for_detector = utility::UIImageGetData(self.image_orig);
    std::shared_ptr<TNN_NS::Mat> image_mat = nullptr;
    if (units == TNNComputeUnitsCPU) {
        image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_ARM, TNN_NS::N8UC4, image_dims, image_data_for_detector.get());
    } else {
        image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_METAL, TNN_NS::N8UC4, image_dims);

        id<MTLTexture> texture_rgba = (__bridge id<MTLTexture>)image_mat->GetData();
        if (!texture_rgba) {
            self.labelResult.text = @"Error texture input rgba is nil";
            NSLog(@"Error texture input rgba is nil");
            return;
        }

        [texture_rgba replaceRegion:MTLRegionMake2D(0, 0, image_orig_width, image_orig_height)
                        mipmapLevel:0
                          withBytes:image_data_for_detector.get()
                        bytesPerRow:image_orig_width * 4];
    }

    //face detect
    std::vector<BlazeFaceInfo> face_info;
    {
        auto input_mat = std::make_shared<TNN_NS::Mat>(image_mat->GetDeviceType(), TNN_NS::N8UC4, target_face_detector_dims);
        status = predictor_face_detector->Resize(image_mat, input_mat, TNNInterpLinear);
        if (status != TNN_OK) {
            self.labelResult.text = [NSString stringWithUTF8String:status.description().c_str()];
            NSLog(@"Error: %s", status.description().c_str());
                return;
        }

        std::shared_ptr<TNNSDKOutput> sdk_output = nullptr;
        status = predictor_face_detector->Predict(std::make_shared<BlazeFaceDetectorInput>(input_mat), sdk_output);

        if (status != TNN_OK) {
            self.labelResult.text = [NSString stringWithUTF8String:status.description().c_str()];
            NSLog(@"Error: %s", status.description().c_str());
                return;
        }
        
        if (sdk_output && dynamic_cast<BlazeFaceDetectorOutput *>(sdk_output.get()))
        {
            auto face_output = dynamic_cast<BlazeFaceDetectorOutput *>(sdk_output.get());
            face_info = face_output->face_list;
        }
        
        auto bench_result     = predictor_face_detector->GetBenchResult();
        self.labelResult.text = [NSString stringWithFormat:@"device: %@      face count:%d\ntime:\n%s", units == TNNComputeUnitsGPU ? @"gpu" : @"arm", (int)face_info.size(), bench_result.Description().c_str()];
    }

    //face mesh
    {
        for (auto face : face_info) {
            auto face_orig = face.AdjustToViewSize(image_orig_height, image_orig_width, 2);
            //1.5*crop
            int crop_h = face_orig.y2 - face_orig.y1;
            int crop_w = face_orig.x2 - face_orig.x1;
            auto crop_rect = CGRectMake(face_orig.x1-0.25*crop_w,
                                        face_orig.y1-0.25*crop_h,
                                        1.5*crop_w,
                                        1.5*crop_h);
            
            
            //TODO: how does utility handles double crop size?
            DimsVector crop_dims = {1, 3, static_cast<int>(crop_rect.size.height), static_cast<int>(crop_rect.size.width)};
            std::shared_ptr<TNN_NS::Mat> croped_mat = std::make_shared<TNN_NS::Mat>(image_mat->GetDeviceType(), TNN_NS::N8UC4, crop_dims);
            status = predictor_face_mesh->Crop(image_mat, croped_mat, crop_rect.origin.x, crop_rect.origin.y);
            if (status != TNN_OK) {
                self.labelResult.text = [NSString stringWithUTF8String:status.description().c_str()];
                NSLog(@"Error: %s", status.description().c_str());
                    return;
            }
            std::shared_ptr<TNN_NS::Mat> input_mat = std::make_shared<TNN_NS::Mat>(image_mat->GetDeviceType(), TNN_NS::N8UC4, target_face_mesh_dims);
            status = predictor_face_mesh->Resize(croped_mat, input_mat, TNNInterpLinear);
            if (status != TNN_OK) {
                self.labelResult.text = [NSString stringWithUTF8String:status.description().c_str()];
                NSLog(@"Error: %s", status.description().c_str());
                    return;
            }

            std::shared_ptr<TNNSDKOutput> sdk_output = nullptr;
            status = predictor_face_mesh->Predict(std::make_shared<FacemeshInput>(input_mat), sdk_output);

            if (status != TNN_OK) {
                self.labelResult.text = [NSString stringWithUTF8String:status.description().c_str()];
                NSLog(@"Error: %s", status.description().c_str());
                    return;
            }
        
            std::vector<FacemeshInfo> face_mesh_info;
            if (sdk_output && dynamic_cast<FacemeshOutput *>(sdk_output.get()))
            {
                auto face_output = dynamic_cast<FacemeshOutput *>(sdk_output.get());
                face_mesh_info = face_output->face_list;
            }
            
            auto image_orig_data  = utility::UIImageGetData(self.image_orig, image_orig_height, image_orig_width);
            Rectangle((void *)image_orig_data.get(), image_orig_height, image_orig_width,
                      crop_rect.origin.x,  crop_rect.origin.y,
                      crop_rect.origin.x+crop_rect.size.width,
                      crop_rect.origin.y+crop_rect.size.height);
            
            if (face_mesh_info.size() > 0) {
                auto face_mesh = face_mesh_info[0];
                auto face_mesh_crop = face_mesh.AdjustToViewSize(crop_rect.size.height, crop_rect.size.width, 2);
                face_mesh_crop = face_mesh_crop.AddOffset(crop_rect.origin.x, crop_rect.origin.y);
                //TODO: how to draw 2d points accoring to the 3d landmark
                for(auto& p:face_mesh_crop.key_points_3d) {
                    TNN_NS::Point((void*)image_orig_data.get(), image_orig_height, image_orig_width, std::get<0>(p), std::get<1>(p), std::get<2>(p)*(-7));
                }
            }

            UIImage *output_image =
                utility::UIImageWithDataRGBA((void *)image_orig_data.get(), image_orig_height, image_orig_width);
            self.imageView.image = output_image;
        }
    }

}

@end

