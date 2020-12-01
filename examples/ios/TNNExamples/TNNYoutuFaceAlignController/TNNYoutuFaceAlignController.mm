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

#import "TNNYoutuFaceAlignController.h"
#import "youtu_face_align.h"
#import "blazeface_detector.h"
#import "UIImage+Utility.h"
#import <Metal/Metal.h>
#import <cstdlib>
#import <sstream>
#import <string>
#import <tnn/tnn.h>

using namespace std;
using namespace TNN_NS;

#define BREAK_IF(cond)               \
    if (cond)                        \
        break

@interface TNNYoutuFaceAlignController ()

@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UISwitch *switchGPU;
@property (weak, nonatomic) IBOutlet UILabel *labelGPU;
@property (weak, nonatomic) IBOutlet UILabel *labelResult;
@property (weak, nonatomic) IBOutlet UIButton *btnExamples;

@property(nonatomic, strong) UIImage* image_orig;

@property std::shared_ptr<BlazeFaceDetector> face_detector;
@property std::shared_ptr<YoutuFaceAlign> predictor_phase1;
@property std::shared_ptr<YoutuFaceAlign> predictor_phase2;
@property bool prev_face;

@property NSMutableArray *result;

@end

@implementation TNNYoutuFaceAlignController
;

- (void)viewDidLoad {
    [super viewDidLoad];
}

- (void)viewWillAppear:(BOOL) animated {
    [super viewWillAppear:animated];
    
    // Iterate all images
    self.result = [NSMutableArray array];
    [[[NSBundle mainBundle] pathsForResourcesOfType:@".jpg" inDirectory:@"decoded_images/."] enumerateObjectsUsingBlock:^(NSString *obj, NSUInteger idx, BOOL *stop) {
        NSString *path = [obj lastPathComponent];
        //printf("path:%s\n", std::string([path UTF8String]).c_str());
        if ([path hasSuffix:@"jpg"]) {
            [self.result addObject:obj];
        }
    }];
    // sort according to the name, ensure the images are processed frame by frame
    [self.result sortUsingSelector:@selector(localizedStandardCompare:)];
    self.image_orig = [UIImage imageWithContentsOfFile:self.result[0]];
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

- (std::shared_ptr<BlazeFaceDetector>) loadFaceDetector {
    std::shared_ptr<BlazeFaceDetector> predictor = nullptr;
    
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

    TNNComputeUnits units = self.switchGPU.isOn ? TNNComputeUnitsGPU : TNNComputeUnitsCPU;
    if(units == TNNComputeUnitsCPU) {
        LOGE("load ARM model!\n");
    } else {
        LOGE("load Metal model!\n");
    }
    auto option = std::make_shared<BlazeFaceDetectorOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        
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
        self.labelResult.text = [NSString stringWithFormat:@"%s", status.description().c_str()];
        NSLog(@"Error: %s", status.description().c_str());
        return nullptr;
    }
    
    return predictor;
}

- (std::shared_ptr<YoutuFaceAlign>) loadYoutuFaceAlign: (int) phase {
    std::shared_ptr<YoutuFaceAlign> predictor = nullptr;
    
    auto library_path = [[NSBundle mainBundle] pathForResource:@"tnn.metallib" ofType:nil];
    NSString *model_path = nil;
    NSString *proto_path = nil;
    NSString *mean_pts_path = nil;
    
    if(1 == phase) {
        model_path = [[NSBundle mainBundle] pathForResource:@"model/youtu_face_alignment/youtu_face_alignment_phase1.tnnmodel"
                                                     ofType:nil];
        proto_path = [[NSBundle mainBundle] pathForResource:@"model/youtu_face_alignment/youtu_face_alignment_phase1.tnnproto"
                                                     ofType:nil];
        mean_pts_path = [[NSBundle mainBundle] pathForResource:@"model/youtu_face_alignment/youtu_mean_pts_phase1.txt"
                                                     ofType:nil];
    } else if(2 == phase) {
        model_path = [[NSBundle mainBundle] pathForResource:@"model/youtu_face_alignment/youtu_face_alignment_phase2.tnnmodel"
                                                     ofType:nil];
        proto_path = [[NSBundle mainBundle] pathForResource:@"model/youtu_face_alignment/youtu_face_alignment_phase2.tnnproto"
                                                     ofType:nil];
        mean_pts_path = [[NSBundle mainBundle] pathForResource:@"model/youtu_face_alignment/youtu_mean_pts_phase2.txt"
                                                     ofType:nil];
    } else{
        NSLog(@"Error: facealign model phase is invalid");
        return nullptr;
    }
    
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
    //youtu facealign models require input with shape 128*128
    const int target_height = 128;
    const int target_width  = 128;
    DimsVector target_dims  = {1, 1, target_height, target_width};

    TNNComputeUnits units = self.switchGPU.isOn ? TNNComputeUnitsGPU : TNNComputeUnitsCPU;
    if(units == TNNComputeUnitsCPU) {
        LOGE("load ARM model!\n");
    } else {
        LOGE("load Metal model!\n");
    }
    auto option = std::make_shared<YoutuFaceAlignOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = library_path.UTF8String;
        option->compute_units = units;
        
        option->input_width = target_width;
        option->input_height = target_height;
        //face threshold
        option->face_threshold = 0.5;
        option->min_face_size = 20;
        //model phase
        option->phase = phase;
        //net_scale
        option->net_scale = phase == 1? 1.2 : 1.3;
        //mean pts path
       option->mean_pts_path = mean_pts_path ? string(mean_pts_path.UTF8String) : "";
    }
        
    predictor = std::make_shared<YoutuFaceAlign>();
    auto status = predictor->Init(option);
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithFormat:@"%s", status.description().c_str()];
        NSLog(@"Error: %s", status.description().c_str());
        return nullptr;
    }
    
    return predictor;
}

- (IBAction)onSwitchChanged:(id)sender {
    self.imageView.image  = self.image_orig;
    self.labelResult.text = nil;
}

- (UIImage*) DrawFaceAlignResult:(std::shared_ptr<TNN_NS::Mat>) phase1_pts :
                             (std::shared_ptr<TNN_NS::Mat>) phase2_pts :
                             (UIImage *) input_image :
                             (NSString *) image_path {
    const int image_orig_height = (int)CGImageGetHeight(input_image.CGImage);
    const int image_orig_width  = (int)CGImageGetWidth(input_image.CGImage);
    auto image_orig_data        = utility::UIImageGetData(input_image, image_orig_height, image_orig_width);
    const float scale_x = 1.0;
    const float scale_y = 1.0;

    auto pts_count_phase1 = TNN_NS::DimsVectorUtils::Count(phase1_pts->GetDims()) / 2;
    float* pts1 = static_cast<float*>(phase1_pts->GetData());
    for(int pid=0; pid < pts_count_phase1; ++pid) {
        int x = static_cast<int>(pts1[pid * 2 + 0]);
        int y = static_cast<int>(pts1[pid * 2 + 1]);
        TNN_NS::Point((void*)image_orig_data.get(), image_orig_height, image_orig_width, x, y, 0, scale_x, scale_y);
    }

    auto pts_count_phase2 = TNN_NS::DimsVectorUtils::Count(phase2_pts->GetDims()) / 2;
    float* pts2 = static_cast<float*>(phase2_pts->GetData());
    for(int pid=0; pid < pts_count_phase2; ++pid) {
        int x = static_cast<int>(pts2[pid * 2 + 0]);
        int y = static_cast<int>(pts2[pid * 2 + 1]);
        TNN_NS::Point((void*)image_orig_data.get(), image_orig_height, image_orig_width, x, y, 0, scale_x, scale_y);
    }

    UIImage *output_image = utility::UIImageWithDataRGBA((void *)image_orig_data.get(), image_orig_height, image_orig_width);

#if TARGET_IPHONE_SIMULATOR
    // save image on simulator
    NSString *out_name = [[image_path lastPathComponent] stringByReplacingOccurrencesOfString: @".jpg" withString:@"_out.jpg"];
    // set to destination directory
    const std::string save_dir = "/tmp/";
    std::string save_path = save_dir+string([out_name UTF8String]);
    NSString *path = [NSString stringWithCString:save_path.c_str()
                                        encoding:[NSString defaultCStringEncoding]];
    [UIImageJPEGRepresentation(output_image, 1.0) writeToFile:path atomically:YES];
#else
    // write to album on real device
    UIImageWriteToSavedPhotosAlbum(output_image, nil, nil, nil);
#endif
    return output_image;
}

- (std::shared_ptr<TNN_NS::Mat>) ConvertImageToMat:(TNNComputeUnits) units : (DimsVector) dims : (std::shared_ptr<char>) image_data {
    std::shared_ptr<Mat> image_mat = nullptr;
    if (units == TNNComputeUnitsGPU) {
        image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_METAL, TNN_NS::N8UC4, dims);

        id<MTLTexture> texture_rgba = (__bridge id<MTLTexture>)image_mat->GetData();
        if (!texture_rgba) {
            self.labelResult.text = @"Error texture input rgba is nil";
            NSLog(@"Error texture input rgba is nil");
            return image_mat;
        }
        [texture_rgba replaceRegion:MTLRegionMake2D(0, 0, dims[3], dims[2])
                        mipmapLevel:0
                          withBytes:image_data.get()
                        bytesPerRow:dims[3] * 4];
    } else if (units == TNNComputeUnitsCPU) {
        image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_ARM, TNN_NS::N8UC4, dims, image_data.get());
    }
    return image_mat;
}

- (void) CheckStatusSetLabel:(Status)status {
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithUTF8String:status.description().c_str()];
        NSLog(@"Error: %s", status.description().c_str());
        return;
    }
}

- (IBAction)onBtnTNNExamples:(id)sender {
    //clear result
    self.labelResult.text = nil;
    //load models
    self.face_detector = [self loadFaceDetector];
    self.predictor_phase1 = [self loadYoutuFaceAlign:1];
    self.predictor_phase2 = [self loadYoutuFaceAlign:2];
    self.prev_face = false;
    
    const int image_orig_height = (int)CGImageGetHeight(self.image_orig.CGImage);
    const int image_orig_width  = (int)CGImageGetWidth(self.image_orig.CGImage);
    TNN_NS::DimsVector orig_image_dims = {1, 3, image_orig_height, image_orig_width};
    TNNComputeUnits compute_units = self.switchGPU.isOn ? TNNComputeUnitsGPU : TNNComputeUnitsCPU;

    auto idx = 0;
    Status status = TNN_OK;
    UIImage* last_frame = nil;

    for (NSString * img_path in self.result) {
        // use autoreleasepool to rease images allocated inside each loop right after each iteration completes,
        // otherwise the memory will be released after the complete loop completes and the code will take too much memory.
        @autoreleasepool {
            LOGE("processing image[%d]:%s\n",idx++,  [[img_path lastPathComponent] UTF8String]);
            auto input_image = [UIImage imageWithContentsOfFile:img_path];
            auto image_data = utility::UIImageGetData(input_image);

            std::shared_ptr<TNN_NS::Mat> image_mat = nullptr;
            std::shared_ptr<TNN_NS::Mat> phase1_pts = nullptr;
            //phase1 model
            {
                // 1) prepare input for phase1 model
                if(!self.prev_face) {
                    // i) get face from detector
                    std::shared_ptr<TNNSDKOutput> sdk_output = nullptr;
                    image_mat = [self ConvertImageToMat: compute_units :orig_image_dims :image_data];
                    status = self.face_detector->Predict(std::make_shared<BlazeFaceDetectorInput>(image_mat), sdk_output);
                    [self CheckStatusSetLabel:status];
                    BREAK_IF(status != TNN_OK);

                    std::vector<BlazeFaceInfo> face_info;
                    if (sdk_output && dynamic_cast<BlazeFaceDetectorOutput *>(sdk_output.get()))
                    {
                        auto face_output = dynamic_cast<BlazeFaceDetectorOutput *>(sdk_output.get());
                        face_info = face_output->face_list;
                    }
                    if(face_info.size() <= 0) {
                        LOGE("Error no faces found!");
                        continue;
                    }
                    auto face = face_info[0];
                    // scale the face point according to the original image size
                    auto face_orig = face.AdjustToViewSize(image_orig_height, image_orig_width, 2);
                    //LOGE("%s, face_origin:(%f,%f,%f,%f), conf=%.4f\n", [[img_path lastPathComponent] UTF8String], face_orig.x1, face_orig.y1, face_orig.x2, face_orig.y2, face_orig.score);
                    // set face region for phase1 model
                    if(!self.predictor_phase1->SetFaceRegion(face_orig.x1, face_orig.y1, face_orig.x2, face_orig.y2)) {
                        LOGE("Error no valid faces found!");
                        continue;
                    }
                }
                // 2) predict
                std::shared_ptr<TNNSDKOutput> sdk_output = nullptr;
                if (image_mat == nullptr) {
                    image_mat = [self ConvertImageToMat: compute_units :orig_image_dims :image_data];
                }
                status = self.predictor_phase1->Predict(std::make_shared<YoutuFaceAlignInput>(image_mat), sdk_output);
                [self CheckStatusSetLabel:status];
                BREAK_IF(status != TNN_OK);
                // update prev_face
                self.prev_face = self.predictor_phase1->GetPrevFace();
                if(!self.prev_face) {
                    LOGE("Next image: %s, will use face detector!\n", [[img_path lastPathComponent] UTF8String]);
                }
                phase1_pts = self.predictor_phase1->GetPrePts();
            }

            std::shared_ptr<TNN_NS::Mat> phase2_pts = nullptr;
            //phase2 model
            {
                // 1) prepare phase1 pts
                self.predictor_phase2->SetPrePts(phase1_pts, true);
                // 2) predict
                std::shared_ptr<TNNSDKOutput> sdk_output = nullptr;
                status = self.predictor_phase2->Predict(std::make_shared<YoutuFaceAlignInput>(image_mat), sdk_output);
                [self CheckStatusSetLabel:status];
                BREAK_IF(status != TNN_OK);
                phase2_pts = self.predictor_phase2->GetPrePts();
            }

            UIImage* output_image = [self DrawFaceAlignResult:phase1_pts :phase2_pts :input_image :img_path];
            if(idx == [self.result count]) {
                last_frame = output_image;
            }
        }
    }
    // update view image
    self.imageView.image = last_frame;
}

@end
