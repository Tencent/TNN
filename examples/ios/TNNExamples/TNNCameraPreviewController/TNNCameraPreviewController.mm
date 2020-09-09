//  Copyright © 2020 tencent. All rights reserved.

#import "TNNCameraPreviewController.h"
#import "TNNCameraVideoDevice.h"
#import <Metal/Metal.h>
#import <CoreMedia/CoreMedia.h>
#import <tnn/tnn.h>
#import "UIImage+Utility.h"
#import "ultra_face_detector.h"

using namespace std;
using namespace TNN_NS;

@interface TNNCameraPreviewController () <TNNCameraVideoDeviceDelegate> {
    std::shared_ptr<UltraFaceDetector> detector_;
}
@property (nonatomic, weak) IBOutlet UIImageView *cameraPreview;
@property (nonatomic, weak) IBOutlet UILabel *labelResult;
@property (nonatomic, weak) IBOutlet UISwitch *switchGPU;
@property (nonatomic, weak) IBOutlet UIButton *rotateCamera;
@property (nonatomic, strong) IBOutlet UISwitch *startDetect;
@property (nonatomic, assign) Boolean isStartDetect;
@property (nonatomic, strong) TNNCameraVideoDevice *cameraDevice;
@property (nonatomic, assign) uint64_t frameCount;
@property (nonatomic, strong) NSString *label;
@property (nonatomic, assign) Boolean isFront;
@property (nonatomic, assign) Boolean useGPU;

@end

@implementation TNNCameraPreviewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.cameraDevice = [[TNNCameraVideoDevice alloc] initWithPreviewView:self.cameraPreview];
    self.cameraDevice.delegate = self;
    self.isStartDetect = YES;
    self.isFront = YES;
    self.useGPU = NO;
    self.frameCount = 0;
    detector_ = nullptr;
    
    [self.cameraDevice startSession];
}

- (void)viewWillAppear:(BOOL)animated
{
    [super viewWillAppear:animated];
    
    auto view = self.cameraPreview.superview;
    [self.cameraPreview removeFromSuperview];
    [self.labelResult removeFromSuperview];
    int width = view.frame.size.width - 60;
    int height = width * 640 / 480;
    
    self.cameraPreview.frame = CGRectMake(30, 100, width, height);
    [view addSubview:self.cameraPreview];
    self.labelResult.frame = CGRectMake(self.cameraPreview.frame.origin.x, self.cameraPreview.frame.origin.y + height + 5, self.labelResult.frame.size.width, self.labelResult.frame.size.height);
    [view addSubview:self.labelResult];
}

- (void)viewWillDisappear:(BOOL)animated
{
    [super viewWillDisappear:animated];
    
    //for safety, set detector_ nullptr after stop camera
    [self.cameraDevice stopSession];
    detector_ = nullptr;
}

#pragma mark - IBAction Interfaces

- (IBAction)onSwitchGPU:(id)sender
{
    self.useGPU = !self.useGPU;
    self.frameCount = 0;
}

- (IBAction)onCameraRotate:(id)sender {
    self.isFront = !self.isFront;
    [self.cameraDevice rotateCamera];
}

#pragma mark - Detect Interfaces

- (std::vector<FaceInfo>)detectFace:(CVImageBufferRef)image_buffer
{
    const int target_height = 640;
    const int target_width = 480;
//    const int target_height = 320;
//    const int target_width = 240;
    CGSize image_buffer_size = CVImageBufferGetDisplaySize(image_buffer);
    DimsVector target_dims = {1, 3, target_height, target_width};
    
    TNN_NS::Status status;
    std::vector<FaceInfo> face_info;
    if (!self.isStartDetect) {
        detector_ = nullptr;
        return face_info;
    }
    
    auto image_data = utility::CVImageBuffRefGetData(image_buffer, target_height, target_width);
    
    ++self.frameCount;
    if (self.frameCount == 1) {
        //Get metallib path from app bundle
        //PS：A script(Build Phases -> Run Script) is added to copy the metallib file from tnn framework project to TNNExamples app
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
            self.label = @"proto or model path is invalid";
            [self updateLabel:self.label];
            NSLog(@"Error: proto or model path is invalid");
            return face_info;
        }

        NSString *protoFormat = [NSString stringWithContentsOfFile:proto_path
                                                       encoding:NSUTF8StringEncoding
                                                          error:nil];
        string proto_content = protoFormat.UTF8String;
        NSData *data = [NSData dataWithContentsOfFile:model_path];
        string model_content = [data length] > 0 ? string((const char *)[data bytes], [data length]) : "";
        if (proto_content.size() <= 0 || model_content.size() <=0) {
            self.label =@"proto or model path is invalid";
            [self updateLabel:self.label];
            NSLog(@"Error: proto or model path is invalid");
            return face_info;
        }
        
        
        TNNComputeUnits units = self.useGPU ? TNNComputeUnitsGPU : TNNComputeUnitsCPU;
        
        detector_ = std::make_shared<UltraFaceDetector>(target_width, target_height, 1, 0.975, 0.23, 1);
        std::vector<int> nchw = {1, 3, target_height, target_width};
        status = detector_->Init(proto_content, model_content, library_path.UTF8String, units, nchw);
        if (status != TNN_OK) {
            self.label = [NSString stringWithFormat:@"%s", status.description().c_str()];
            [self updateLabel:self.label];
            NSLog(@"Error: %s", status.description().c_str());
            return face_info;
        }
        
        BenchOption bench_option;
        bench_option.forward_count = 1;
        detector_->SetBenchOption(bench_option);
    }
    
    if (detector_ == nullptr) return face_info;
    
    //for muti-thread safety, increase ref count, to insure detector is not released while detecting face
    auto detector_async_thread = detector_;
    
    auto compute_units = detector_async_thread->GetComputeUnits();
    if (compute_units == TNNComputeUnitsGPU) {
         auto image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_METAL, TNN_NS::N8UC4, target_dims);
        
        id<MTLTexture> texture_rgba = (__bridge id<MTLTexture>)image_mat->GetData();
        if (!texture_rgba) {
            self.label = @"Error texture input rgba is nil";
            [self updateLabel:self.label];
            NSLog(@"Error texture input rgba is nil");
            return face_info;
        }
        
        [texture_rgba replaceRegion:MTLRegionMake2D(0, 0, target_width, target_height)
                        mipmapLevel:0
                          withBytes:image_data.get()
                        bytesPerRow:target_width*4];
        status = detector_async_thread->Detect(image_mat, target_height, target_width, face_info);
    }
    else if (compute_units == TNNComputeUnitsCPU)
    {
        auto image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_ARM, TNN_NS::N8UC4, target_dims, image_data.get());
        status = detector_async_thread->Detect(image_mat, target_height, target_width, face_info);
    }
    
    face_info = AdjustFaceInfoToOriginalSize(face_info, target_height, target_width,
                                             image_buffer_size.height, image_buffer_size.width);
    
    if (status != TNN_OK) {
        self.label = [NSString stringWithUTF8String:status.description().c_str()];
        [self updateLabel:self.label];
        NSLog(@"Error: %s", status.description().c_str());
        return face_info;
    }

    auto bench_result = detector_async_thread->GetBenchResult();
    self.label = [NSString stringWithFormat:@"device: %@      face count:%d\ntime:\n%s",
                             compute_units == TNNComputeUnitsGPU ? @"gpu": @"arm",
                             (int)face_info.size(),
                             bench_result.Description().c_str()];
    [self updateLabel:self.label];
    return face_info;
}
#pragma mark - Draw Image Intefaces

- (void)drawPreview:(CMSampleBufferRef)buffer {
    CVImageBufferRef image_buffer = CMSampleBufferGetImageBuffer(buffer);
    CGSize size = CVImageBufferGetDisplaySize(image_buffer);
    int target_height = 640;
    int target_width = 480;
//    if (target_height != (int)size.height && target_width != (int)size.width) {
//        target_height = size.height;
//        target_width = size.width;
//        dispatch_async(dispatch_get_main_queue(), ^{
//            self.cameraPreview.frame = CGRectMake(self.cameraPreview.frame.origin.x, self.cameraPreview.frame.origin.y, 240, 320);
//        });
//    }
    
    //check release mode at Product->Scheme when running
    //运行时请在Product->Scheme中确认意见调整到release模式
    std::vector<FaceInfo> faceInfo = [self detectFace:image_buffer];
    
    
    std::vector<CGRect> faceRects;
    for (int i = 0; i < faceInfo.size(); i++) {
        auto face = faceInfo[i];
        CGRect faceRect = CGRectMake(face.x1, face.y1, face.x2-face.x1, face.y2-face.y1);
        faceRects.push_back(faceRect);
    }
    
    UIImage *originImage = utility::UIImageWithCVImageBuffRef(image_buffer);
    [self drawFaceRectWithImage:originImage withRects:faceRects];
}

- (void)drawFaceRectWithImage:(UIImage*)image withRects:(std::vector<CGRect>)rects
{
    CGSize imageSize = image.size;
    //UIGraphicsBeginImageContextWithOptions(imageSize,YES,1.0);
    UIGraphicsBeginImageContextWithOptions(imageSize,NO,1.0);
    [image drawInRect:CGRectMake(0, 0, imageSize.width, imageSize.height)];
    
    //step1: clear all content
    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextSetFillColorWithColor(context, [UIColor clearColor].CGColor);
    CGContextFillRect(context, CGRectMake(0, 0, imageSize.width, imageSize.height));
    
    CGContextSetStrokeColorWithColor(context, [UIColor redColor].CGColor);
    CGContextSetFillColorWithColor(context, [UIColor whiteColor].CGColor);
    if (imageSize.height > 0 && imageSize.width > 0) {
        for (int i = 0; i < rects.size(); ++i) {
            CGRect rect = rects[i];
            CGContextMoveToPoint(context, rect.origin.x, rect.origin.y);
            CGContextAddLineToPoint(context, rect.origin.x, rect.origin.y + rect.size.height);
            CGContextStrokePath(context);
            CGContextMoveToPoint(context, rect.origin.x, rect.origin.y);
            CGContextAddLineToPoint(context, (rect.origin.x + rect.size.width), rect.origin.y);
            CGContextStrokePath(context);
            CGContextMoveToPoint(context, (rect.origin.x + rect.size.width), rect.origin.y + rect.size.height);
            CGContextAddLineToPoint(context, rect.origin.x, rect.origin.y + rect.size.height);
            CGContextStrokePath(context);
            CGContextMoveToPoint(context, (rect.origin.x + rect.size.width), rect.origin.y + rect.size.height);
            CGContextAddLineToPoint(context, (rect.origin.x + rect.size.width), rect.origin.y);
            CGContextStrokePath(context);
        }
    }
    
    UIImage *o_image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    dispatch_async(dispatch_get_main_queue(), ^{
        UIImage *flip_image = o_image;
        if (self.isFront) {
            UIImageOrientation orientation = UIImageOrientationUpMirrored;
            flip_image = [UIImage imageWithCGImage:o_image.CGImage scale:o_image.scale orientation:orientation];
        }
        [self.cameraPreview setImage:flip_image];
    });
}


#pragma mark - Camera Interfaces
- (void)cameraDeviceEvent:(CameraDeviceEvent)event withAguments:(NSDictionary *)args {
    switch (event) {
        case CameraDeviceEvent_FrameReceived:
        {
            CMSampleBufferRef buffer = (CMSampleBufferRef)[[args objectForKey:@"buffer"] integerValue];
            [self drawPreview:buffer];
        }
            break;
            
        default:
            break;
    }
}


- (void)updateLabel:(NSString *)text
{
    dispatch_async(dispatch_get_main_queue(), ^{
        self.labelResult.text = self.label;
    });
}

@end
