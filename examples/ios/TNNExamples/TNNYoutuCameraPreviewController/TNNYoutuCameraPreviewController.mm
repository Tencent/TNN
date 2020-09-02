//  Copyright © 2020 tencent. All rights reserved.

#import "TNNYoutuCameraPreviewController.h"
#import "TNNCameraVideoDevice.h"
#import <Metal/Metal.h>
#import <CoreMedia/CoreMedia.h>
#import <tnn/tnn.h>
#import "TNNBoundingBox.h"
#include "TNNFPSCounter.h"
#import "UIImage+Utility.h"
#import "UltraFaceDetector.h"

using namespace std;
using namespace TNN_NS;

typedef void(^CommonCallback)(Status);
#define kMaxBuffersInFlight 1

@interface TNNYoutuCameraPreviewController () <TNNCameraVideoDeviceDelegate> {
    std::vector<std::shared_ptr<ObjectInfo> > _object_list_last;
}
@property (nonatomic, weak) IBOutlet UIImageView *cameraPreview;
@property (nonatomic, weak) IBOutlet UILabel *labelResult;
@property (nonatomic, weak) IBOutlet UILabel *labelFPS;
@property (nonatomic, weak) IBOutlet UISwitch *switchGPU;
@property (nonatomic, weak) IBOutlet UIButton *rotateCamera;

@property (nonatomic, strong) TNNCameraVideoDevice *cameraDevice;
@property (nonatomic, strong) NSString *label;

@property (nonatomic, strong) dispatch_semaphore_t inflightSemaphore;

@property (nonatomic, strong) NSArray<TNNBoundingBox *> *boundingBoxes;
@property (nonatomic, strong) NSArray<UIColor *> *colors;
@end

@implementation TNNYoutuCameraPreviewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.navigationItem.title = _viewModel.title;
    
    //colors for each class
    auto colors = [NSMutableArray array];
    for (NSNumber *r in @[@(0.2), @(0.4), @(0.6), @(0.8), @(1.0)]) {
        for (NSNumber *g in @[@(0.3), @(0.7)]) {
            for (NSNumber *b in @[@(0.4), @(0.8)]) {
                [colors addObject:[UIColor colorWithRed:[r floatValue]
                                                  green:[g floatValue]
                                                   blue:[b floatValue]
                                                  alpha:1]];
            }
        }
    }
    self.colors = colors;
    
    _object_list_last = {};
    _fps_counter = std::make_shared<TNNFPSCounter>();
    
    _boundingBoxes = [NSArray array];
    _inflightSemaphore = dispatch_semaphore_create(kMaxBuffersInFlight);
    self.cameraDevice = [[TNNCameraVideoDevice alloc] init];
    self.cameraDevice.delegate = self;
    if (self.cameraDevice.videoPreviewLayer) {
        [self.cameraPreview.layer addSublayer:self.cameraDevice.videoPreviewLayer];
        [self resizePreviewLayer];
    }
    
    // add the bounding box layers to the UI, on top of the video preview.
    [self setupBoundingBox:12];
    
    //set up camera
    auto camera = _viewModel.preferFrontCamera ? AVCaptureDevicePositionFront : AVCaptureDevicePositionBack;
    [_cameraDevice switchCamera:camera
                     withPreset:AVCaptureSessionPreset640x480
                     completion:^(BOOL) {
    }];
    
    //init network
    auto units = self.switchGPU.isOn ? TNNComputeUnitsGPU : TNNComputeUnitsCPU;
    [self loadNeuralNetwork:units callback:^(Status status) {
        if (status != TNN_OK) {
            //刷新界面
            [self showSDKOutput:nullptr withStatus:status];
        }
    }];
    
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    [self resizePreviewLayer];
}

- (void)viewWillDisappear:(BOOL)animated
{
    [super viewWillDisappear:animated];
    
    //for safety, set _predictor nullptr after stop camera
    [self.cameraDevice stopSession];
}

- (void)setupBoundingBox:(NSUInteger)maxNumber {
    // Set up the bounding boxes.
    auto boundingBoxes = [NSMutableArray arrayWithArray:_boundingBoxes];
    for (NSUInteger i=_boundingBoxes.count; i<maxNumber; i++) {
        [boundingBoxes addObject:[[TNNBoundingBox alloc] init]];
    }
    
    for (TNNBoundingBox *iter in boundingBoxes) {
        [iter hide];
        [iter removeFromSuperLayer];
        
        [iter addToLayer:_cameraPreview.layer];
    }
    self.boundingBoxes = boundingBoxes;
}

- (void)resizePreviewLayer {
    if (_cameraDevice && _cameraPreview) {
        _cameraDevice.videoPreviewLayer.frame = _cameraPreview.bounds;
    }
}

- (void)loadNeuralNetwork:(TNNComputeUnits)units
                 callback:(CommonCallback)callback {
    //异步加载模型
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        Status status = [self.viewModel loadNeuralNetworkModel:units];
        dispatch_async(dispatch_get_main_queue(), ^{
            if (callback) {
                callback(status);
            }
        });
    });
}

#pragma mark - IBAction Interfaces

- (IBAction)onSwitchGPU:(id)sender
{
    //init network
    auto units = self.switchGPU.isOn ? TNNComputeUnitsGPU : TNNComputeUnitsCPU;
    [self loadNeuralNetwork:units callback:^(Status status) {
        if (status != TNN_OK) {
            //刷新界面
            [self showSDKOutput:nullptr withStatus:status];
        }
    }];
}

- (IBAction)onCameraRotate:(id)sender {
    auto position = [self.cameraDevice cameraPosition];
    position = (position == AVCaptureDevicePositionBack) ?
    AVCaptureDevicePositionFront : AVCaptureDevicePositionBack;
    
    [self.cameraDevice switchCamera:position
                         withPreset:AVCaptureSessionPreset640x480
                         completion:^(BOOL succes) {
    }];
}

#pragma mark - predict Interfaces
- (void)predict:(CVImageBufferRef)image_buffer {
    if (!_viewModel || !_viewModel.predictor) return;
    
    // block until the next GPU buffer is available.
    dispatch_semaphore_wait(_inflightSemaphore, DISPATCH_TIME_FOREVER);
    
    //for muti-thread safety, increase ref count, to insure predictor is not released while detecting object
    auto fps_counter_async_thread = _fps_counter;
    auto predictor_async_thread = _viewModel.predictor;
    //auto compute_units = _viewModel.predictor->GetComputeUnits();
    
    int origin_width = (int)CVPixelBufferGetWidth(image_buffer);
    int origin_height = (int)CVPixelBufferGetHeight(image_buffer);
    CGSize origin_image_size = CGSizeMake(origin_width, origin_height);
    
    //异步运行模型
    CVBufferRetain(image_buffer);
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        Status status = TNN_OK;
        std::map<std::string, double> map_fps;
        
        //resize
        fps_counter_async_thread->Begin("resize");

        auto image_data = utility::CVImageBuffRefGetData(image_buffer, origin_height, origin_width);
        
        fps_counter_async_thread->End("resize");
        CVBufferRelease(image_buffer);
        
        std::shared_ptr<TNNSDKOutput> sdk_output = nullptr;
        do {
            fps_counter_async_thread->Begin("detect");
            
            status = [self.viewModel Run:image_data:origin_height:origin_width:sdk_output];
            
            fps_counter_async_thread->End("detect");
            map_fps = fps_counter_async_thread->GetAllFPS();
            
        } while (0);
        
        dispatch_sync(dispatch_get_main_queue(), ^{
            [self showSDKOutput:sdk_output
            withOriginImageSize:origin_image_size
                     withStatus:status];
            [self showFPS:map_fps];
        });
        
        dispatch_semaphore_signal(self.inflightSemaphore);
    });

}

- (void)showSDKOutput:(std::shared_ptr<TNNSDKOutput>)output
       withOriginImageSize:(CGSize)size
           withStatus:(Status)status {
    auto face = [self.viewModel getFace:output];
    [self showFaceAlignment:face withOriginImageSize:size withStatus:status];
}

- (void)showFaceAlignment:(YoutuFaceAlignInfo) face
            withOriginImageSize:(CGSize)origin_size
            withStatus:(Status)status {
    //Object
    auto camera_pos = [self.cameraDevice cameraPosition];
    auto camera_gravity = [self.cameraDevice.videoPreviewLayer videoGravity];
    int video_gravity = 0;
    if (camera_gravity == AVLayerVideoGravityResizeAspectFill) {
        video_gravity = 2;
    } else if(camera_gravity == AVLayerVideoGravityResizeAspect) {
        video_gravity = 1;
    }
    {
        auto view_width = self.cameraPreview.bounds.size.width;
        auto view_height = self.cameraPreview.bounds.size.height;
        //LOGE("origin_size:%f,%f\n", origin_size.height, origin_size.width);
        auto view_face = face.AdjustToImageSize(origin_size.height, origin_size.width);
        view_face = view_face.AdjustToViewSize(view_height, view_width, video_gravity);
        if (camera_pos == AVCaptureDevicePositionFront) {
            view_face = view_face.FlipX();
        }
        auto pts = view_face.key_points;
        [_boundingBoxes[0] showMarkAtPoints:pts withColor:self.colors[0]];
    }
    //status
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithFormat:@"%s", status.description().c_str()];
    }
}

- (void)showFPS:(std::map<std::string, double>) map_fps {
    NSMutableString *fps = [NSMutableString stringWithFormat:@"device: %@",
                            self.switchGPU.isOn ? @"metal\n" : @"arm\n"];
    int index = 0;
    for (auto item : map_fps) {
        [fps appendFormat:@"%@fps %s: %.2f", index++ > 0 ? @"\n" : @"", item.first.c_str(), item.second];
    }
    self.labelFPS.text = fps;
}

#pragma mark - TNNCameraVideoDeviceDelegate
- (void)cameraDevice:(TNNCameraVideoDevice *)camera
     didCaptureVideo:(CMSampleBufferRef)videoBuffer
        withPosition:(AVCaptureDevicePosition)position
         atTimestamp:(CMTime)time {
//    auto texture = [camera getMTLTexture:videoBuffer];
//    NSLog(@"texture:%p", texture);
    auto imageBuffer = CMSampleBufferGetImageBuffer(videoBuffer);
    [self predict:imageBuffer];
}
@end

