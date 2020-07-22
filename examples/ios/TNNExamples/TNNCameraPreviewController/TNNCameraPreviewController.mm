//  Copyright © 2020 tencent. All rights reserved.

#import "TNNCameraPreviewController.h"
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

#define TEST_IMAGE_SSD 0

@interface TNNCameraPreviewController () <TNNCameraVideoDeviceDelegate> {
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

@implementation TNNCameraPreviewController

- (void)viewDidLoad {
    [super viewDidLoad];
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
    [_cameraDevice switchCamera:AVCaptureDevicePositionBack
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
    auto units = self.switchGPU.isOn ? TNNComputeUnitsCPU : TNNComputeUnitsGPU;
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
    
    const auto target_dims = self.viewModel.predictor->GetInputShape();
    auto target_height = target_dims[2];
    auto target_width = target_dims[3];
    // block until the next GPU buffer is available.
    dispatch_semaphore_wait(_inflightSemaphore, DISPATCH_TIME_FOREVER);
    
    //for muti-thread safety, increase ref count, to insure predictor is not released while detecting object
    auto fps_counter_async_thread = _fps_counter;
    auto predictor_async_thread = _viewModel.predictor;
    auto compute_units = _viewModel.predictor->GetComputeUnits();
    
    int origin_width = CVPixelBufferGetWidth(image_buffer);
    int origin_height = CVPixelBufferGetHeight(image_buffer);
    CGSize origin_image_size = CGSizeMake(origin_width, origin_height);

#if TEST_IMAGE_SSD
    origin_image_size = CGSizeMake(768, 538);
#endif
    
    //异步运行模型
    CVBufferRetain(image_buffer);
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        Status status = TNN_OK;
        std::vector<FaceInfo> face_info;
        std::map<std::string, double> map_fps;
        
        //resize
        fps_counter_async_thread->Begin("resize");
#if TEST_IMAGE_SSD
        auto image_png = [UIImage imageWithContentsOfFile:[[NSBundle mainBundle] pathForResource:@"dog_cropped.jpg"
                                                                                          ofType:nil]];
        auto image_data = utility::UIImageGetData(image_png, target_height, target_width);
        
        
#else
        auto image_data = utility::CVImageBuffRefGetData(image_buffer, target_height, target_width);
//        auto image_terget = utility::UIImageWithDataRGBA(image_data.get(), target_height, target_width);
//        UIImageWriteToSavedPhotosAlbum(image_terget, nil, nil, nil);
#endif
        
        fps_counter_async_thread->End("resize");
        CVBufferRelease(image_buffer);
        
        std::shared_ptr<TNNSDKOutput> sdk_output = nullptr;
        do {
            if (compute_units == TNNComputeUnitsGPU) {
                 auto image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_METAL, TNN_NS::N8UC4, target_dims);

                id<MTLTexture> texture_rgba = (__bridge id<MTLTexture>)image_mat->GetData();
                if (!texture_rgba) {
                    status = Status(TNNERR_NET_ERR, "Error texture input rgba is nil");
                    break;
                }

                [texture_rgba replaceRegion:MTLRegionMake2D(0, 0, target_width, target_height)
                                mipmapLevel:0
                                  withBytes:image_data.get()
                                bytesPerRow:target_width*4];
                fps_counter_async_thread->Begin("detect");
                status = predictor_async_thread->Predict(std::make_shared<UltraFaceDetectorInput>(image_mat), sdk_output);
                fps_counter_async_thread->End("detect");
            }
            else if (compute_units == TNNComputeUnitsCPU)
            {
                auto image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_ARM, TNN_NS::N8UC4, target_dims, image_data.get());
                fps_counter_async_thread->Begin("detect");
                status = predictor_async_thread->Predict(std::make_shared<UltraFaceDetectorInput>(image_mat), sdk_output);
                fps_counter_async_thread->End("detect");
            }
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
    auto object_list = [self.viewModel getObjectList:output];
    [self showObjectInfo:object_list withOriginImageSize:size withStatus:status];
}

- (void)showObjectInfo:(std::vector<std::shared_ptr<ObjectInfo> >)object_list
            withOriginImageSize:(CGSize)origin_size
            withStatus:(Status)status {
    object_list = [self reorder:object_list];
    
    //Object
    auto camera_pos = [self.cameraDevice cameraPosition];
    for (int i=0; i<_boundingBoxes.count; i++) {
        if ( i < object_list.size()) {
            auto object = object_list[i];
            auto view_width = self.cameraPreview.bounds.size.width;
            auto view_height = self.cameraPreview.bounds.size.height;
            auto label = [self.viewModel labelForObject:object];
            auto view_face = object->AdjustToImageSize(origin_size.height, origin_size.width);
            view_face = view_face.AdjustToViewSize(view_height, view_width, 1);
            if (camera_pos == AVCaptureDevicePositionFront) {
                view_face = view_face.FlipX();
            }
            [_boundingBoxes[i] showText:label
                              withColor:self.colors[i]
                                atFrame:CGRectMake(view_face.x1, view_face.y1,
                                                   view_face.x2-view_face.x1,
                                                   view_face.y2-view_face.y1)];
        } else {
            [_boundingBoxes[i] hide];
        }
    }
    
    //status
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithFormat:@"%s", status.description().c_str()];
    }
}

- (std::vector<std::shared_ptr<ObjectInfo> >)reorder:(std::vector<std::shared_ptr<ObjectInfo> >) object_list {
    if (_object_list_last.size() > 0 && object_list.size() > 0) {
        std::vector<std::shared_ptr<ObjectInfo> > object_list_reorder;
        //按照原有排序插入object_list中原先有的元素
        for (int index_last = 0; index_last < _object_list_last.size(); index_last++) {
            auto object_last = _object_list_last[index_last];
            //寻找最匹配元素
            int index_target = 0;
            float area_target = -1;
            for (int index=0; index<object_list.size(); index++) {
                auto object = object_list[index];
                auto area = object_last->IntersectionRatio(object.get());
                if (area > area_target) {
                    area_target = area;
                    index_target = index;
                }
            }

            if (area_target > 0) {
                object_list_reorder.push_back(object_list[index_target]);
                //删除指定下标元素
                object_list.erase(object_list.begin() + index_target);
            }
        }

        //插入原先没有的元素
        if (object_list.size() > 0) {
            object_list_reorder.insert(object_list_reorder.end(), object_list.begin(), object_list.end());
        }

        _object_list_last = object_list_reorder;
        return object_list_reorder;
    } else{
        _object_list_last = object_list;
        return object_list;
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
