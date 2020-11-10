//  Copyright © 2020 tencent. All rights reserved.

#import "TNNCameraPreviewController.h"
#import "TNNCameraVideoDevice.h"
#import <Metal/Metal.h>
#import <CoreMedia/CoreMedia.h>
#import <tnn/tnn.h>
#import "TNNBoundingBox.h"
#import "TNNMaskImage.h"
#include "tnn_fps_counter.h"
#import "UIImage+Utility.h"
#import "ultra_face_detector.h"

using namespace std;
using namespace TNN_NS;

typedef void(^CommonCallback)(Status);
#define kMaxBuffersInFlight 1

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

@property (nonatomic, strong) TNNMaskImage *maskImage;
@property (nonatomic, strong) NSArray<TNNBoundingBox *> *boundingBoxes;
@property (nonatomic, strong) NSArray<UIColor *> *colors;
@end

@implementation TNNCameraPreviewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.navigationItem.title = self.viewModel.title;
    
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
    // maskimage layer
    _maskImage = [[TNNMaskImage alloc] init];
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
    auto camera = self.viewModel.preferFrontCamera ? AVCaptureDevicePositionFront : AVCaptureDevicePositionBack;
    [_cameraDevice switchCamera:camera
                     withPreset:AVCaptureSessionPreset640x480
                     completion:^(BOOL) {
    }];
    
    //init network
    [self.switchGPU setOn:self.viewModel.preferGPU];
    auto units = self.switchGPU.isOn ? TNNComputeUnitsGPU : TNNComputeUnitsCPU;
    [self loadNeuralNetwork:units callback:^(Status status) {
        if (status != TNN_OK) {
            //刷新界面
            [self showSDKOutput:nullptr withOriginImageSize:CGSizeZero withStatus:status];
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
    [_maskImage hide];
    [_maskImage removeFromSuperLayer];
    [_maskImage addToLayer:_cameraPreview.layer];
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
            [self showSDKOutput:nullptr withOriginImageSize:CGSizeZero withStatus:status];
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
- (void)predictSampleBuffer:(CMSampleBufferRef)video_buffer
                 withCamera:(TNNCameraVideoDevice *)camera
               andPosition:(AVCaptureDevicePosition)position
                atTimestamp:(CMTime)time {
    if (!self.viewModel || !self.viewModel.predictor) return;
    
    const auto target_dims = self.viewModel.predictor->GetInputShape();
    // block until the next GPU buffer is available.
    dispatch_semaphore_wait(_inflightSemaphore, DISPATCH_TIME_FOREVER);
    
    //for muti-thread safety, increase ref count, to insure predictor is not released while detecting object
    auto fps_counter_async_thread = _fps_counter;
    auto predictor_async_thread = self.viewModel.predictor;
    auto compute_units = self.viewModel.predictor->GetComputeUnits();
    
    CVImageBufferRef image_buffer = CMSampleBufferGetImageBuffer(video_buffer);
    int origin_width = (int)CVPixelBufferGetWidth(image_buffer);
    int origin_height = (int)CVPixelBufferGetHeight(image_buffer);
    CGSize origin_image_size = CGSizeMake(origin_width, origin_height);
    
    id<MTLTexture> image_texture = nil;
    if (compute_units == TNNComputeUnitsGPU) {
        image_texture = [camera getMTLTextureFromImageBuffer:image_buffer];
    }
    //NSLog(@"==== (%d, %d)", origin_height, origin_width);
    
    //异步运行模型
    CVBufferRetain(image_buffer);
    auto image_texture_ref = CFBridgingRetain(image_texture);
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        Status status = TNN_OK;
        std::map<std::string, double> map_fps;

        //Note：智能指针必须在resize后才能释放
        std::shared_ptr<char> image_data = nullptr;
        std::shared_ptr<TNN_NS::Mat> image_mat = nullptr;
        // devan: to support generate UIImage, set channel to 4
        auto origin_dims = {1, 4, origin_height, origin_width};
        if (compute_units == TNNComputeUnitsCPU) {
            image_data = utility::CVImageBuffRefGetData(image_buffer);
            image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_ARM, TNN_NS::N8UC4, origin_dims, image_data.get());
        } else {
            image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_METAL, TNN_NS::N8UC4, origin_dims, (void *)image_texture_ref);
        }

//        auto input_mat = std::make_shared<TNN_NS::Mat>(image_mat->GetDeviceType(), TNN_NS::N8UC4, target_dims);
//#ifndef END2END
//        //resize
//        fps_counter_async_thread->Begin("resize");
//#endif
//        predictor_async_thread->Resize(image_mat, input_mat, TNNInterpLinear);
//#ifndef END2END
//        fps_counter_async_thread->End("resize");
//#endif
        
        std::shared_ptr<TNNSDKOutput> sdk_output = nullptr;
        do {
            fps_counter_async_thread->Begin("detect");
            status = predictor_async_thread->Predict(std::make_shared<TNNSDKInput>(image_mat), sdk_output);
            fps_counter_async_thread->End("detect");
        } while (0);
        
        CVBufferRelease(image_buffer);
        CFBridgingRelease(image_texture_ref);
        
        map_fps = fps_counter_async_thread->GetAllFPS();
        //auto time = fps_counter_async_thread->GetAllTime();

        dispatch_sync(dispatch_get_main_queue(), ^{
            [self showSDKOutput:sdk_output
            withOriginImageSize:origin_image_size
                     withStatus:status];
            [self showFPS:map_fps];
            //[self showTime: time];
        });
        
        dispatch_semaphore_signal(self.inflightSemaphore);
    });

}

- (void)showSDKOutput:(std::shared_ptr<TNNSDKOutput>)output
       withOriginImageSize:(CGSize)size
           withStatus:(Status)status {
    auto object_list = [self.viewModel getObjectList:output];
    [self showObjectInfo:object_list withOriginImageSize:size withStatus:status];
    auto mask_data   = [self.viewModel getImage:output];
    [self showMask:mask_data withOriginImageSize:size withStatus:status];
}

- (void)showObjectInfo:(std::vector<std::shared_ptr<ObjectInfo> >)object_list
            withOriginImageSize:(CGSize)origin_size
            withStatus:(Status)status {
    //status
    if (status != TNN_OK) {
        self.labelResult.text = [NSString stringWithFormat:@"%s", status.description().c_str()];
        
        for (int i=0; i<_boundingBoxes.count; i++) {
            [_boundingBoxes[i] hide];
        }
    } else {
        object_list = [self reorder:object_list];
        
        //Object
        auto camera_pos = [self.cameraDevice cameraPosition];
        auto camera_gravity = [self.cameraDevice.videoPreviewLayer videoGravity];
        int video_gravity = 0;
        if (camera_gravity == AVLayerVideoGravityResizeAspectFill) {
            video_gravity = 2;
        } else if(camera_gravity == AVLayerVideoGravityResizeAspect) {
            video_gravity = 1;
        }
        for (int i=0; i<_boundingBoxes.count; i++) {
            if ( i < object_list.size()) {
                auto object = object_list[i];
                auto view_width = self.cameraPreview.bounds.size.width;
                auto view_height = self.cameraPreview.bounds.size.height;
                auto label = [self.viewModel labelForObject:object];
                auto view_face = object->AdjustToImageSize(origin_size.height, origin_size.width);
                view_face = view_face.AdjustToViewSize(view_height, view_width, video_gravity);
                if (camera_pos == AVCaptureDevicePositionFront) {
                    view_face = view_face.FlipX();
                }
                [_boundingBoxes[i] showText:label
                                  withColor:self.colors[i]
                                    atFrame:CGRectMake(view_face.x1, view_face.y1,
                                                       view_face.x2-view_face.x1,
                                                       view_face.y2-view_face.y1)];
    //            [_boundingBoxes[i] showMarkAtPoints:{{(view_face.x1+view_face.x2)/2, (view_face.y1+view_face.y2)/2}} withColor:[UIColor redColor]];
                [_boundingBoxes[i] showMarkAtPoints:view_face.key_points withColor:[UIColor greenColor]];
            } else {
                [_boundingBoxes[i] hide];
            }
        }
    }
}

- (void)showMask:(ImageInfo)image_info
            withOriginImageSize:(CGSize)origin_size
            withStatus:(Status)status {
    if (!image_info.data)
        return;
    auto camera_pos = [self.cameraDevice cameraPosition];
    if (camera_pos == AVCaptureDevicePositionFront) {
        image_info = image_info.FlipX();
    }
    // devan: method to support RGB data?
    UIImage* image = utility::UIImageWithDataRGBA(image_info.data.get(), image_info.image_height, image_info.image_width);
    [_maskImage showImage:image atFrame:_cameraPreview.bounds];
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
        NSLog(@"%@fps %s: %.2f",  index++ > 0 ? @"\n" : @"", item.first.c_str(), item.second);
    }
    self.labelFPS.text = fps;
}

- (void)showTime:(std::map<std::string, double>) map_time {
    NSMutableString *fps = [NSMutableString stringWithFormat:@"device: %@",
                            self.switchGPU.isOn ? @"metal\n" : @"arm\n"];
    int index = 0;
    for (auto item : map_time) {
        [fps appendFormat:@"%@time %s: %.4f ms", index++ > 0 ? @"\n" : @"", item.first.c_str(), item.second];
        //LOGE("=== %s: %.4f\n", item.first.c_str(), item.second);
    }
    self.labelFPS.text = fps;
}

#pragma mark - TNNCameraVideoDeviceDelegate
- (void)cameraDevice:(TNNCameraVideoDevice *)camera
     didCaptureVideo:(CMSampleBufferRef)videoBuffer
        withPosition:(AVCaptureDevicePosition)position
         atTimestamp:(CMTime)time {
    [self predictSampleBuffer:videoBuffer
                   withCamera:camera
                  andPosition:position
                  atTimestamp:time];
}

@end
