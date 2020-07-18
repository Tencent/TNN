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
const int target_height = 640;
const int target_width = 480;

@interface TNNCameraPreviewController () <TNNCameraVideoDeviceDelegate> {
    std::shared_ptr<UltraFaceDetector> _detector;
    std::vector<FaceInfo> _faces_last;
    std::shared_ptr<TNNFPSCounter> _fps_counter;
}
@property (nonatomic, weak) IBOutlet UIImageView *cameraPreview;
@property (nonatomic, weak) IBOutlet UILabel *labelResult;
@property (nonatomic, weak) IBOutlet UILabel *labelFPS;
@property (nonatomic, weak) IBOutlet UISwitch *switchGPU;
@property (nonatomic, weak) IBOutlet UIButton *rotateCamera;

@property (nonatomic, strong) TNNCameraVideoDevice *cameraDevice;
@property (nonatomic, assign) uint64_t frameCount;
@property (nonatomic, strong) NSString *label;
@property (nonatomic, assign) Boolean isFront;
@property (nonatomic, assign) Boolean useGPU;

@property (nonatomic, strong) dispatch_group_t startupGroup;
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
    
    _faces_last = {};
    _fps_counter = std::make_shared<TNNFPSCounter>();
    
    _boundingBoxes = [NSArray array];
    _startupGroup = dispatch_group_create();
    _inflightSemaphore = dispatch_semaphore_create(kMaxBuffersInFlight);
    
    self.cameraDevice = [[TNNCameraVideoDevice alloc] initWithPreviewView:self.cameraPreview];
    self.cameraDevice.delegate = self;
    
    //set up camera
    dispatch_group_enter(_startupGroup);
    [_cameraDevice setupCamera:AVCaptureSessionPreset640x480
                    completion:^(BOOL) {
        if (self.cameraDevice.videoPreviewLayer) {
            [self.cameraPreview.layer addSublayer:self.cameraDevice.videoPreviewLayer];
            [self resizePreviewLayer];
        }
        dispatch_group_leave(self.startupGroup);
    }];
    
    //init network
    dispatch_group_enter(_startupGroup);
    [self createNeuralNetwork:^(Status status) {
        if (status != TNN_OK) {
            //刷新界面
            [self showObjectInfo:std::vector<FaceInfo>() withStatus:status];
        }
        dispatch_group_leave(self.startupGroup);
    }];
    
    dispatch_group_notify(_startupGroup, dispatch_get_main_queue(), ^{
        // add the bounding box layers to the UI, on top of the video preview.
        [self setupBoundingBox:12];
        
        [self.cameraDevice startSession];
    });
    
    self.isFront = YES;
    self.useGPU = YES;
    self.frameCount = 0;
}

- (void)viewWillAppear:(BOOL)animated
{
    [super viewWillAppear:animated];
}

- (void)viewWillDisappear:(BOOL)animated
{
    [super viewWillDisappear:animated];
    
    //for safety, set _detector nullptr after stop camera
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
    _cameraDevice.videoPreviewLayer.frame = _cameraPreview.bounds;
}

- (void)createNeuralNetwork:(CommonCallback)callback {
    //异步加载模型
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        Status status = [self loadNeuralNetworkModel];
        dispatch_async(dispatch_get_main_queue(), ^{
            if (callback) {
                callback(status);
            }
        });
    });
}

-(Status)loadNeuralNetworkModel {
    Status status = TNN_OK;
    
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
        status = Status(TNNERR_NET_ERR, "Error: proto or model path is invalid");
        NSLog(@"Error: proto or model path is invalid");
        return status;
    }

    NSString *protoFormat = [NSString stringWithContentsOfFile:proto_path
                                                   encoding:NSUTF8StringEncoding
                                                      error:nil];
    string proto_content = protoFormat.UTF8String;
    NSData *data = [NSData dataWithContentsOfFile:model_path];
    string model_content = [data length] > 0 ? string((const char *)[data bytes], [data length]) : "";
    if (proto_content.size() <= 0 || model_content.size() <=0) {
        status = Status(TNNERR_NET_ERR, "Error: proto or model path is invalid");
        NSLog(@"Error: proto or model path is invalid");
        return status;
    }
    
    TNNComputeUnits units = self.useGPU ? TNNComputeUnitsGPU : TNNComputeUnitsCPU;
    
    _detector = std::make_shared<UltraFaceDetector>(target_width, target_height, 1, 0.975, 0.23, 1);
    std::vector<int> nchw = {1, 3, target_height, target_width};
    status = _detector->Init(proto_content, model_content, library_path.UTF8String, units, nchw);
    if (status != TNN_OK) {
        NSLog(@"Error: %s", status.description().c_str());
        return status;
    }
    
    BenchOption bench_option;
    bench_option.forward_count = 1;
    _detector->SetBenchOption(bench_option);
    return status;
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

#pragma mark - predict Interfaces
- (void)predict:(CVImageBufferRef)image_buffer {
    if (_detector == nullptr) return;
    
    const DimsVector target_dims = {1, 3, target_height, target_width};
    
    // block until the next GPU buffer is available.
    dispatch_semaphore_wait(_inflightSemaphore, DISPATCH_TIME_FOREVER);
    
    //for muti-thread safety, increase ref count, to insure detector is not released while detecting face
    auto fps_counter_async_thread = _fps_counter;
    auto detector_async_thread = _detector;
    auto compute_units = detector_async_thread->GetComputeUnits();
    
    //异步运行模型
    CVBufferRetain(image_buffer);
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        Status status = TNN_OK;
        std::vector<FaceInfo> face_info;
        std::map<std::string, double> map_fps;
        
        //resize
        fps_counter_async_thread->Begin("resize");
        auto image_data = utility::CVImageBuffRefGetData(image_buffer, target_height, target_width);
        fps_counter_async_thread->End("resize");
        CVBufferRelease(image_buffer);
        
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
                status = detector_async_thread->Detect(image_mat, target_height, target_width, face_info);
                fps_counter_async_thread->End("detect");
            }
            else if (compute_units == TNNComputeUnitsCPU)
            {
                auto image_mat = std::make_shared<TNN_NS::Mat>(DEVICE_ARM, TNN_NS::N8UC4, target_dims, image_data.get());
                fps_counter_async_thread->Begin("detect");
                status = detector_async_thread->Detect(image_mat, target_height, target_width, face_info);
                fps_counter_async_thread->End("detect");
            }
            map_fps = fps_counter_async_thread->GetAllFPS();
    
//            face_info = AdjustFaceInfoToOriginalSize(face_info, target_height, target_width,
//                                                     image_buffer_size.height, image_buffer_size.width);
        } while (0);
        
        dispatch_async(dispatch_get_main_queue(), ^{
            NSLog(@"face count:%d", (int)face_info.size());
            [self showObjectInfo:face_info withStatus:status];
            [self showFPS:map_fps];
        });
        
        dispatch_semaphore_signal(self.inflightSemaphore);
    });

}

- (void)showObjectInfo:(std::vector<FaceInfo>)faces withStatus:(Status)status {
    faces = [self reorder:faces];
    
    //Object
    auto camera_pos = [self.cameraDevice cameraPosition];
    for (int i=0; i<_boundingBoxes.count; i++) {
        if ( i < faces.size()) {
            auto face = faces[i];
            auto view_width = self.cameraPreview.bounds.size.width;
            auto view_height = self.cameraPreview.bounds.size.height;
            auto label = [NSString stringWithFormat:@"%.2f", face.score];
            auto view_face = face.AdjustToViewSize(view_height, view_width, 2);
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

- (std::vector<FaceInfo>)reorder:(std::vector<FaceInfo>) faces {
    if (_faces_last.size() > 0 && faces.size() > 0) {
        std::vector<FaceInfo> faces_reorder;
        //按照原有排序插入faces中原先有的元素
        for (int index_last = 0; index_last < _faces_last.size(); index_last++) {
            auto face_last = _faces_last[index_last];
            //寻找最匹配元素
            int index_target = 0;
            float area_target = -1;
            for (int index=0; index<faces.size(); index++) {
                auto face = faces[index];
                auto area = face_last.IntersectionRatio(&face);
                if (area > area_target) {
                    area_target = area;
                    index_target = index;
                }
            }
            
            if (area_target > 0) {
                faces_reorder.push_back(faces[index_target]);
                //删除指定下标元素
                faces.erase(faces.begin() + index_target);
            }
        }
        
        //插入原先没有的元素
        if (faces.size() > 0) {
            faces_reorder.insert(faces_reorder.end(), faces.begin(), faces.end());
        }
        
        _faces_last = faces_reorder;
        return faces_reorder;
    } else{
        _faces_last = faces;
        return faces;
    }
}

- (void)showFPS:(std::map<std::string, double>) map_fps {
    NSMutableString *fps = [NSMutableString string];
    int index = 0;
    for (auto item : map_fps) {
        [fps appendFormat:@"%@%s: %.2f", index++ > 0 ? @"\n" : @"", item.first.c_str(), item.second];
    }
    self.labelFPS.text = fps;
}

#pragma mark - TNNCameraVideoDeviceDelegate
- (void)cameraDevice:(TNNCameraVideoDevice *)camera
     didCaptureVideo:(CMSampleBufferRef)videoBuffer
        withPosition:(AVCaptureDevicePosition)position
         atTimestamp:(CMTime)time {
    NSLog(@"didCaptureVideo");
//    auto texture = [camera getMTLTexture:videoBuffer];
//    NSLog(@"texture:%p", texture);
    auto imageBuffer = CMSampleBufferGetImageBuffer(videoBuffer);
    [self predict:imageBuffer];
}
@end
