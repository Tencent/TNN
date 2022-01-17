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

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <CoreMedia/CoreMedia.h>

typedef NS_ENUM(NSInteger, CameraDeviceEvent) {
    CameraDeviceEvent_Started = 0,
    CameraDeviceEvent_Stopped ,
    CameraDeviceEvent_Restarted,
    CameraDeviceEvent_FrameStarted,
    CameraDeviceEvent_FrameReceived,
    CameraDeviceEvent_PositionChanged,
    CameraDeviceEvent_FlashModeSetted,
    CameraDeviceEvent_FocusBegan,
    CameraDeviceEvent_FocusEnded,
    CameraDeviceEvent_ExposureBegan,
    CameraDeviceEvent_ExposureEnded,
};
@class TNNCameraVideoDevice;

typedef void(^CameraSetupCallback)(BOOL);

@protocol TNNCameraVideoDeviceDelegate <NSObject>
@optional
- (void)cameraDevice:(TNNCameraVideoDevice *)camera
     didCaptureVideo:(CMSampleBufferRef)videoBuffer
        withPosition:(AVCaptureDevicePosition)position
         atTimestamp:(CMTime)time;
- (void)cameraDevice:(TNNCameraVideoDevice *)camera
     didCapturePhoto:(CMSampleBufferRef)photoBuffer
         previewImage:(CMSampleBufferRef)previewBuffer;
@end

@interface TNNCameraVideoDevice : NSObject
@property (nonatomic, weak) NSObject<TNNCameraVideoDeviceDelegate> *delegate;
@property (nonatomic, strong, readonly) AVCaptureVideoPreviewLayer *videoPreviewLayer;
@property (nonatomic, assign) AVCaptureVideoOrientation videoOrientation;
@property (nonatomic, strong, readonly) dispatch_queue_t queue;

- (instancetype)init;
- (void)startSession;
- (void)stopSession;

- (void)switchCamera:(AVCaptureDevicePosition)sessionPreset
          withPreset:(AVCaptureSessionPreset)sessionPreset
         completion:(CameraSetupCallback)completion;

-(id<MTLTexture>)getMTLTexture:(CMSampleBufferRef)sampleBuffer;
-(id<MTLTexture>)getMTLTextureFromImageBuffer:(CVImageBufferRef)imageBuffer;
-(UIImage *)getUIImage:(CMSampleBufferRef)sampleBuffer;

- (AVCaptureDevicePosition)cameraPosition;
@end
