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

@protocol TNNCameraVideoDeviceDelegate <NSObject>

- (void)cameraDeviceEvent:(CameraDeviceEvent)event withAguments:(NSDictionary *)args;

@end

@interface TNNCameraVideoDevice : NSObject
@property (nonatomic, weak) NSObject<TNNCameraVideoDeviceDelegate> *delegate;

- (id)initWithPreviewView:(UIView *)view;
- (void)startSession;
- (void)stopSession;
- (AVCaptureDevicePosition)rotateCamera;

@end
