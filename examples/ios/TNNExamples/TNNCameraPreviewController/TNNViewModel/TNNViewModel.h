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

#import <UIKit/UIKit.h>
#import "tnn_sdk_sample.h"

using namespace::TNN_NS;

@interface TNNViewModel : NSObject
@property (nonatomic, assign) std::shared_ptr<TNNSDKSample> predictor;

@property (nonatomic, strong) NSString *title;
@property (nonatomic, assign) UIDeviceOrientation preferDeviceOrientation;
@property (nonatomic, assign) bool preferFrontCamera;
@property (nonatomic, assign) TNNComputeUnits preferComputeUnits;
-(Status)loadNeuralNetworkModel:(TNNComputeUnits)units;

//Object Detection
-(std::vector<std::shared_ptr<ObjectInfo> >)getObjectList:(std::shared_ptr<TNNSDKOutput>)output;
-(ImageInfo)getImage:(std::shared_ptr<TNNSDKOutput>)sdk_output;
-(NSString*)labelForObject:(std::shared_ptr<ObjectInfo>)object;

//Custom UI control
- (BOOL)showImageAtMinorPreview;

- (void)setupCustomView:(UIView *)view
           layoutHeight:(NSLayoutConstraint *)viewLayoutHeight;

//Custom UI control
- (void)adajustStackPrevieView:(UIStackView *)stackView;
@end
