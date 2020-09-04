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
#import "TNNViewModel.h"
#import "TNNYoutuFaceAlignViewModel.h"
#import "YoutuFaceAlign.h"
#import "TNNFPSCounter.h"

#import "UIImage+Utility.h"

@interface TNNYoutuFaceAlignViewModel : TNNViewModel
@property (nonatomic, assign) std::shared_ptr<YoutuFaceAlign> predictor_phase1;
@property (nonatomic, assign) std::shared_ptr<YoutuFaceAlign> predictor_phase2;

@property bool prev_face;

@property (nonatomic, strong) dispatch_semaphore_t device_change_lock;

-(Status)loadNeuralNetworkModel:(TNNComputeUnits)units;

-(Status)Run:(std::shared_ptr<char>)image_data height:(int) height width :(int) width output:(std::shared_ptr<TNNSDKOutput>&) sdk_output counter:(std::shared_ptr<TNNFPSCounter>) counter;

-(YoutuFaceAlignInfo)getFace:(std::shared_ptr<TNNSDKOutput>)sdk_output;

@end
