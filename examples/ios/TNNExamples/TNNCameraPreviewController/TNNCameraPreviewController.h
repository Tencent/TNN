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

#import "TNNExamplesController.h"
#import "tnn_sdk_sample.h"
#import "tnn_fps_counter.h"
#import "TNNViewModel.h"

@interface TNNCameraPreviewController : TNNExamplesController {
    std::shared_ptr<TNNFPSCounter> _fps_counter;
}

@property (nonatomic, strong) TNNViewModel *viewModel;

- (void)showSDKOutput:(std::shared_ptr<TNNSDKOutput>)output
  withOriginImageSize:(CGSize)size
           withStatus:(Status)status;

@end
