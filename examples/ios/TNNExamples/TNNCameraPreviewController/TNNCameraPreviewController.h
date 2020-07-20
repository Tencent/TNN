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
#import "TNNSDKSample.h"
#include "TNNFPSCounter.h"

@interface TNNCameraPreviewController : TNNExamplesController {
    std::shared_ptr<TNN_NS::TNNSDKSample> _predictor;
    std::vector<TNN_NS::ObjectInfo> _faces_last;
    std::shared_ptr<TNNFPSCounter> _fps_counter;
}

@end
