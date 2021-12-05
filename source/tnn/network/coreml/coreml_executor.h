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

#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <string>
#include <vector>
#include "Model.pb-c.h"

@interface CoreMLExecutor : NSObject

- (NSURL*)saveModel:(CoreML__Specification__Model*)model API_AVAILABLE(ios(12.0));
- (bool)build:(NSURL*)modelUrl API_AVAILABLE(ios(12.0));
- (bool)cleanup;
- (NSString*) getMLModelFilePath;

@property MLModel* model API_AVAILABLE(ios(12.0));
@property NSString* mlModelFilePath;
@property NSString* compiledModelFilePath;
@property(nonatomic, readonly) int coreMlVersion;
@end


