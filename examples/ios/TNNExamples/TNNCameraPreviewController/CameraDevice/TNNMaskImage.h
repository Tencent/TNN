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
#import <UIKit/UIKit.h>
#include <memory>
#include <vector>

@interface TNNMaskImage : NSObject
@property (nonatomic, strong, readonly) CALayer *imageLayer;

- (instancetype)init;

-(void)addToLayer:(CALayer *)layer;
-(void)removeFromSuperLayer;

- (void)showImage:(UIImage *)image atFrame:(CGRect)frame;


- (void)hide;
@end


