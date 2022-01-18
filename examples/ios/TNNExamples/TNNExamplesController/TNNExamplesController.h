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
#import "TNNViewModel.h"
#import "TOSegmentedControl.h"
#include "tnn_sdk_sample.h"

@interface TNNExamplesController : UIViewController
@property(nonatomic, weak) IBOutlet TOSegmentedControl *switchDevice;
@property (nonatomic, weak) IBOutlet UIView *customOptionView;
@property (nonatomic, weak) IBOutlet NSLayoutConstraint *customOptionViewHeight;
@property (nonatomic, strong) TNNViewModel *viewModel;

- (void)onSwitchChanged:(id)sender;

-(TNNComputeUnits)getComputeUnitsForIndex:(NSInteger)index;
-(NSString *)getNSSTringForComputeUnits:(TNNComputeUnits)unit;

- (BOOL)shouldAutorotate;
- (UIInterfaceOrientationMask)supportedInterfaceOrientations;
@end


@interface UIViewController (UIDeviceOrientation)
- (void)forceToOrientation:(UIDeviceOrientation)orientation;
- (void)clearNavigationBarLeft;
@end
