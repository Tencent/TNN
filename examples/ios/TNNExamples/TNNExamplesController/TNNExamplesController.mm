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

@interface TNNExamplesController () {
}

@end

@implementation TNNExamplesController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    __weak typeof(self) weakSelf = self;
    self.switchDevice.backgroundColor = [UIColor systemGrayColor];
    self.switchDevice.thumbColor = [UIColor systemGreenColor];
    self.switchDevice.items = @[@"CPU", @"GPU", @"NPU"];
    self.switchDevice.segmentTappedHandler = ^(NSInteger index, BOOL reversed) {
        [weakSelf onSwitchChanged:weakSelf.switchDevice];
    };
    
    [self.viewModel setupCustomView:self.customOptionView
                       layoutHeight:self.customOptionViewHeight];
}

- (void)onSwitchChanged:(id)sender {
}

@end
