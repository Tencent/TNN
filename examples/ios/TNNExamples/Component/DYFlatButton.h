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

typedef enum : NSUInteger {
    DYAutoHighlightModeNone = 0,
    DYAutoHighlightModeTitle = 1<<0,
    DYAutoHighlightModeBorder = 1<<1,
    DYAutoHighlightModeBackground = 1<<2,
    DYAutoHighlightModeAll =     DYAutoHighlightModeTitle|DYAutoHighlightModeBorder|DYAutoHighlightModeBackground,
} DYAutoHighlightMode;

@interface DYFlatButton : UIButton
/**highlight状态 亮度变化系数， >0, 变亮， <0,变暗*/
@property (nonatomic, assign) CGFloat hightLightBrightnessPercent;
/**是否自动根据hightLightBrightnessPercent确定highlight状态的颜色*/
@property (nonatomic, assign) DYAutoHighlightMode autoHighlightMode;

- (void)setBorderColor:(UIColor *)color forState:(UIControlState)state;
- (UIColor *)borderColorForState:(UIControlState)state;

- (void)setBackgroundColor:(UIColor *)color forState:(UIControlState)state;
- (UIColor *)backgroundColorForState:(UIControlState)state;
@end
