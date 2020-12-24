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

#import "DYFlatButton.h"
#import "UIColor+Utility.h"

@interface DYFlatButton ()
@property (nonatomic, strong) NSMutableDictionary *borderColorDict;
@property (nonatomic, strong) NSMutableDictionary *backgroundColorDict;
@end

@implementation DYFlatButton
- (instancetype)initWithFrame:(CGRect)frame
{
    self = [super initWithFrame:frame];
    if (self) {
        [self setup];
    }
    return self;
}

- (instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super initWithCoder:aDecoder];
    if (self) {
        [self setup];
    }
    return self;
}

- (void)setup {
    _borderColorDict = [NSMutableDictionary dictionary];
    _backgroundColorDict = [NSMutableDictionary dictionary];
    
    _autoHighlightMode = DYAutoHighlightModeTitle|DYAutoHighlightModeBorder;
    _hightLightBrightnessPercent = -0.2;
    
    BOOL containsEdgeInsets = ! UIEdgeInsetsEqualToEdgeInsets(self.contentEdgeInsets, UIEdgeInsetsZero);
    self.contentEdgeInsets = containsEdgeInsets ? self.contentEdgeInsets : UIEdgeInsetsMake(0, 12, 0, 12);
    
    self.layer.masksToBounds = YES;
    
    if (self.backgroundColor) {
        [self setBackgroundColor:self.backgroundColor forState:UIControlStateNormal];
    }
}

- (void)layoutSubviews {
    UIControlState state = self.state;
    
    self.layer.borderWidth = self.layer.borderWidth ?: 1.0f;
    if (_autoHighlightMode & DYAutoHighlightModeTitle) {
        [self setTitleColor:[self titleColorForState:state] forState:state];
    }
    if (_autoHighlightMode & DYAutoHighlightModeBackground) {
        self.backgroundColor = [self backgroundColorForState:state];
    }
    
    self.layer.borderColor = [self borderColorForState:state].CGColor;
    
    [super layoutSubviews];
    self.layer.cornerRadius = self.layer.cornerRadius ?: CGRectGetHeight(self.frame) / 2.0f;
}

- (void)setHighlighted:(BOOL)highlighted {
    [super setHighlighted:highlighted];
}

- (UIColor *)titleColorForState:(UIControlState)state {
    if ((_autoHighlightMode & DYAutoHighlightModeTitle) &&
        (UIControlStateHighlighted & state)) {
        state = state & (~UIControlStateHighlighted);
        UIColor *color = [super titleColorForState: state];
        return [color lighten:_hightLightBrightnessPercent];
    } else {
        return [super titleColorForState:state];
    }
}

- (UIColor *)borderColorForState:(UIControlState)state {
    UIColor *color = _borderColorDict[@(state)];
    if (color == nil) {
        if (state == UIControlStateNormal) {
            color = [self titleColorForState:UIControlStateNormal];
        } else {
            if ((_autoHighlightMode & DYAutoHighlightModeBorder)
                && (UIControlStateHighlighted & state)) {
                state = state & (~UIControlStateHighlighted);
                UIColor *color = [self borderColorForState:state];
                if (color) {
                    return [color lighten:_hightLightBrightnessPercent];
                }
            } 
        }
    }
    return color;
}

- (void)setBorderColor:(UIColor *)color forState:(UIControlState)state {
    _borderColorDict[@(state)] = color;
}

- (UIColor *)backgroundColorForState:(UIControlState)state {
    UIColor *color = _backgroundColorDict[@(state)];
    if (color == nil) {
        if (state == UIControlStateNormal) {
            color = [self backgroundColor];
        } else {
            if ((_autoHighlightMode & DYAutoHighlightModeBackground)
                && (UIControlStateHighlighted & state)) {
                state = state & (~UIControlStateHighlighted);
                UIColor *color = [self backgroundColorForState:state];
                if (color) {
                    return [color lighten:_hightLightBrightnessPercent];
                }
            } 
        }
    }
    return color;
}

- (void)setBackgroundColor:(UIColor *)color forState:(UIControlState)state {
    _backgroundColorDict[@(state)] = color;
}

@end
