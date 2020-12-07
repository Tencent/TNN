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

#import "UIColor+Utility.h"

@implementation UIColor (Utility)
+ (UIColor *)colorWithHexNumber:(NSUInteger)hexColor {
    CGFloat r = ((hexColor>>16) & 0xFF) / 255.0;
    CGFloat g = ((hexColor>>8) & 0xFF) / 255.0;
    CGFloat b = (hexColor & 0xFF) / 255.0;
    return [UIColor colorWithRed:r green:g blue:b alpha:1.0f];
}

+ (UIColor *)colorWithHexNumber:(NSUInteger)hexColor alpha:(CGFloat)alpha{
    CGFloat r = ((hexColor>>16) & 0xFF) / 255.0;
    CGFloat g = ((hexColor>>8) & 0xFF) / 255.0;
    CGFloat b = (hexColor & 0xFF) / 255.0;
    return [UIColor colorWithRed:r green:g blue:b alpha:alpha];
}

- (BOOL)isDarkColor {
    const CGFloat *RGB = CGColorGetComponents(self.CGColor);
    if (RGB == nil) {
        return YES;
    }
    return (0.2126 * RGB[0] + 0.7152 * RGB[1] + 0.0722 * RGB[2]) < 0.5;
}

- (BOOL)isBlackOrWhite {
    const CGFloat *RGB = CGColorGetComponents(self.CGColor);
    if (RGB == nil) {
        return YES;
    }
    return (RGB[0] > 0.91 && RGB[1] > 0.91 && RGB[2] > 0.91) || (RGB[0] < 0.09 && RGB[1] < 0.09 && RGB[2] < 0.09);
}

- (BOOL)isDistinct:(UIColor *)color {
    const CGFloat *bg = CGColorGetComponents(self.CGColor);
    const CGFloat *fg = CGColorGetComponents(color.CGColor);
    
    const CGFloat threshold = 0.35;
    if (fabs(bg[0] - fg[0]) > threshold || fabs(bg[1] - fg[1]) > threshold || fabs(bg[2] - fg[2]) > threshold) {
        if (fabs(bg[0] - bg[1]) < 0.03 && fabs(bg[0] - bg[2]) < 0.03) {
            if (fabs(fg[0] - fg[1]) < 0.03 && fabs(fg[0] - fg[2]) < 0.03) {
                return NO;
            }
        }
        return YES;
    }
    return NO;
}

- (UIColor *)colorWithMinimumSaturation:(CGFloat)minSaturation {
    CGFloat hue = 0.0;
    CGFloat saturation = 0.0;
    CGFloat brightness = 0.0;
    CGFloat alpha = 0.0;
    
    [self getHue:&hue saturation:&saturation brightness:&brightness alpha:&alpha];
    
    if (saturation < minSaturation) {
        return [UIColor colorWithHue:hue saturation:minSaturation brightness:brightness alpha:alpha];
    } else {
        return self;
    }
}

- (BOOL)isContrastingColor:(UIColor *)color {
    const CGFloat *bg = CGColorGetComponents(self.CGColor);
    const CGFloat *fg = CGColorGetComponents(color.CGColor);
    
    const CGFloat bgLum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2];
    const CGFloat fgLum = 0.2126 * fg[0] + 0.7152 * fg[1] + 0.0722 * fg[2];
    
    const BOOL bgGreater = bgLum > fgLum;
    const CGFloat nom = bgGreater ? bgLum : fgLum;
    const CGFloat denom = bgGreater ? fgLum : bgLum;
    const CGFloat contrast = (nom + 0.05) / (denom + 0.05);
    //    return 1.6 < contrast;
    return 2.4 < contrast;
}

- (UIColor *)darken {
    return [self darken:0.2];
}

- (UIColor *)darken:(CGFloat)percentage {
    CGFloat h=0, s=0, b=0, a=0;
    [self getHue:&h
      saturation:&s
      brightness:&b
           alpha:&a];
    
    b = b*(1-percentage);
    
    return [UIColor colorWithHue:h
                      saturation:s
                      brightness:b
                           alpha:a];
}

- (UIColor *)lighten {
    return [self lighten:0.2];
}

- (UIColor *)lighten:(CGFloat)percentage {
    CGFloat h=0, s=0, b=0, a=0;
    [self getHue:&h
      saturation:&s
      brightness:&b
           alpha:&a];
    
    b = b*(1+percentage);
    
    return [UIColor colorWithHue:h
                      saturation:s
                      brightness:b
                           alpha:a];
}

//+ (UIColor *)gb_pinkColor
//{
//    return [UIColor colorWithRed:206/255.0 green:67/255.0 blue:130/255.0 alpha:1];
//}
//
//+ (UIColor *)gb_yellowColor
//{
//    return [UIColor colorWithRed:253/255.0 green:197/255.0 blue:0/255.0 alpha:1];
//}
//
//+ (UIColor *)gb_orangeColor
//{
//    return [UIColor colorWithRed:255/255.0 green:167/255.0 blue:28/255.0 alpha:1];
//}
//
//+ (UIColor *)gb_greenColor
//{
//    return [UIColor colorWithRed:158/255.0 green:211/255.0 blue:15/255.0 alpha:1];
//}
//
//+ (UIColor *)gb_blueColor
//{
//    return [UIColor colorWithRed:100/255.0 green:194/255.0 blue:227/255.0 alpha:1];
//}
//
//+ (UIColor *)gb_purpleColor
//{
//    return [UIColor colorWithRed:124/255.0 green:118/255.0 blue:247/255.0 alpha:1];
//}
@end
