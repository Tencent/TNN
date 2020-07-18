//  Copyright © 2020 tencent. All rights reserved.

#import "TNNBoundingBox.h"

@implementation TNNBoundingBox
- (instancetype)init {
    self = [super init];
    if (self != nil) {
        _shapeLayer = [[CAShapeLayer alloc] init];
        _shapeLayer.fillColor = [UIColor clearColor].CGColor;
        _shapeLayer.lineWidth = 2;
        _shapeLayer.hidden = YES;

        _textLayer =[[CATextLayer alloc] init];
        _textLayer.foregroundColor = [UIColor blackColor].CGColor;
        _textLayer.hidden = YES;
        _textLayer.contentsScale = [UIScreen mainScreen].scale;
        _textLayer.fontSize = 14;
        {
            auto font = [UIFont systemFontOfSize:14];
            CFStringRef fontName = (__bridge CFStringRef)font.fontName;
            CGFontRef fontRef = CGFontCreateWithFontName(fontName);
            _textLayer.font = fontRef;
            CGFontRelease(fontRef);
        }

        _textLayer.alignmentMode = kCAAlignmentCenter;
    }
    return self;
}

- (void)addToLayer:(CALayer *)layer {
    [layer addSublayer:_shapeLayer];
    [layer addSublayer:_textLayer];
}

-(void)removeFromSuperLayer {
    [_shapeLayer removeFromSuperlayer];
    [_textLayer removeFromSuperlayer];
}

- (void)showText:(NSString *)text withColor:(UIColor *)color atFrame:(CGRect)frame {
    [CATransaction setDisableActions:YES];
    
    auto path = [UIBezierPath bezierPathWithRect:frame];
    _shapeLayer.path = path.CGPath;
    _shapeLayer.strokeColor = color.CGColor;
    _shapeLayer.hidden = NO;

    _textLayer.string = text;
    _textLayer.backgroundColor = color.CGColor;
    _textLayer.hidden = NO;

    auto attributes = @{NSFontAttributeName:[UIFont systemFontOfSize:14]};

    auto textRect = [text boundingRectWithSize:CGSizeMake(400, 100)
                                       options:NSStringDrawingTruncatesLastVisibleLine
                                    attributes:attributes
                                       context:nil];
    
    _textLayer.frame = CGRectMake(frame.origin.x - 1,
                                  frame.origin.y - textRect.size.height,
                                  textRect.size.width + 10,
                                  textRect.size.height);
}

- (void)hide {
    _shapeLayer.hidden = YES;
    _textLayer.hidden = YES;
}
@end
