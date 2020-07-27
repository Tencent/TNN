//  Copyright © 2020 tencent. All rights reserved.

#import "TNNBoundingBox.h"

@interface  TNNBoundingBox ()
@property (nonatomic, strong) CAShapeLayer *boxLayer;
@property (nonatomic, strong) CATextLayer *textLayer;

@property (nonatomic, strong) NSArray<CAShapeLayer *> *markLayers;
@end

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
        
        _markLayers = [NSArray array];
    }
    return self;
}

- (void)addToLayer:(CALayer *)layer {
    [layer addSublayer:_boxLayer];
    [layer addSublayer:_textLayer];
}

-(void)removeFromSuperLayer {
    [_boxLayer removeFromSuperlayer];
    [_textLayer removeFromSuperlayer];
}

- (void)showText:(NSString *)text withColor:(UIColor *)color atFrame:(CGRect)frame {
    [CATransaction setDisableActions:YES];
    
    auto path = [UIBezierPath bezierPath];
    [path moveToPoint:CGPointMake(frame.origin.x-2, frame.origin.y)];
    [path addLineToPoint:CGPointMake(frame.origin.x+2, frame.origin.y)];
    [path moveToPoint:CGPointMake(frame.origin.x, frame.origin.y-2)];
    [path addLineToPoint:CGPointMake(frame.origin.x, frame.origin.y+2)];
    [path closePath];
    
//    auto path = [UIBezierPath bezierPathWithRect:frame];
    _boxLayer.path = path.CGPath;
    _boxLayer.strokeColor = color.CGColor;
    _boxLayer.hidden = NO;

    _textLayer.string = text;
    _textLayer.backgroundColor = color.CGColor;
    _textLayer.hidden = YES;

    auto attributes = @{NSFontAttributeName:[UIFont systemFontOfSize:14]};

    auto textRect = [text boundingRectWithSize:CGSizeMake(400, 100)
                                       options:NSStringDrawingTruncatesLastVisibleLine
                                    attributes:attributes
                                       context:nil];
    
    _textLayer.frame = CGRectMake(frame.origin.x - 1,
                                  frame.origin.y - textRect.size.height,
                                  textRect.size.width + 10,
                                  textRect.size.height);
    
    [CATransaction setDisableActions:NO];
}

- (void)showMarkAtPoints:(std::vector<std::pair<float, float>>)points withColor:(UIColor *)color {
    [CATransaction setDisableActions:YES];
    
    NSMutableArray<CAShapeLayer *> *newMarkLayers = [NSMutableArray arrayWithArray:_markLayers];
    
    //add more layers if need
    for (auto i=_markLayers.count; i<points.size(); i++) {
        auto boxLayer = [[CAShapeLayer alloc] init];
        boxLayer.fillColor = [UIColor clearColor].CGColor;
        boxLayer.lineWidth = 1;
        boxLayer.hidden = YES;
        
        [newMarkLayers addObject:boxLayer];
    }
    
    for (auto i=0; i<newMarkLayers.count; i++) {
        auto layer = newMarkLayers[i];
        if (i < points.size()) {
            auto pt = points[i];
            auto path = [UIBezierPath bezierPath];
            [path moveToPoint:CGPointMake(pt.first-2, pt.second)];
            [path addLineToPoint:CGPointMake(pt.first+2, pt.second)];
            [path moveToPoint:CGPointMake(pt.first, pt.second-2)];
            [path addLineToPoint:CGPointMake(pt.first, pt.second+2)];
            [path closePath];
            
            layer.path = path.CGPath;
            layer.strokeColor = color.CGColor;
            layer.hidden = NO;
        } else {
            layer.hidden = YES;
        }
    }
    
    [CATransaction setDisableActions:NO];
}

- (void)hide {
    [CATransaction setDisableActions:YES];
    
    _boxLayer.hidden = YES;
    _textLayer.hidden = YES;
    
    auto markLayers = _markLayers;
    for (CAShapeLayer * item in markLayers) {
        item.hidden = YES;
    }
    
    [CATransaction setDisableActions:NO];
}
@end
