//  Copyright © 2020 tencent. All rights reserved.

#import "TNNBoundingBox.h"

@interface  TNNBoundingBox ()
@property (nonatomic, strong) CAShapeLayer *boxLayer;
@property (nonatomic, strong) CATextLayer *textLayer;

@property (nonatomic, strong) NSArray<CAShapeLayer *> *markLayers;
@property (nonatomic, strong) NSArray<CAShapeLayer *> *lineLayers;
@end

@implementation TNNBoundingBox
- (instancetype)init {
    self = [super init];
    if (self != nil) {
        _boxLayer = [[CAShapeLayer alloc] init];
        _boxLayer.fillColor = [UIColor clearColor].CGColor;
        _boxLayer.lineWidth = 2;
        _boxLayer.hidden = YES;

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
        _lineLayers = [NSArray array];
    }
    return self;
}

- (void)addToLayer:(CALayer *)layer {
    [layer addSublayer:_boxLayer];
    [layer addSublayer:_textLayer];
    
    auto markLayers = _markLayers;
    for (CAShapeLayer * item in markLayers) {
        [layer addSublayer:item];
    }
    
    auto lineLayers = _lineLayers;
    for (CAShapeLayer * item in lineLayers) {
        [layer addSublayer:item];
    }
}

-(void)removeFromSuperLayer {
    [_boxLayer removeFromSuperlayer];
    [_textLayer removeFromSuperlayer];
    
    auto markLayers = _markLayers;
    for (CAShapeLayer * item in markLayers) {
        [item removeFromSuperlayer];
    }
    
    auto lineLayers = _lineLayers;
    for (CAShapeLayer * item in lineLayers) {
        [item removeFromSuperlayer];
    }
}

- (void)showText:(NSString *)text withColor:(UIColor *)color hideTextFrame:(bool)hideTextFrame atFrame:(CGRect)frame {
    [CATransaction setDisableActions:YES];
    
    auto path = [UIBezierPath bezierPathWithRect:frame];
    _boxLayer.path = path.CGPath;
    _boxLayer.strokeColor = color.CGColor;
    _boxLayer.hidden = hideTextFrame? YES : NO;

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
    
    [CATransaction setDisableActions:NO];
}

- (void)showMarkAtPoints:(std::vector<std::pair<float, float>>)points withColor:(UIColor *)color
                  circle:(BOOL)circle {
    [CATransaction setDisableActions:YES];
    
    NSMutableArray<CAShapeLayer *> *newMarkLayers = [NSMutableArray arrayWithArray:_markLayers];
    
    //add more layers if need
    for (auto i=_markLayers.count; i<points.size(); i++) {
        auto boxLayer = [[CAShapeLayer alloc] init];
        boxLayer.fillColor = [UIColor clearColor].CGColor;
        if (circle == YES)
            boxLayer.fillColor = color.CGColor;
        boxLayer.lineWidth = 1;
        boxLayer.hidden = YES;
        
        [newMarkLayers addObject:boxLayer];
    }
    
    auto super_layer = _boxLayer.superlayer;
    for (auto i=0; i<newMarkLayers.count; i++) {
        auto layer = newMarkLayers[i];
        if (layer.superlayer != super_layer) {
            [layer removeFromSuperlayer];
            [super_layer addSublayer:layer];
        }
        
        if (i < points.size()) {
            auto pt = points[i];
            auto path = [UIBezierPath bezierPath];
            if (circle == NO) {
                [path moveToPoint:CGPointMake(pt.first-2, pt.second)];
                [path addLineToPoint:CGPointMake(pt.first+2, pt.second)];
                [path moveToPoint:CGPointMake(pt.first, pt.second-2)];
                [path addLineToPoint:CGPointMake(pt.first, pt.second+2)];
            } else {
                CGPoint center = CGPointMake(pt.first, pt.second);
                [path addArcWithCenter:center radius:2 startAngle:0 endAngle:2 * M_PI clockwise:YES];
            }
            [path closePath];
            
            layer.path = path.CGPath;
            layer.strokeColor = color.CGColor;
            layer.hidden = NO;
        } else {
            layer.hidden = YES;
        }
    }
    _markLayers = newMarkLayers;
    
    [CATransaction setDisableActions:NO];
}

- (void)showLines:(std::vector<std::pair<float, float>>)points lines:(std::vector<std::pair<int, int>>)lines
        withColor:(UIColor *)color {
    [CATransaction setDisableActions:YES];
    
    NSMutableArray<CAShapeLayer *> *newLineLayers = [NSMutableArray arrayWithArray:_lineLayers];
    
    //add more layers if need
    for (auto i=_lineLayers.count; i<lines.size(); i++) {
        auto boxLayer = [[CAShapeLayer alloc] init];
        boxLayer.fillColor = [UIColor clearColor].CGColor;
        boxLayer.lineWidth = 1;
        boxLayer.hidden = YES;
        
        [newLineLayers addObject:boxLayer];
    }
    
    int line_cnt = 0;
    auto super_layer = _boxLayer.superlayer;
    for (auto i=0; i<newLineLayers.count; ++i) {
        auto layer = newLineLayers[i];
        if (layer.superlayer != super_layer) {
            [layer removeFromSuperlayer];
            [super_layer addSublayer:layer];
        }
        
        if (i < lines.size()) {
            auto line_start = lines[i].first;
            auto line_end = lines[i].second;
            
            if (line_start >= _markLayers.count || line_end >= _markLayers.count)
                continue;
            if (line_start >= points.size() || line_end >= points.size())
                continue;
            
            auto start_point = points[line_start];
            auto end_point = points[line_end];
            auto path = [UIBezierPath bezierPath];
            path.lineWidth = 4.0;
            [path moveToPoint:CGPointMake(start_point.first, start_point.second)];
            [path addLineToPoint:CGPointMake(end_point.first, end_point.second)];
            [path closePath];
            
            layer.path = path.CGPath;
            layer.strokeColor = color.CGColor;
            layer.hidden = NO;
            line_cnt += 1;
        } else {
            layer.hidden = YES;
        }
    }
    _lineLayers = newLineLayers;
    
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
    
    auto lineLayers = _lineLayers;
    for(CAShapeLayer * item in lineLayers) {
        item.hidden = YES;
    }
    
    [CATransaction setDisableActions:NO];
}
@end
