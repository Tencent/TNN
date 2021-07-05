//  Copyright © 2020 tencent. All rights reserved.

#import "TNNMaskImage.h"

@interface  TNNMaskImage()
@property (nonatomic, strong) CALayer *imageLayer;
@end

@implementation TNNMaskImage
- (instancetype)init {
    self = [super init];
    if (self != nil) {
        _imageLayer = [[CAShapeLayer alloc] init];
        _imageLayer.hidden = YES;
    }
    return self;
}

- (void)addToLayer:(CALayer *)layer {
    [layer addSublayer:_imageLayer];
}

-(void)removeFromSuperLayer {
    [_imageLayer removeFromSuperlayer];
}

- (void)showImage:(UIImage *)image atFrame:(CGRect)frame {
    [CATransaction setDisableActions:YES];
    
    _imageLayer.frame = frame;
    _imageLayer.contents = (id)image.CGImage;
    
    _imageLayer.hidden = NO;
    
    [CATransaction setDisableActions:NO];
}

- (void)hide {
    [CATransaction setDisableActions:YES];
    
    _imageLayer.hidden = YES;
    
    [CATransaction setDisableActions:NO];
}
@end

