//
//  TOSegmentedControlItem.m
//
//  Copyright 2019 Timothy Oliver. All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//  sell copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
//  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
//  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#import "TOSegmentedControlSegment.h"
#import "TOSegmentedControl.h"
#import <UIKit/UIKit.h>

// -------------------------------------------------
// Access to private properties in segmented control parent

@interface TOSegmentedControl ()
@property (nonatomic, readonly) UIView *trackView;
@property (nonatomic, readonly) UIImage *arrowImage;
@end

// -------------------------------------------------
// Private Interface

@interface TOSegmentedControlSegment ()

// Weak reference to our parent segmented control
@property (nonatomic, weak) TOSegmentedControl *segmentedControl;

// Read-write access to the item view
@property (nonatomic, strong, readwrite) UIView *itemView;

// When made reversible, the container for arrow image view to show
@property (nonatomic, strong, readwrite) UIView *arrowView;

// When made reversible, the arrow image view to show
@property (nonatomic, strong) UIImageView *arrowImageView;

@end

@implementation TOSegmentedControlSegment

#pragma mark - Object Lifecyle -

- (instancetype)initWithObject:(id)object
           forSegmentedControl:(TOSegmentedControl *)segmentedControl
{
    if (![object isKindOfClass:NSString.class] && ![object isKindOfClass:UIImage.class]) {
        return nil;
    }
    
    if (self = [super init]) {
        if ([object isKindOfClass:NSString.class]) {
            _title = (NSString *)object;
        }
        else {
            _image = (UIImage *)object;
        }
        _segmentedControl = segmentedControl;
        [self commonInit];
    }
    
    return self;
}

- (instancetype)initWithTitle:(NSString *)title
          forSegmentedControl:(nonnull TOSegmentedControl *)segmentedControl
{
    if (self = [super init]) {
        _title = [title copy];
        _segmentedControl = segmentedControl;
        [self commonInit];
    }
    
    return self;
}

- (instancetype)initWithImage:(UIImage *)image
          forSegmentedControl:(nonnull TOSegmentedControl *)segmentedControl
{
    if (self = [super init]) {
        _image = image;
        _segmentedControl = segmentedControl;
        [self commonInit];
    }
    
    return self;
}

- (instancetype)initWithTitle:(NSString *)title
                   reversible:(BOOL)reversible
          forSegmentedControl:(nonnull TOSegmentedControl *)segmentedControl
{
    if (self = [super init]) {
        _title = [title copy];
        _isReversible = reversible;
        _segmentedControl = segmentedControl;
        [self commonInit];
    }
    
    return self;
}

- (instancetype)initWithImage:(UIImage *)image
                   reversible:(BOOL)reversible
          forSegmentedControl:(nonnull TOSegmentedControl *)segmentedControl
{
    if (self = [super init]) {
        _image = image;
        _isReversible = reversible;
        _segmentedControl = segmentedControl;
        [self commonInit];
    }
    
    return self;
}

- (void)dealloc
{
    [self.itemView removeFromSuperview];
}

#pragma mark - Comnvenience Initializers -

+ (NSArray *)segmentsWithObjects:(NSArray *)objects forSegmentedControl:(nonnull TOSegmentedControl *)segmentedControl
{
    NSMutableArray *array = [NSMutableArray array];
    
    // Create an object for each item in the array.
    // Skip anything that isn't an image or a label
    for (id object in objects) {
        TOSegmentedControlSegment *item = nil;
        if ([object isKindOfClass:NSString.class]) {
            item = [[TOSegmentedControlSegment alloc] initWithTitle:object
                                             forSegmentedControl:segmentedControl];
        }
        else if ([object isKindOfClass:UIImage.class]) {
            item = [[TOSegmentedControlSegment alloc] initWithImage:object
                                             forSegmentedControl:segmentedControl];
        }

        if (item) { [array addObject:item]; }
    }
    
    return [NSArray arrayWithArray:array];
}

#pragma mark - Set-up -

- (void)commonInit
{    
    // Create the initial image / label view
    [self refreshItemView];
    
    // Refresh the reversible state
    [self refreshReversibleView];
}

#pragma mark - Reversible Management -

- (void)refreshReversibleView
{
    // If we're no longer (Or never were) reversible,
    // hide and exit out
    if (!self.isReversible) {
        self.arrowView.hidden = YES;
        return;
    }
    
    // Create the arrow view if we haven't done so yet
    if (self.arrowView == nil && self.arrowImageView == nil) {
        UIImage *arrow = self.segmentedControl.arrowImage;
        self.arrowView = [[UIView alloc] initWithFrame:(CGRect){CGPointZero, arrow.size}];
        self.arrowView.alpha = 0.0f;
        [self.segmentedControl.trackView addSubview:self.arrowView];

        self.arrowImageView = [[UIImageView alloc] initWithImage:arrow];
        self.arrowImageView.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
        [self.arrowView addSubview:self.arrowImageView];
    }

    // Perform these updates, but never animate them
    [UIView performWithoutAnimation:^{
        // Set the tint color
        self.arrowView.tintColor = self.segmentedControl.itemColor;
    }];
}

#pragma mark - View Management -

- (UILabel *)makeLabelForTitle:(NSString *)title
{
    if (title.length == 0) { return nil; }
    
    // Object is a string. Create a label
    UILabel *label = [[UILabel alloc] init];
    label.text = title;
    label.textAlignment = NSTextAlignmentCenter;
    label.textColor = self.segmentedControl.itemColor;
    label.font = self.segmentedControl.selectedTextFont;
    label.adjustsFontSizeToFitWidth = YES;
    label.minimumScaleFactor = 0.3f;
    [label sizeToFit]; // Size to the selected font
    label.font = self.segmentedControl.textFont;
    label.backgroundColor = [UIColor clearColor];
    return label;
}

- (UIImageView *)makeImageViewForImage:(UIImage *)image
{
    // Object is an image. Create an image view
    UIImageView *imageView = [[UIImageView alloc] initWithImage:image];
    imageView.tintColor = self.segmentedControl.itemColor;
    return imageView;
}

- (void)refreshItemView
{
    // Convenience check for whether the view is a label or image
    UIImageView *imageView = self.imageView;
    UILabel *label = self.label;

    // If we didn't change the type, just update the current
    // view with the new type
    if (imageView && self.image) {
        [(UIImageView *)self.itemView setImage:self.image];
    }
    
    // If it's already a label, refresh the text
    if (label && self.title) {
        [(UILabel *)self.itemView setText:self.title];
    }
    
    // If it's an image view, but the title text is set, swap them out
    if (!label && self.title) {
        [imageView removeFromSuperview];
        imageView = nil;
        
        self.itemView = [self makeLabelForTitle:self.title];
        [self.segmentedControl.trackView addSubview:self.itemView];
        
        label = (UILabel *)self.itemView;
    }
    
    // If it's a label view, but the image is set, swap them out
    if (!imageView && self.image) {
        [label removeFromSuperview];
        label = nil;
        
        self.itemView = [self makeImageViewForImage:self.image];
        [self.segmentedControl.trackView addSubview:self.itemView];
        
        imageView = (UIImageView *)self.itemView;
    }
    
    // Update the label view
    label.textColor = self.segmentedControl.itemColor;
    // Set the frame off the selected text as it is larger
    label.font = self.segmentedControl.selectedTextFont;
    [label sizeToFit];

    // Set back to default font
    label.font = self.segmentedControl.textFont;
    
    // Update the image view
    imageView.tintColor = self.segmentedControl.itemColor;
}

- (void)toggleDirection
{
    self.isReversed = !self.isReversed;
}

#pragma mark - Public Accessors -

- (void)setTitle:(NSString *)title
{
    // Copy text, and regenerate the view if need be
    _title = [title copy];
    _image = nil;
    [self refreshItemView];
}

- (void)setImage:(UIImage *)image
{
    if (_image == image) { return; }
    _image = image;
    _title = nil;
    [self refreshItemView];
}

- (void)setIsReversible:(BOOL)isReversible
{
    if (_isReversible == isReversible) { return; }
    _isReversible = isReversible;
    [self refreshReversibleView];
}

- (UILabel *)label
{
    if ([self.itemView isKindOfClass:UILabel.class]) {
        return (UILabel *)self.itemView;
    }
    
    return nil;
}

- (UIImageView *)imageView
{
    if ([self.itemView isKindOfClass:UIImageView.class]) {
        return (UIImageView *)self.itemView;
    }
    
    return nil;
}

- (void)setArrowImageReversed:(BOOL)reversed
{
    if (reversed) {
        self.arrowImageView.transform = CGAffineTransformRotate(CGAffineTransformIdentity, M_PI);
        return;
    }

    self.arrowImageView.transform = CGAffineTransformIdentity;
}

@end
