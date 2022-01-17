//
//  TOSegmentedControlItem.h
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

#import <Foundation/Foundation.h>

@class UIView;
@class UIImage;
@class UIImageView;
@class UILabel;
@class TOSegmentedControl;

NS_ASSUME_NONNULL_BEGIN

/**
 A private model object that holds all of the
 information and state for a single item in the
 segmented control.
 */
@interface TOSegmentedControlSegment : NSObject

/** When item is a label, the text to display */
@property (nonatomic, copy, nullable) NSString *title;

/** When item is an image, the image to display */
@property (nonatomic, strong, nullable) UIImage *image;

/** Whether the item can be tapped to toggle direction */
@property (nonatomic, assign) BOOL isReversible;

/** Whether the item is currently reveresed or not */
@property (nonatomic, assign) BOOL isReversed;

/** Whether this item is enabled or disabled. */
@property (nonatomic, assign) BOOL isDisabled;

/** Whether the item is selected or not. */
@property (nonatomic, assign) BOOL isSelected;

/** The view (either image or label) for this item */
@property (nonatomic, readonly) UIView *itemView;

/** If the item is a string, the subsequent label view (nil if an image) */
@property (nonatomic, nullable, readonly) UILabel *label;

/** If the item is an image, the subsequent image view (nil if a string) */
@property (nonatomic, nullable, readonly) UIImageView *imageView;

/** If the item is reversible, the subsequent arrow image view. */
@property (nonatomic, nullable, readonly) UIView *arrowView;

/// Create an array of objects given an array of strings and images
+ (NSArray *)segmentsWithObjects:(NSArray *)objects
             forSegmentedControl:(TOSegmentedControl *)segmentedControl;;

/// Create a non-reversible item from this class
- (nullable instancetype)initWithObject:(id)object
                    forSegmentedControl:(TOSegmentedControl *)segmentedControl;
- (instancetype)initWithTitle:(NSString *)title
          forSegmentedControl:(TOSegmentedControl *)segmentedControl;
- (instancetype)initWithImage:(UIImage *)image
          forSegmentedControl:(TOSegmentedControl *)segmentedControl;

/// Create a potentially reversible item from this class
- (instancetype)initWithTitle:(NSString *)title
                   reversible:(BOOL)reversible
          forSegmentedControl:(TOSegmentedControl *)segmentedControl;
- (instancetype)initWithImage:(UIImage *)image
                   reversible:(BOOL)reversible
          forSegmentedControl:(TOSegmentedControl *)segmentedControl ;

/// If the item is reversible, flip the direction
- (void)toggleDirection;

/// Re-synchronize the item view when the segmented control style changes
- (void)refreshItemView;

/// Rotates the arrow image view to 180 degrees and back again
- (void)setArrowImageReversed:(BOOL)reversed;

@end

NS_ASSUME_NONNULL_END
