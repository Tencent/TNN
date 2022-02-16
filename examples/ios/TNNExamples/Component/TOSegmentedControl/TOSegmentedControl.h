//
//  TOSegmentedControl.h
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
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

/**
 A UI control that presents several
 options to the user in a horizontal, segmented layout.
 
 Only one segment may be selected at a time and, if desired,
 may be designated as 'reversible' with an arrow icon indicating
 its direction.
 */

NS_SWIFT_NAME(SegmentedControl)
IB_DESIGNABLE @interface TOSegmentedControl : UIControl

/** The items currently assigned to this segmented control. (Can be a combination of strings and images) */
@property (nonatomic, copy, nullable) NSArray *items;

/** A block that is called whenever a segment is tapped. */
@property (nonatomic, copy) void (^segmentTappedHandler)(NSInteger segmentIndex, BOOL reversed);

/** The number of segments this segmented control has. */
@property (nonatomic, readonly) NSInteger numberOfSegments;

/** The index of the currently segment. (May be manually set) */
@property (nonatomic, assign) NSInteger selectedSegmentIndex;

/** Whether the selected segment is also reveresed. */
@property (nonatomic, assign) BOOL selectedSegmentReversed;

/** The index values of all of the segments that are reversible. */
@property (nonatomic, strong) NSArray<NSNumber *> *reversibleSegmentIndexes;

/** The amount of rounding in the corners (Default is 9.0f) */
@property (nonatomic, assign) IBInspectable CGFloat cornerRadius;

/** Set the background color of the track in the segmented control (Default is light grey) */
@property (nonatomic, strong, null_resettable) IBInspectable UIColor *backgroundColor;

/** Set the color of the thumb view. (Default is white) */
@property (nonatomic, strong, null_resettable) IBInspectable UIColor *thumbColor;

/** Set the color of the separator lines between each item. (Default is dark grey) */
@property (nonatomic, strong, null_resettable) IBInspectable UIColor *separatorColor;

/** The color of the text labels / images (Default is black) */
@property (nonatomic, strong, null_resettable) IBInspectable UIColor *itemColor;

/** The color of the selected labels / images (Default is black) */
@property (nonatomic, strong, null_resettable) IBInspectable UIColor *selectedItemColor;

/** The font of the text items (Default is system default at 10 points) */
@property (nonatomic, strong, null_resettable) IBInspectable UIFont *textFont;

/** The font of the text item when it's been selected (Default is bold system default 10) */
@property (nonatomic, strong, null_resettable) IBInspectable UIFont *selectedTextFont;

/** The amount of insetting the thumb view is from the edge of the track (Default is 2.0f) */
@property (nonatomic, assign) IBInspectable CGFloat thumbInset;

/** The opacity of the shadow surrounding the thumb view*/
@property (nonatomic, assign) IBInspectable CGFloat thumbShadowOpacity;

/** The vertical offset of the shadow */
@property (nonatomic, assign) IBInspectable CGFloat thumbShadowOffset;

/** The radius of the shadow */
@property (nonatomic, assign) IBInspectable CGFloat thumbShadowRadius;

/**
 Creates a new segmented control with the provided items.

 @param items An array of either images, or strings to display
*/
- (instancetype)initWithItems:(nullable NSArray *)items NS_SWIFT_NAME(init(items:));

/**
 Replaces the content of an existing segment with a new image.

 @param image The image to set.
 @param index The index of the segment to set.
*/
- (void)setImage:(UIImage *)image forSegmentAtIndex:(NSInteger)index NS_SWIFT_NAME(set(_:forSegmentAt:));

/**
 Replaces the content of an existing segment with a new image,
 and optionally makes it reversible.

 @param image The image to set.
 @param reversible Whether the item can be tapped multiple times to flip directions.
 @param index The index of the segment to set.
*/
- (void)setImage:(UIImage *)image reversible:(BOOL)reversible
                              forSegmentAtIndex:(NSInteger)index NS_SWIFT_NAME(set(_:reversible:forSegmentAt:));

/**
 Returns the image that was assigned to a specific segment.
 Will return nil if the content at that segment is not an image.

 @param index The index at which the image is located.
*/
- (nullable UIImage *)imageForSegmentAtIndex:(NSInteger)index NS_SWIFT_NAME(image(forSegmentAt:));

/**
 Sets the content of a given segment to a text label.

 @param title The text to display at the segment.
 @param index The index of the segment to set.
*/
- (void)setTitle:(NSString *)title forSegmentAtIndex:(NSInteger)index NS_SWIFT_NAME(set(_:forSegmentAt:));

/**
 Sets the content of a given segment to a text label, and
 optionally makes it reversible.

 @param title The text to display at the segment.
 @param reversible Whether the item can be tapped multiple times to flip directions.
 @param index The index of the segment to set.
*/
- (void)setTitle:(NSString *)title reversible:(BOOL)reversible
                               forSegmentAtIndex:(NSInteger)index NS_SWIFT_NAME(set(_:reversible:forSegmentAt:));

/**
 Returns the string of the title that was assigned to a specific segment.
 Will return nil if the content at that segment is not a string.

 @param index The index at which the image is located.
*/
- (nullable NSString *)titleForSegmentAtIndex:(NSInteger)index NS_SWIFT_NAME(titleForSegment(for:));

/**
 Adds a new text segment to the end of the list.
 
 @param title The title of the new item.
*/
- (void)addSegmentWithTitle:(NSString *)title NS_SWIFT_NAME(addSegment(withTitle:));

/**
 Adds a new text segment to the end of the list, and optionally makes it reversible.
 
 @param title The title of the new item.
 @param reversible Whether the item is reversible or not.
*/
- (void)addSegmentWithTitle:(NSString *)title reversible:(BOOL)reversible NS_SWIFT_NAME(addSegment(withTitle:reversible:));

/**
 Adds a new image segment to the end of the list.
 
 @param image The image of the new item.
*/
- (void)addSegmentWithImage:(UIImage *)image NS_SWIFT_NAME(addSegment(with:));

/**
 Adds a new image segment to the end of the list, and optionally makes it reversible.
 
 @param image The image of the new item.
 @param reversible Whether the item is reversible or not.
*/
- (void)addSegmentWithImage:(UIImage *)image reversible:(BOOL)reversible NS_SWIFT_NAME(addSegment(with:reversible:));

/**
 Inserts a new image segment at the specified index.

 @param image The image to set.
 @param index The index of the segment to which the image will be set.
*/
- (void)insertSegmentWithImage:(UIImage *)image atIndex:(NSInteger)index NS_SWIFT_NAME(insertSegment(with:at:));

/**
 Inserts a new image segment at the specified segment index, and optionally makes it reversible.

 @param image The image to set.
 @param reversible Whether the item is reversible or not.
 @param index The index of the segment to which the image will be set.
*/
- (void)insertSegmentWithImage:(UIImage *)image reversible:(BOOL)reversible
                                                atIndex:(NSInteger)index NS_SWIFT_NAME(insertSegment(with:reversible:at:));

/**
 Inserts a new title segment at the specified index.

 @param title The title to set.
 @param index The index of the segment to which the image will be set.
*/
- (void)insertSegmentWithTitle:(NSString *)title atIndex:(NSInteger)index NS_SWIFT_NAME(insertSegment(withTitle:at:));

/**
 Inserts a new title segment at the specified index, and optionally makes it reversible.

 @param title The title to set.
 @param reversible Whether the item is reversible or not.
 @param index The index of the segment to which the image will be set.
*/
- (void)insertSegmentWithTitle:(NSString *)title reversible:(BOOL)reversible
                       atIndex:(NSInteger)index NS_SWIFT_NAME(insertSegment(withTitle:reversible:at:));

/**
 Remove the last segment in the list
*/
- (void)removeLastSegment NS_SWIFT_NAME(removeLastSegment());

/**
 Removes the segment at the specified index.

 @param index The index of the segment to remove.
*/
- (void)removeSegmentAtIndex:(NSInteger)index NS_SWIFT_NAME(removeSegment(at:));

/**
 Removes all of the items from this control.
*/
- (void)removeAllSegments NS_SWIFT_NAME(removeAllSegments());

/**
 Enables or disables the segment at the specified index.

 @param enabled Whether the segment is enabled or not.
 @param index The specific index to enable/disable.
*/
- (void)setEnabled:(BOOL)enabled forSegmentAtIndex:(NSInteger)index NS_SWIFT_NAME(setEnabled(_:forSegmentAt:));

/**
 Returns whether the segment at the specified index is currently enabled or not.

 @param index The index to check.
*/
- (BOOL)isEnabledForSegmentAtIndex:(NSInteger)index NS_SWIFT_NAME(isEnabledForSegment(at:));

/**
 Sets whether a specific segment is currently reversible or not.

 @param reversible Whether the segment is reversible or not.
 @param index The specific index to enable/disable.
*/
- (void)setReversible:(BOOL)reversible forSegmentAtIndex:(NSInteger)index NS_SWIFT_NAME(setReversible(_:forSegmentAt:));

/**
 Returns whether the segment at the specified index is reversible or not.

 @param index The index to check.
*/
- (BOOL)isReversibleForSegmentAtIndex:(NSInteger)index NS_SWIFT_NAME(isReversibleForSegment(at:));

/**
 Sets whether a specific segment is currently in a reversed state or not.

 @param reversed Whether the segment is currently reversed or not.
 @param index The specific index to enable/disable.
*/
- (void)setReversed:(BOOL)reversed forSegmentAtIndex:(NSInteger)index NS_SWIFT_NAME(setReversed(_:forSegmentAt:));

/**
 Returns whether the segment at the specified index is currently reversed or not.

 @param index The index to check.
*/
- (BOOL)isReversedForSegmentAtIndex:(NSInteger)index NS_SWIFT_NAME(isReversed(at:));

/**
 Sets which segment is currently selected, and optionally play an animation during the transition.

 @param selectedSegmentIndex The index of the segment to select.
 @param animated Whether the transition to the newly selected index is animated or not.
*/
- (void)setSelectedSegmentIndex:(NSInteger)selectedSegmentIndex animated:(BOOL)animated NS_SWIFT_NAME(setSelectedSegmentIndex(_:animated:));

@end

NS_ASSUME_NONNULL_END

FOUNDATION_EXPORT double TOSegmentedControlFrameworkVersionNumber;
FOUNDATION_EXPORT const unsigned char TOSegmentedControlFrameworkVersionString[];
