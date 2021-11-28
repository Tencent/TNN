//
//  TOSegmentedControl.m
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

#import "TOSegmentedControl.h"
#import "TOSegmentedControlSegment.h"

// ----------------------------------------------------------------
// Static Members

// A cache to hold images generated for this view that may be shared.
static NSMapTable *_imageTable = nil;

// Statically referenced key names for the images stored in the map table.
static NSString * const kTOSegmentedControlArrowImage = @"arrowIcon";
static NSString * const kTOSegmentedControlSeparatorImage = @"separatorImage";

// When tapped the amount the focused elements will shrink / fade
static CGFloat const kTOSegmentedControlSelectedTextAlpha = 0.3f;
static CGFloat const kTOSegmentedControlDisabledAlpha = 0.4f;
static CGFloat const kTOSegmentedControlSelectedScale = 0.95f;
static CGFloat const kTOSegmentedControlDirectionArrowAlpha = 0.4f;
static CGFloat const kTOSegmentedControlDirectionArrowMargin = 2.0f;

// ----------------------------------------------------------------
// Private Members

@interface TOSegmentedControl ()

/** The private list of item objects, storing state and view data */
@property (nonatomic, strong) NSMutableArray<TOSegmentedControlSegment *> *segments;

/** Keep track when the user taps explicitily on the thumb view */
@property (nonatomic, assign) BOOL isDraggingThumbView;

/** Track if the user drags the thumb off the original segment. This disables reversing. */
@property (nonatomic, assign) BOOL didDragOffOriginalSegment;

/** Before we commit to a new selected index, this is the index the user has dragged over */
@property (nonatomic, assign) NSInteger focusedIndex;

/** The background rounded "track" view */
@property (nonatomic, strong) UIView *trackView;

/** The view that shows which view is highlighted */
@property (nonatomic, strong) UIView *thumbView;

/** The separator views between each of the items */
@property (nonatomic, strong) NSMutableArray<UIView *> *separatorViews;

/** A weakly retained image table that holds cached images for us. */
@property (nonatomic, readonly) NSMapTable *imageTable;

/** An arrow icon used to denote when a view is reversible. */
@property (nonatomic, readonly) UIImage *arrowImage;

/** A rounded line used as the separator line. */
@property (nonatomic, readonly) UIImage *separatorImage;

/** The width of each segment */
@property (nonatomic, readonly) CGFloat segmentWidth;

/** Convenience property for testing if there are no segments */
@property (nonatomic, readonly) BOOL hasNoSegments;

@end

@implementation TOSegmentedControl

#pragma mark - Class Init -

- (instancetype)initWithItems:(NSArray *)items
{
    if (self = [super initWithFrame:(CGRect){0.0f, 0.0f, 300.0f, 32.0f}]) {
        [self commonInit];
        self.items = [self sanitizedItemArrayWithItems:items];
    }

    return self;
}

- (instancetype)initWithCoder:(NSCoder *)coder
{
    if (self = [super initWithCoder:coder]) {
        [self commonInit];
    }

    return self;
}

- (instancetype)initWithFrame:(CGRect)frame
{
    if (self = [super initWithFrame:frame]) {
        [self commonInit];
    }

    return self;
}

- (instancetype)init
{
    if (self = [super initWithFrame:(CGRect){0.0f, 0.0f, 300.0f, 32.0f}]) {
        [self commonInit];
    }

    return self;
}

- (void)commonInit
{
    // Create content view
    self.trackView = [[UIView alloc] initWithFrame:self.bounds];
    self.trackView.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    self.trackView.layer.masksToBounds = YES;
    self.trackView.userInteractionEnabled = NO;
    #ifdef __IPHONE_13_0
    if (@available(iOS 13.0, *)) { self.trackView.layer.cornerCurve = kCACornerCurveContinuous; }
    #endif
    [self addSubview:self.trackView];

    // Create thumb view
    self.thumbView = [[UIView alloc] initWithFrame:CGRectMake(2.0f, 2.0f, 100.0f, 28.0f)];
    self.thumbView.layer.shadowColor = [UIColor blackColor].CGColor;
    #ifdef __IPHONE_13_0
    if (@available(iOS 13.0, *)) { self.thumbView.layer.cornerCurve = kCACornerCurveContinuous; }
    #endif
    [self.trackView addSubview:self.thumbView];

    // Create list for managing each item
    self.segments = [NSMutableArray array];
    
    // Create containers for views
    self.separatorViews = [NSMutableArray array];
    
    // Set default resettable values
    self.backgroundColor = nil;
    self.thumbColor = nil;
    self.separatorColor = nil;
    self.itemColor = nil;
    self.selectedItemColor = nil;
    self.textFont = nil;
    self.selectedTextFont = nil;

    // Set default values
    self.selectedSegmentIndex = -1;
    self.cornerRadius = 8.0f;
    self.thumbInset = 2.0f;
    self.thumbShadowRadius = 3.0f;
    self.thumbShadowOffset = 2.0f;
    self.thumbShadowOpacity = 0.13f;
    
    // Configure view interaction
    // When the user taps down in the view
    [self addTarget:self
             action:@selector(didTapDown:withEvent:)
   forControlEvents:UIControlEventTouchDown];
    
    // When the user drags, either inside or out of the view
    [self addTarget:self
             action:@selector(didDragTap:withEvent:)
   forControlEvents:UIControlEventTouchDragInside|UIControlEventTouchDragOutside];
    
    // When the user's finger leaves the bounds of the view
    [self addTarget:self
             action:@selector(didExitTapBounds:withEvent:)
   forControlEvents:UIControlEventTouchDragExit];
    
    // When the user's finger re-enters the bounds
    [self addTarget:self
             action:@selector(didEnterTapBounds:withEvent:)
   forControlEvents:UIControlEventTouchDragEnter];
    
    // When the user taps up, either inside or out
    [self addTarget:self
             action:@selector(didEndTap:withEvent:)
   forControlEvents:UIControlEventTouchUpInside|UIControlEventTouchUpOutside|UIControlEventTouchCancel];
}

#pragma mark - Item Management -

- (NSMutableArray *)sanitizedItemArrayWithItems:(NSArray *)items
{
    // Filter the items to extract only strings and images
    NSMutableArray *sanitizedItems = [NSMutableArray array];
    for (id item in items) {
        if (![item isKindOfClass:[UIImage class]] && ![item isKindOfClass:[NSString class]]) {
            continue;
        }
        [sanitizedItems addObject:item];
    }

    return sanitizedItems;
}

- (void)updateSeparatorViewCount
{
    // Work out how many separators we need (One less than segments)
    NSInteger numberOfSeparators = self.segments.count - 1;

    // Cap the number at 0 if there were no segments
    numberOfSeparators = MAX(0, numberOfSeparators);

    // Add as many separators as needed
    while (self.separatorViews.count < numberOfSeparators) {
        UIImageView *separator = [[UIImageView alloc] initWithImage:self.separatorImage];
        separator.tintColor = self.separatorColor;
        [self.trackView insertSubview:separator atIndex:0];
        [self.separatorViews addObject:separator];
    }

    // Substract as many separators as needed
    while (self.separatorViews.count > numberOfSeparators) {
        UIView *separator = self.separatorViews.lastObject;
        [self.separatorViews removeLastObject];
        [separator removeFromSuperview];
    }
}

#pragma mark - Public Item Access -

- (nullable UIImage *)imageForSegmentAtIndex:(NSInteger)index
{
    if (index < 0 || index >= self.segments.count) { return nil; }
    return [self objectForSegmentAtIndex:index class:UIImage.class];
}

- (nullable NSString *)titleForSegmentAtIndex:(NSInteger)index
{
    return [self objectForSegmentAtIndex:index class:NSString.class];
}

- (nullable id)objectForSegmentAtIndex:(NSInteger)index class:(Class)class
{
    // Make sure the index provided is valid
    if (index < 0 || index >= self.items.count) { return nil; }

    // Return the item only if it is an image
    id item = self.items[index];
    if ([item isKindOfClass:class]) { return item; }
    
    // Return nil if a label or anything else
    return nil;
}

#pragma mark Add New Items

- (void)addSegmentWithImage:(UIImage *)image
{
    [self addSegmentWithImage:image reversible:NO];
}

- (void)addSegmentWithImage:(UIImage *)image reversible:(BOOL)reversible
{
    [self addSegmentWithObject:image reversible:reversible];
}

- (void)addSegmentWithTitle:(NSString *)title
{
    [self addSegmentWithTitle:title reversible:NO];
}

- (void)addSegmentWithTitle:(NSString *)title reversible:(BOOL)reversible
{
    [self addSegmentWithObject:title reversible:reversible];
}

- (void)addSegmentWithObject:(id)object reversible:(BOOL)reversible
{
    [self insertSegmentWithObject:object reversible:reversible atIndex:self.segments.count];
}

#pragma mark Inserting New Items

- (void)insertSegmentWithTitle:(NSString *)title atIndex:(NSInteger)index
{
    [self insertSegmentWithTitle:title reversible:NO atIndex:index];
}

- (void)insertSegmentWithTitle:(NSString *)title reversible:(BOOL)reversible atIndex:(NSInteger)index
{
    [self insertSegmentWithObject:title reversible:reversible atIndex:index];
}

- (void)insertSegmentWithImage:(UIImage *)image atIndex:(NSInteger)index
{
    [self insertSegmentWithImage:image reversible:NO atIndex:index];
}

- (void)insertSegmentWithImage:(UIImage *)image reversible:(BOOL)reversible atIndex:(NSInteger)index
{
    [self insertSegmentWithObject:image reversible:reversible atIndex:index];
}

- (void)insertSegmentWithObject:(id)object reversible:(BOOL)reversible atIndex:(NSInteger)index
{
    // If an invalid index was provided, cap it to the available range
    if (index < 0) { index = 0; }
    if (index >= self.segments.count) { index = self.segments.count; }

    // Add item to master list (Create a new list if one didn't exist)
    NSMutableArray *items = nil;
    if (self.items) { items = [self.items mutableCopy]; }
    else { items = [NSMutableArray array]; }
    [items insertObject:object atIndex:index];
    _items = [NSArray arrayWithArray:items];

    // Add new item object to internal list
    TOSegmentedControlSegment *segment = [[TOSegmentedControlSegment alloc] initWithObject:object
                                                                       forSegmentedControl:self];
    segment.isReversible = reversible;
    [self.segments insertObject:segment atIndex:index];

    // Update number of separators
    [self updateSeparatorViewCount];

    // Perform new layout update
    [self setNeedsLayout];
}

#pragma mark Replacing Items

- (void)setImage:(UIImage *)image forSegmentAtIndex:(NSInteger)index
{
    [self setImage:image reversible:NO forSegmentAtIndex:index];
}

- (void)setImage:(UIImage *)image reversible:(BOOL)reversible forSegmentAtIndex:(NSInteger)index
{
    [self setObject:image reversible:reversible forSegmentAtIndex:index];
}

- (void)setTitle:(NSString *)title forSegmentAtIndex:(NSInteger)index
{
    [self setTitle:title reversible:NO forSegmentAtIndex:index];
}

- (void)setTitle:(NSString *)title reversible:(BOOL)reversible forSegmentAtIndex:(NSInteger)index
{
    [self setObject:title reversible:reversible forSegmentAtIndex:index];
}

- (void)setObject:(id)object reversible:(BOOL)reversible forSegmentAtIndex:(NSInteger)index
{
    NSAssert([object isKindOfClass:NSString.class] || [object isKindOfClass:UIImage.class],
                @"TOSegmentedControl: Only images and strings are supported.");
    
    // Make sure we don't go out of bounds
    if (index < 0 || index >= self.items.count) { return; }
    
    // Remove the item from the item list and insert the new one
    NSMutableArray *items = [self.items mutableCopy];
    [items removeObjectAtIndex:index];
    [items insertObject:object atIndex:index];
    _items = [NSArray arrayWithArray:items];
    
    // Update the item object at that point for the new item
    TOSegmentedControlSegment *segment = self.segments[index];
    if ([object isKindOfClass:NSString.class]) { segment.title = object; }
    if ([object isKindOfClass:UIImage.class]) { segment.image = object; }
    segment.isReversible = reversible;
    
    // Re-layout the views
    [self setNeedsLayout];
}

#pragma mark Deleting Items

- (void)removeLastSegment
{
    [self removeSegmentAtIndex:self.items.count - 1];
}

- (void)removeSegmentAtIndex:(NSInteger)index
{
    if (index < 0 || index >= self.items.count) { return; }

    // Remove from the item list
    NSMutableArray *items = self.items.mutableCopy;
    [items removeObjectAtIndex:index];
    _items = items;

    // Remove item object
    [self.segments removeObjectAtIndex:index];

    // Update number of separators
    [self updateSeparatorViewCount];
}

- (void)removeAllSegments
{
    // Remove all item objects
    self.segments = [NSMutableArray array];

    // Remove all separators
    for (UIView *separator in self.separatorViews) {
        [separator removeFromSuperview];
    }
    [self.separatorViews removeAllObjects];

    // Delete the items array
    _items = nil;
}

#pragma mark Enabled/Disabled

- (void)setEnabled:(BOOL)enabled forSegmentAtIndex:(NSInteger)index
{
    if (index < 0 || index >= self.segments.count) { return; }
    self.segments[index].isDisabled = !enabled;
    [self setNeedsLayout];

    // If we disabled the selected index, choose another one
    if (self.selectedSegmentIndex >= 0 && !self.segments[self.selectedSegmentIndex].isDisabled) {
        return;
    }

    // Loop ahead of the selected segment index to find the next enabled one
    for (NSInteger i = self.selectedSegmentIndex; i < self.segments.count; i++) {
        if (self.segments[i].isDisabled) { continue; }
        self.selectedSegmentIndex = i;
        return;
    }

    // If that failed, loop forward to find an enabled one before it
    for (NSInteger i = self.selectedSegmentIndex; i >= 0; i--) {
        if (self.segments[i].isDisabled) { continue; }
        self.selectedSegmentIndex = i;
        return;
    }

    // Nothing is enabled, default back to deselecting everything
    self.selectedSegmentIndex = -1;
}

- (BOOL)isEnabledForSegmentAtIndex:(NSInteger)index
{
    if (index < 0 || index >= self.segments.count) { return NO; }
    return !self.segments[index].isDisabled;
}

#pragma mark - Reversible Management -

// Accessors for setting when a segment is reversible.

- (void)setReversible:(BOOL)reversible forSegmentAtIndex:(NSInteger)index
{
    if (index < 0 || index >= self.segments.count) { return; }
    self.segments[index].isReversible = reversible;
}

- (BOOL)isReversibleForSegmentAtIndex:(NSInteger)index
{
    if (index < 0 || index >= self.segments.count) { return NO; }
    return !self.segments[index].isReversible;
}

// Accessors for toggling whether a reversible segment is currently reversed.
- (void)setReversed:(BOOL)reversed forSegmentAtIndex:(NSInteger)index
{
    if (index < 0 || index >= self.segments.count) { return; }
    self.segments[index].isReversed = reversed;
}

- (BOOL)isReversedForSegmentAtIndex:(NSInteger)index
{
    if (index < 0 || index >= self.segments.count) { return NO; }
    return !self.segments[index].isReversed;
}

#pragma mark - View Layout -

- (void)layoutThumbView
{
    // Hide the thumb view if no segments are selected
    if (self.selectedSegmentIndex < 0 || !self.enabled) {
        self.thumbView.hidden = YES;
        return;
    }

    // Lay-out the thumb view
    CGRect frame = [self frameForSegmentAtIndex:self.selectedSegmentIndex];
    self.thumbView.frame = frame;
    self.thumbView.hidden = NO;

    // Match the shadow path to the new size of the thumb view
    CGPathRef oldShadowPath = self.thumbView.layer.shadowPath;
    UIBezierPath *shadowPath = [UIBezierPath bezierPathWithRoundedRect:(CGRect){CGPointZero, frame.size}
                                                          cornerRadius:self.cornerRadius - self.thumbInset];

    // If the segmented control is animating its shape, to prevent the
    // shadow from visibly snapping, perform a resize animation on it
    CABasicAnimation *boundsAnimation = [self.layer animationForKey:@"bounds.size"];
    if (oldShadowPath != NULL && boundsAnimation) {
        CABasicAnimation *shadowAnimation = [CABasicAnimation animationWithKeyPath:@"shadowPath"];
        shadowAnimation.fromValue = (__bridge id)oldShadowPath;
        shadowAnimation.toValue = (id)shadowPath.CGPath;
        shadowAnimation.duration = boundsAnimation.duration;
        shadowAnimation.timingFunction = boundsAnimation.timingFunction;
        [self.thumbView.layer addAnimation:shadowAnimation forKey:@"shadowPath"];
    }
    self.thumbView.layer.shadowPath = shadowPath.CGPath;
}

- (void)layoutItemViews
{
    // Lay out the item views
    NSInteger i = 0;
    for (TOSegmentedControlSegment *item in self.segments) {
        UIView *itemView = item.itemView;
        [itemView sizeToFit];
        [self.trackView addSubview:itemView];

        // Get the container frame that the item will be aligned with
        CGRect thumbFrame = [self frameForSegmentAtIndex:i];
        
        // Work out the appropriate size of the item
        CGRect itemFrame = itemView.frame;
        
        // Cap its size to be within the segmented frame
        itemFrame.size.height = MIN(thumbFrame.size.height, itemFrame.size.height);
        itemFrame.size.width = MIN(thumbFrame.size.width, itemFrame.size.width);
        
        // If the item is reversible, make sure there is also room to show the arrow
        CGFloat arrowSpacing = (self.arrowImage.size.width + kTOSegmentedControlDirectionArrowMargin) * 2.0f;
        if (item.isReversible && (itemFrame.size.width + arrowSpacing) > thumbFrame.size.width) {
            itemFrame.size.width -= arrowSpacing;
        }
        
        // Center the item in the container
        itemFrame.origin.x = CGRectGetMidX(thumbFrame) - (itemFrame.size.width * 0.5f);
        itemFrame.origin.y = CGRectGetMidY(thumbFrame) - (itemFrame.size.height * 0.5f);
        
        // Set the item frame
        itemView.frame = CGRectIntegral(itemFrame);

        // Make sure they are all unselected
        [self setItemAtIndex:i selected:NO];

        // If the item is disabled, make it faded
        if (!self.enabled || item.isDisabled) {
            itemView.alpha = kTOSegmentedControlDisabledAlpha;
        }

        i++;
    }

    // Exit out if there is nothing selected
    if (self.selectedSegmentIndex < 0) { return; }

    // Set the selected state for the current selected index
    [self setItemAtIndex:self.selectedSegmentIndex selected:YES];
}

- (void)layoutSeparatorViews
{
    CGSize size = self.trackView.frame.size;
    CGFloat segmentWidth = self.segmentWidth;
    CGFloat xOffset = (_thumbInset + segmentWidth) - 1.0f;
    NSInteger i = 0;
    for (UIView *separatorView in self.separatorViews) {
       CGRect frame = separatorView.frame;
       frame.size.width = 1.0f;
       frame.size.height = (size.height - (self.cornerRadius) * 2.0f) + 2.0f;
       frame.origin.x = xOffset + (segmentWidth * i);
       frame.origin.y = (size.height - frame.size.height) * 0.5f;
       separatorView.frame = CGRectIntegral(frame);
       i++;
    }

   // Update the alpha of the separator views
   [self refreshSeparatorViewsForSelectedIndex:self.selectedSegmentIndex];
}

- (void)layoutSubviews
{
    [super layoutSubviews];

    // Lay-out the thumb view
    [self layoutThumbView];

    // Lay-out the item views
    [self layoutItemViews];

    // Lay-out the separator views
    [self layoutSeparatorViews];
}

- (CGFloat)segmentWidth
{
    return floorf((self.bounds.size.width - (_thumbInset * 2.0f)) / self.numberOfSegments);
}

- (CGRect)frameForSegmentAtIndex:(NSInteger)index
{
    CGSize size = self.trackView.frame.size;
    
    CGRect frame = CGRectZero;
    frame.origin.x = _thumbInset + (self.segmentWidth * index) + ((_thumbInset * 2.0f) * index);
    frame.origin.y = _thumbInset;
    frame.size.width = self.segmentWidth;
    frame.size.height = size.height - (_thumbInset * 2.0f);
    
    // Cap the position of the frame so it won't overshoot
    frame.origin.x = MAX(_thumbInset, frame.origin.x);
    frame.origin.x = MIN(size.width - (self.segmentWidth + _thumbInset), frame.origin.x);
    
    return CGRectIntegral(frame);
}

- (CGRect)frameForImageArrowViewWithItemFrame:(CGRect)itemFrame
{
    CGRect frame = CGRectZero;
    frame.size = self.arrowImage.size;
    frame.origin.x = CGRectGetMaxX(itemFrame) + kTOSegmentedControlDirectionArrowMargin;
    frame.origin.y = ceilf(CGRectGetMidY(itemFrame) - (frame.size.height * 0.5f));
    return frame;
}

- (NSInteger)segmentIndexForPoint:(CGPoint)point
{
    CGFloat segmentWidth = floorf(self.frame.size.width / self.numberOfSegments);
    NSInteger segment = floorf(point.x / segmentWidth);
    segment = MAX(segment, 0);
    segment = MIN(segment, self.numberOfSegments-1);
    return segment;
}

- (void)setThumbViewShrunken:(BOOL)shrunken
{
    CGFloat scale = shrunken ? kTOSegmentedControlSelectedScale : 1.0f;
    self.thumbView.transform = CGAffineTransformScale(CGAffineTransformIdentity,
                                                      scale, scale);
}

- (void)setItemViewAtIndex:(NSInteger)segmentIndex shrunken:(BOOL)shrunken
{
    NSAssert(segmentIndex >= 0 && segmentIndex < self.items.count,
             @"TOSegmentedControl: Array should not be out of bounds");

    TOSegmentedControlSegment *segment = self.segments[segmentIndex];
    UIView *itemView = segment.itemView;
    CGRect itemFrame = itemView.frame;
    CGPoint itemViewCenter = itemView.center;

    if (shrunken == NO) {
        itemView.transform = CGAffineTransformIdentity;
    }
    else {
        CGFloat scale = kTOSegmentedControlSelectedScale;
        itemView.transform = CGAffineTransformScale(CGAffineTransformIdentity,
                                                          scale, scale);
    }

    // If we have a reversible image view, manipulate its transformation
    // to match the position and scale of the item view
    UIView *arrowView = segment.arrowView;
    if (arrowView == nil) { return; }

    if (!shrunken) {
        arrowView.transform = CGAffineTransformIdentity;
        return;
    }

    CGFloat scale = kTOSegmentedControlSelectedScale;
    CGRect arrowFrame = [self frameForImageArrowViewWithItemFrame:itemFrame];

    // Work out the delta between the middle of the item view,
    // and the middle of the image view
    CGPoint offset = CGPointZero;
    offset.x = (CGRectGetMidX(arrowFrame) - itemViewCenter.x);

    // Create a transformation matrix that applies the scale to the arrow,
    // with the transformation origin being the middle of the item view
    CGAffineTransform transform = arrowView.transform;
    transform = CGAffineTransformTranslate(transform, -offset.x, -offset.y);
    transform = CGAffineTransformScale(transform, scale, scale);
    transform = CGAffineTransformTranslate(transform, offset.x, offset.y);
    arrowView.transform = transform;
}

- (void)setItemViewAtIndex:(NSInteger)segmentIndex reversed:(BOOL)reversed
{
    NSAssert(segmentIndex >= 0 && segmentIndex < self.items.count,
             @"TOSegmentedControl: Array should not be out of bounds");

    TOSegmentedControlSegment *segment = self.segments[segmentIndex];
    [segment setArrowImageReversed:reversed];
}

- (void)setItemAtIndex:(NSInteger)index selected:(BOOL)selected
{
    NSAssert(index >= 0 && index < self.segments.count,
             @"TOSegmentedControl: Array should not be out of bounds");

    // Tell the segment to select itself in order to show the reversible arrow
    TOSegmentedControlSegment *segment = self.segments[index];

    // Update the alpha of the reversible arrow
    segment.arrowView.alpha = selected ? kTOSegmentedControlDirectionArrowAlpha : 0.0f;

    // The rest of this code deals with swapping the font
    // of the label. Cancel out if we're an image.
    UILabel *label = segment.label;
    if (label == nil) { return; }

    // Set the font
    UIFont *font = selected ? self.selectedTextFont : self.textFont;
    label.font = font;

    // Set the text color
    label.textColor = selected ? self.selectedItemColor : self.itemColor;
    
    // Set the arrow tint color
    segment.arrowView.tintColor = label.textColor;
    
    // Re-apply the arrow image view to the translated frame
    segment.arrowView.frame = [self frameForImageArrowViewWithItemFrame:label.frame];

    // Ensure the arrow view is set to the right orientation
    [segment setArrowImageReversed:segment.isReversed];
}

- (void)setItemAtIndex:(NSInteger)index faded:(BOOL)faded
{
    NSAssert(index >= 0 && index < self.segments.count,
             @"Array should not be out of bounds");
    UIView *itemView = self.segments[index].itemView;
    itemView.alpha = faded ? kTOSegmentedControlSelectedTextAlpha : 1.0f;
}

- (void)refreshSeparatorViewsForSelectedIndex:(NSInteger)index
{
    // Hide the separators on either side of the selected segment
    NSInteger i = 0;
    for (UIView *separatorView in self.separatorViews) {
        // if the view is disabled, the thumb view will be hidden
        if (!self.enabled) {
            separatorView.alpha = 1.0f;
            continue;
        }

        separatorView.alpha = (i == index || i == (index - 1)) ? 0.0f : 1.0f;
        i++;
    }
}

#pragma mark - Touch Interaction -

- (void)didTapDown:(UIControl *)control withEvent:(UIEvent *)event
{
    // Exit out if the control is disabled
    if (!self.enabled || self.hasNoSegments) { return; }

    // Determine which segment the user tapped
    CGPoint tapPoint = [event.allTouches.anyObject locationInView:self];
    NSInteger tappedIndex = [self segmentIndexForPoint:tapPoint];

    // If the control or item is disabled, pass
    if (self.segments[tappedIndex].isDisabled) {
        return;
    }
    
    // Work out if we tapped on the thumb view, or on an un-selected segment
    self.isDraggingThumbView = (tappedIndex == self.selectedSegmentIndex);

    // Track if we drag off this segment
    self.didDragOffOriginalSegment = NO;

    // Track the currently selected item as the focused one
    self.focusedIndex = tappedIndex;

    // Work out which animation effects to apply
    if (!self.isDraggingThumbView) {
        [UIView animateWithDuration:0.35f animations:^{
            [self setItemAtIndex:tappedIndex faded:YES];
        }];
        
        [self setSelectedSegmentIndex:tappedIndex animated:YES];
        return;
    }
    
    id animationBlock = ^{
        [self setThumbViewShrunken:YES];
        [self setItemViewAtIndex:self.selectedSegmentIndex shrunken:YES];
    };
    
    // Animate the transition
    [UIView animateWithDuration:0.3f
                          delay:0.0f
         usingSpringWithDamping:1.0f
          initialSpringVelocity:0.1f
                        options:UIViewAnimationOptionBeginFromCurrentState
                     animations:animationBlock
                     completion:nil];
}

- (void)didDragTap:(UIControl *)control withEvent:(UIEvent *)event
{
    // Exit out if the control is disabled
    if (!self.enabled || self.hasNoSegments) { return; }

    CGPoint tapPoint = [event.allTouches.anyObject locationInView:self];
    NSInteger tappedIndex = [self segmentIndexForPoint:tapPoint];
    
    if (tappedIndex == self.focusedIndex) {
      return;
    }

    // If the control or item is disabled, pass
    if (self.segments[tappedIndex].isDisabled) {
        return;
    }

    // Track that we dragged off the first segments
    self.didDragOffOriginalSegment = YES;

    // Handle transitioning when not dragging the thumb view
    if (!self.isDraggingThumbView) {
        // If we dragged out of the bounds, disregard
        if (self.focusedIndex < 0) { return; }
        
        id animationBlock = ^{
            // Deselect the current item
            [self setItemAtIndex:self.focusedIndex faded:NO];
            
            // Fade the text if it is NOT the thumb track one
            if (tappedIndex != self.selectedSegmentIndex) {
                [self setItemAtIndex:tappedIndex faded:YES];
            }
        };
        
        // Perform a faster change over animation
        [UIView animateWithDuration:0.3f
                              delay:0.0f
                            options:UIViewAnimationOptionBeginFromCurrentState
                         animations:animationBlock
                         completion:nil];
        
        // Update the focused item
        self.focusedIndex = tappedIndex;
        return;
    }
    
    // Get the new frame of the segment
    CGRect frame = [self frameForSegmentAtIndex:tappedIndex];
    
    // Work out the center point from the frame
    CGPoint center = (CGPoint){CGRectGetMidX(frame), CGRectGetMidY(frame)};

    // Create the animation block
    id animationBlock = ^{
        self.thumbView.center = center;
        
        // Deselect the focused item
        [self setItemAtIndex:self.focusedIndex selected:NO];
        [self setItemViewAtIndex:self.focusedIndex shrunken:NO];
        
        // Select the new one
        [self setItemAtIndex:tappedIndex selected:YES];
        [self setItemViewAtIndex:tappedIndex shrunken:YES];
        
        // Update the separators
        [self refreshSeparatorViewsForSelectedIndex:tappedIndex];
    };
    
    // Perform the animation
    [UIView animateWithDuration:0.45
                          delay:0.0f
         usingSpringWithDamping:1.0f
          initialSpringVelocity:1.0f
                        options:UIViewAnimationOptionBeginFromCurrentState
                     animations:animationBlock
                     completion:nil];
    
    // Update the focused item
    self.focusedIndex = tappedIndex;
}

- (void)didExitTapBounds:(UIControl *)control withEvent:(UIEvent *)event
{
    // Exit out if the control is disabled
    if (!self.enabled || self.hasNoSegments) { return; }

    // No effects needed when tracking the thumb view
    if (self.isDraggingThumbView) { return; }
    
    // Un-fade the focused item
    [UIView animateWithDuration:0.45f
                          delay:0.0f
                        options:UIViewAnimationOptionBeginFromCurrentState
                     animations:^{ [self setItemAtIndex:self.focusedIndex faded:NO]; }
                     completion:nil];
    
    // Disable the focused index
    self.focusedIndex = -1;
}

- (void)didEnterTapBounds:(UIControl *)control withEvent:(UIEvent *)event
{
    // Exit out if the control is disabled
    if (!self.enabled || self.hasNoSegments) { return; }

    // No effects needed when tracking the thumb view
    if (self.isDraggingThumbView) { return; }
    
    CGPoint tapPoint = [event.allTouches.anyObject locationInView:self];
    self.focusedIndex = [self segmentIndexForPoint:tapPoint];
    
    // Un-fade the focused item
    [UIView animateWithDuration:0.45f
                          delay:0.0f
                        options:UIViewAnimationOptionBeginFromCurrentState
                     animations:^{ [self setItemAtIndex:self.focusedIndex faded:YES]; }
                     completion:nil];
}

- (void)didEndTap:(UIControl *)control withEvent:(UIEvent *)event
{
    // Exit out if the control is disabled
    if (!self.enabled || self.hasNoSegments) { return; }

    // Capture the touch object in order to track its state
    UITouch *touch = event.allTouches.anyObject;

    // Check if the tap was cancelled (In which case we shouldn't commit non-drag events)
    BOOL isCancelled = (touch.phase == UITouchPhaseCancelled);

    // Work out the final place where we released
    CGPoint tapPoint = [touch locationInView:self];
    NSInteger tappedIndex = [self segmentIndexForPoint:tapPoint];

    TOSegmentedControlSegment *segment = self.segments[tappedIndex];

    // If we WEREN'T dragging the thumb view, work out where we need to move to
    if (!self.isDraggingThumbView) {
        if (segment.isDisabled) { return; }

        // If we weren't cancelled, animate to the new index
        if (!isCancelled) {
            [self setSelectedSegmentIndex:tappedIndex animated:YES];
        }
        else {
            // Else, reset the currently highlighted item
            [self didExitTapBounds:self withEvent:event];
        }

        // Reset the focused index flag
        self.focusedIndex = -1;

        return;
    }

    // Update the state and alert the delegate
    if (self.selectedSegmentIndex != tappedIndex) {
        _selectedSegmentIndex = tappedIndex;
        [self sendIndexChangedEventActions];
    }
    else if (segment.isReversible && !self.didDragOffOriginalSegment) {
        // If the item was reversible, and we never changed segments,
        // trigger the reverse alert delegate
        [segment toggleDirection];
        [self sendIndexChangedEventActions];
    }

    // Work out which animation effects to apply
    id animationBlock = ^{
        [self setThumbViewShrunken:NO];
        [self setItemViewAtIndex:self.selectedSegmentIndex shrunken:NO];
        [self setItemViewAtIndex:self.selectedSegmentIndex
                        reversed:self.selectedSegmentReversed];
    };

    // Animate the transition
    [UIView animateWithDuration:0.3f
                         delay:0.0f
        usingSpringWithDamping:1.0f
         initialSpringVelocity:0.1f
                       options:UIViewAnimationOptionBeginFromCurrentState
                    animations:animationBlock
                    completion:nil];

    // Reset the focused index flag
    self.focusedIndex = -1;
}

- (void)sendIndexChangedEventActions
{
    // Trigger the action event for any targets that were
    [self sendActionsForControlEvents:UIControlEventValueChanged];

    // Trigger the block if it is set
    if (self.segmentTappedHandler) {
        self.segmentTappedHandler(self.selectedSegmentIndex,
                                  self.selectedSegmentReversed);
    }
}

#pragma mark - Accessors -

// -----------------------------------------------
// Selected Item Index

- (void)setSelectedSegmentIndex:(NSInteger)selectedSegmentIndex animated:(BOOL)animated
{
    if (self.selectedSegmentIndex == selectedSegmentIndex) { return; }

    // Set the new value
    _selectedSegmentIndex = selectedSegmentIndex;

    // Cap the value
    _selectedSegmentIndex = MAX(selectedSegmentIndex, -1);
    _selectedSegmentIndex = MIN(selectedSegmentIndex, self.numberOfSegments - 1);

    // Send the update alert
    if (_selectedSegmentIndex >= 0) {
        [self sendIndexChangedEventActions];
    }

    if (!animated) {
        // Trigger a view layout
        [self setNeedsLayout];
        return;
    }

    // Create an animation block that will update the position of the
    // thumb view and restore all of the item views
    id animationBlock = ^{
        // Un-fade all of the item views
        for (NSInteger i = 0; i < self.segments.count; i++) {
            // De-select everything
            [self setItemAtIndex:i faded:NO];
            [self setItemAtIndex:i selected:NO];

            // Select the currently selected index
            [self setItemAtIndex:self.selectedSegmentIndex selected:YES];

            // Move the thumb view
            self.thumbView.frame = [self frameForSegmentAtIndex:self.selectedSegmentIndex];

            // Update the separators
            [self refreshSeparatorViewsForSelectedIndex:self.selectedSegmentIndex];
        }
    };

    // Commit the animation
    [UIView animateWithDuration:0.45
                          delay:0.0f
         usingSpringWithDamping:1.0f
          initialSpringVelocity:2.0f
                        options:UIViewAnimationOptionBeginFromCurrentState
                     animations:animationBlock
                     completion:nil];


}

// -----------------------------------------------
// Selected Item Reversed

- (void)setSelectedSegmentReversed:(BOOL)selectedSegmentReversed
{
    if (self.selectedSegmentIndex < 0) { return; }
    TOSegmentedControlSegment *segment = self.segments[self.selectedSegmentIndex];
    if (segment.isReversible == NO) { return; }
    segment.isReversed = selectedSegmentReversed;
}

- (BOOL)selectedSegmentReversed
{
    if (self.selectedSegmentIndex < 0) { return NO; }
    TOSegmentedControlSegment *segment = self.segments[self.selectedSegmentIndex];
    if (segment.isReversible == NO) { return NO; }
    return segment.isReversed;
}

// -----------------------------------------------
// Items

- (void)setItems:(NSArray *)items
{
    if (items == _items) { return; }

    // Remove all current items
    [self removeAllSegments];

    // Set the new array
    _items = [self sanitizedItemArrayWithItems:items];

    // Create the list of item objects  to track their state
    _segments = [TOSegmentedControlSegment segmentsWithObjects:_items
                                           forSegmentedControl:self].mutableCopy;
    
    // Update the number of separators
    [self updateSeparatorViewCount];

    // Trigger a layout update
    [self setNeedsLayout];

    // Set the initial selected index
    self.selectedSegmentIndex = (_items.count > 0) ? 0 : -1;
}

// -----------------------------------------------
// Corner Radius

- (void)setCornerRadius:(CGFloat)cornerRadius
{
    self.trackView.layer.cornerRadius = cornerRadius;
    self.thumbView.layer.cornerRadius = (self.cornerRadius - _thumbInset) + 1.0f;
}

- (CGFloat)cornerRadius { return self.trackView.layer.cornerRadius; }

// -----------------------------------------------
// Thumb Color

- (void)setThumbColor:(UIColor *)thumbColor
{
    self.thumbView.backgroundColor = thumbColor;
    if (self.thumbView.backgroundColor != nil) { return; }

    // On iOS 12 and below, simply set the thumb view to be white
    self.thumbView.backgroundColor = [UIColor whiteColor];

    // For iOS 13 and up, create a dynamic provider that will trigger a color change
    #ifdef __IPHONE_13_0
    if (@available(iOS 13.0, *)) {
        // Create the provider block that will trigger each time the trait collection changes
        id dynamicColorProvider = ^UIColor *(UITraitCollection *traitCollection) {
            // Dark color
            if (traitCollection.userInterfaceStyle == UIUserInterfaceStyleDark) {
                return [UIColor colorWithRed:0.357f green:0.357f blue:0.376f alpha:1.0f];
            }

            // Default light color
            return [UIColor whiteColor];
        };

        // Assign the dynamic color to the view
        self.thumbView.backgroundColor = [UIColor colorWithDynamicProvider:dynamicColorProvider];
    }
    #endif
}
- (UIColor *)thumbColor { return self.thumbView.backgroundColor; }

// -----------------------------------------------
// Background Color

- (void)setBackgroundColor:(UIColor *)backgroundColor
{
    [super setBackgroundColor:[UIColor clearColor]];
    _trackView.backgroundColor = backgroundColor;

    // Exit out if we don't need to reset to defaults
    if (_trackView.backgroundColor != nil) { return; }

    // Set the default color for iOS 12 and below
    backgroundColor = [UIColor colorWithRed:0.0f green:0.0f blue:0.08f alpha:0.06666f];
    _trackView.backgroundColor = backgroundColor;

    // For iOS 13 and up, create a dynamic provider that will trigger on trait changes
    #ifdef __IPHONE_13_0
    if (@available(iOS 13.0, *)) {

        // Create the provider block that will trigger each time the trait collection changes
        id dynamicColorProvider = ^UIColor *(UITraitCollection *traitCollection) {
            // Dark color
            if (traitCollection.userInterfaceStyle == UIUserInterfaceStyleDark) {
                return [UIColor colorWithRed:0.898f green:0.898f blue:1.0f alpha:0.12f];
            }

            // Default light color
            return backgroundColor;
        };

        // Assign the dynamic color to the view
        _trackView.backgroundColor = [UIColor colorWithDynamicProvider:dynamicColorProvider];
    }
    #endif
}
- (UIColor *)backgroundColor { return self.trackView.backgroundColor; }

// -----------------------------------------------
// Separator Color

- (void)setSeparatorColor:(UIColor *)separatorColor
{
    _separatorColor = separatorColor;
    if (_separatorColor == nil) {
        // Set the default color for iOS 12 and below
        separatorColor = [UIColor colorWithRed:0.0f green:0.0f blue:0.08f alpha:0.1f];

        // On iOS 13 and up, set up a dynamic provider for dynamic light and dark colors
        #ifdef __IPHONE_13_0
        if (@available(iOS 13.0, *)) {
            // Create the provider block that will trigger each time the trait collection changes
            id dynamicColorProvider = ^UIColor *(UITraitCollection *traitCollection) {
                // Dark color
                if (traitCollection.userInterfaceStyle == UIUserInterfaceStyleDark) {
                    return [UIColor colorWithRed:0.918f green:0.918f blue:1.0f alpha:0.16f];
                }

                // Default light color
                return separatorColor;
            };

            // Assign the dynamic color to the view
            separatorColor = [UIColor colorWithDynamicProvider:dynamicColorProvider];
        }
        #endif

        _separatorColor = separatorColor;
    }

    for (UIView *separatorView in self.separatorViews) {
        separatorView.tintColor = _separatorColor;
    }
}

// -----------------------------------------------
// Item Color

- (void)setItemColor:(UIColor *)itemColor
{
    _itemColor = itemColor;
    if (_itemColor == nil) {
        _itemColor = [UIColor blackColor];

        // Assign the dynamic label color on iOS 13 and up
        #ifdef __IPHONE_13_0
        if (@available(iOS 13.0, *)) {
            _itemColor = [UIColor labelColor];
        }
        #endif
    }

    // Set each item to the color
    for (TOSegmentedControlSegment *item in self.segments) {
        [item refreshItemView];
    }
}

//-------------------------------------------------
// Selected Item Color

- (void)setSelectedItemColor:(UIColor *)selectedItemColor
{
    _selectedItemColor = selectedItemColor;
    if (_selectedItemColor == nil) {
        _selectedItemColor = [UIColor blackColor];

        // Assign the dynamic label color on iOS 13 and up
        #ifdef __IPHONE_13_0
        if (@available(iOS 13.0, *)) {
            _selectedItemColor = [UIColor labelColor];
        }
        #endif
    }

    // Set each item to the color
    for (TOSegmentedControlSegment *item in self.segments) {
        [item refreshItemView];
    }
}

// -----------------------------------------------
// Text Font

- (void)setTextFont:(UIFont *)textFont
{
    _textFont = textFont;
    if (_textFont == nil) {
        _textFont = [UIFont systemFontOfSize:13.0f weight:UIFontWeightMedium];
    }

    // Set each item to adopt the new font
    for (TOSegmentedControlSegment *item in self.segments) {
        [item refreshItemView];
    }
}

// -----------------------------------------------
// Selected Text Font

- (void)setSelectedTextFont:(UIFont *)selectedTextFont
{
    _selectedTextFont = selectedTextFont;
    if (_selectedTextFont == nil) {
        _selectedTextFont = [UIFont systemFontOfSize:13.0f weight:UIFontWeightSemibold];
    }

    // Set each item to adopt the new font
    for (TOSegmentedControlSegment *item in self.segments) {
        [item refreshItemView];
    }
}

// -----------------------------------------------
// Thumb Inset

- (void)setThumbInset:(CGFloat)thumbInset
{
    _thumbInset = thumbInset;
    self.thumbView.layer.cornerRadius = (self.cornerRadius - _thumbInset) + 1.0f;
}

// -----------------------------------------------
// Shadow Properties

- (void)setThumbShadowOffset:(CGFloat)thumbShadowOffset {self.thumbView.layer.shadowOffset = (CGSize){0.0f, thumbShadowOffset}; }
- (CGFloat)thumbShadowOffset { return self.thumbView.layer.shadowOffset.height; }

- (void)setThumbShadowOpacity:(CGFloat)thumbShadowOpacity { self.thumbView.layer.shadowOpacity = thumbShadowOpacity; }
- (CGFloat)thumbShadowOpacity { return self.thumbView.layer.shadowOpacity; }

- (void)setThumbShadowRadius:(CGFloat)thumbShadowRadius { self.thumbView.layer.shadowRadius = thumbShadowRadius; }
- (CGFloat)thumbShadowRadius { return self.thumbView.layer.shadowRadius; }

// -----------------------------------------------
// Number of segments

- (NSInteger)numberOfSegments { return self.segments.count; }

// -----------------------------------------------
// Setting all reversible indexes
- (void)setReversibleSegmentIndexes:(NSArray<NSNumber *> *)reversibleSegmentIndexes
{
    for (NSInteger i = 0; i < self.numberOfSegments; i++) {
        BOOL reversible = [reversibleSegmentIndexes indexOfObject:@(i)] != NSNotFound;
        [self setReversible:reversible forSegmentAtIndex:i];
    }
}

- (NSArray<NSNumber *> *)reversibleSegmentIndexes
{
    NSMutableArray *array = [NSMutableArray array];
    for (NSInteger i = 0; i < self.numberOfSegments; i++) {
        if ([self isReversibleForSegmentAtIndex:i]) {
            [array addObject:@(i)];
        }
    }
    
    return [NSArray arrayWithArray:array];
}

- (BOOL)hasNoSegments { return self.segments.count <= 0; }

#pragma mark - Image Creation and Management -

- (UIImage *)arrowImage
{
    // Retrieve from the image table
    UIImage *arrowImage = [self.imageTable objectForKey:kTOSegmentedControlArrowImage];
    if (arrowImage != nil) { return arrowImage; }

    // Generate for the first time
    UIGraphicsBeginImageContextWithOptions((CGSize){8.0f, 4.0f}, NO, 0.0f);
    {
        UIBezierPath* bezierPath = [UIBezierPath bezierPath];
        [bezierPath moveToPoint: CGPointMake(7.25, 0.75)];
        [bezierPath addLineToPoint: CGPointMake(4, 3.25)];
        [bezierPath addLineToPoint: CGPointMake(0.75, 0.75)];
        [UIColor.blackColor setStroke];
        bezierPath.lineWidth = 1.5;
        bezierPath.lineCapStyle = kCGLineCapRound;
        bezierPath.lineJoinStyle = kCGLineJoinRound;
        [bezierPath stroke];
        arrowImage = UIGraphicsGetImageFromCurrentImageContext();
    }
    UIGraphicsEndImageContext();

    // Force to always be template
    arrowImage = [arrowImage imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];

    // Save to the map table for next time
    [self.imageTable setObject:arrowImage forKey:kTOSegmentedControlArrowImage];

    return arrowImage;
}

- (UIImage *)separatorImage
{
    UIImage *separatorImage = [self.imageTable objectForKey:kTOSegmentedControlSeparatorImage];
    if (separatorImage != nil) { return separatorImage; }

    UIGraphicsBeginImageContextWithOptions((CGSize){1.0f, 3.0f}, NO, 0.0f);
    {
        UIBezierPath* separatorPath = [UIBezierPath bezierPathWithRoundedRect:CGRectMake(0, 0, 1, 3) cornerRadius:0.5];
        [UIColor.blackColor setFill];
        [separatorPath fill];
        separatorImage = UIGraphicsGetImageFromCurrentImageContext();
    }
    UIGraphicsEndImageContext();

    // Format image to be resizable and tint-able.
    separatorImage = [separatorImage resizableImageWithCapInsets:(UIEdgeInsets){1.0f, 0.0f, 1.0f, 0.0f}
                                                    resizingMode:UIImageResizingModeTile];
    separatorImage = [separatorImage imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];

    return separatorImage;
}

- (NSMapTable *)imageTable
{
    // The map table is a global instance that allows all instances of
    // segmented controls to efficiently share the same images.

    // The images themselves are weakly referenced, so they will be cleaned
    // up from memory when all segmented controls using them are deallocated.

    if (_imageTable) { return _imageTable; }
    _imageTable = [NSMapTable mapTableWithKeyOptions:NSPointerFunctionsStrongMemory
                                        valueOptions:NSPointerFunctionsWeakMemory];
    return _imageTable;
}

@end
