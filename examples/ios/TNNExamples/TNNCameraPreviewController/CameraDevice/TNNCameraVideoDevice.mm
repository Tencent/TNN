//  Copyright © 2020 tencent. All rights reserved.

#import "TNNCameraVideoDevice.h"
#import <AVFoundation/AVFoundation.h>
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>

API_AVAILABLE(ios(10.0))
@interface TNNCameraVideoDevice ()<AVCaptureVideoDataOutputSampleBufferDelegate,
AVCapturePhotoCaptureDelegate> {
}
@property (nonatomic, strong) AVCaptureSession *captureSession;
@property (nonatomic, strong) AVCaptureDevice *captureDevice;
@property (nonatomic, strong) AVCaptureDeviceInput *captureDeviceInput;
@property (nonatomic, strong) AVCaptureVideoDataOutput *videoOutput;
@property (nonatomic, strong) AVCapturePhotoOutput *photoOutput;
@property (nonatomic, strong) AVCaptureVideoPreviewLayer *videoPreviewLayer;

@property (nonatomic, strong) dispatch_queue_t queue;
@property (nonatomic, strong) id <MTLDevice> device;

@property (nonatomic, assign) CVMetalTextureCacheRef textureCache;
@end

@implementation TNNCameraVideoDevice

- (instancetype)init
{
    self = [super init];
    if (self) {
        _queue = dispatch_queue_create("camera.queue", NULL);
        _captureSession = [[AVCaptureSession alloc] init];
        _device = MTLCreateSystemDefaultDevice();
        
        _videoPreviewLayer = [AVCaptureVideoPreviewLayer layerWithSession:_captureSession];
        _videoPreviewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
        _videoPreviewLayer.connection.videoOrientation = AVCaptureVideoOrientationPortrait;
        
        _textureCache = nil;
    }
    return self;
}


- (void)dealloc {
    if (_textureCache) {
        CFRelease(_textureCache);
        _textureCache = nil;
    }
}

- (void)switchCamera:(AVCaptureDevicePosition)position
 withPreset:(AVCaptureSessionPreset)sessionPreset
completion:(CameraSetupCallback)completion {
    [self stopSession];
    dispatch_async(_queue, ^{
        auto success = [self switchCamera:position withPreset:sessionPreset];
        dispatch_async(dispatch_get_main_queue(), ^{
            if (success) {
                [self startSession];
            }
            completion(success);
        });
    });
}

- (BOOL)switchCamera:(AVCaptureDevicePosition)position
          withPreset:(AVCaptureSessionPreset)sessionPreset {
    CVMetalTextureCacheFlush(_textureCache, 0);
    if (CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, _device, nil, &_textureCache) != kCVReturnSuccess) {
        NSLog(@"Error: setupCaptureSession could not create a texture cache");
        return NO;
    }
    
    // 1.创建会话
    [_captureSession beginConfiguration];
    _captureSession.sessionPreset = sessionPreset;
    
    // 2.创建输入设备
    _captureDevice = [self cameraWithPosition:position];
    if (!_captureDevice) {
        NSLog(@"Error: no video AVCaptureDevice availablee");
        return NO;
    }
    
    // 3.创建输入，并连接会话
    if (_captureDeviceInput) {
        [_captureSession removeInput:_captureDeviceInput];
        _captureDeviceInput = nil;
    }
    NSError *error = nil;
    _captureDeviceInput = [AVCaptureDeviceInput deviceInputWithDevice:_captureDevice error:&error];
    if (error || !_captureDeviceInput) {
        NSLog(@"Error: no video AVCaptureDeviceInput availablee");
        return NO;
    }
    if ([_captureSession canAddInput:_captureDeviceInput]) {
        [_captureSession addInput:_captureDeviceInput];
    }
    
    // 4.创建输出，并连接会话
    if (_photoOutput) {
        [_captureSession removeOutput:_photoOutput];
        _photoOutput = nil;
    }
    if (@available(iOS 10.0, *)) {
        _photoOutput = [[AVCapturePhotoOutput alloc] init];
    }
    if (_photoOutput && [_captureSession canAddOutput:_photoOutput]) {
        [_captureSession addOutput:_photoOutput];
    }
    [self addVideoOutput];
    
    [_captureSession commitConfiguration];
    return YES;
}

- (void)addVideoOutput
{
    if (_videoOutput) {
        [_captureSession removeOutput:_videoOutput];
        _videoOutput = nil;
    }
    _videoOutput = [[AVCaptureVideoDataOutput alloc] init];
    _videoOutput.alwaysDiscardsLateVideoFrames = YES;
    _videoOutput.videoSettings = @{ (NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA) };
    [_videoOutput setSampleBufferDelegate:self queue:_queue];
    if ([_captureSession canAddOutput:_videoOutput]) {
        [_captureSession addOutput:_videoOutput];
    } else {
        NSLog(@"couldn't add video output");
    }
    auto connection = [_videoOutput connectionWithMediaType:AVMediaTypeVideo];
    connection.videoOrientation = AVCaptureVideoOrientationPortrait;
}

- (AVCaptureDevice *) cameraWithPosition:(AVCaptureDevicePosition) position
{
    NSArray *devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    for (AVCaptureDevice *device in devices)
    {
        if ([device position] == position) return device;
    }
    return nil;
}

- (AVCaptureDevicePosition)cameraPosition {
    auto pos = AVCaptureDevicePositionUnspecified;
    if (_captureDevice) {
        pos = _captureDevice.position;
        return pos;
    }
    if (_captureSession.inputs.count > 0) {
        auto currentCameraInput = [_captureSession.inputs objectAtIndex:0];
        pos = ((AVCaptureDeviceInput*)currentCameraInput).device.position;
    }
    return pos;
}

- (AVCaptureDevicePosition)rotateCamera
{
    AVCaptureDevicePosition pos = AVCaptureDevicePositionUnspecified;
    //Change camera source
    if(_captureSession)
    {
        //Indicate that some changes will be made to the session
        [_captureSession beginConfiguration];

        //Remove existing input
        AVCaptureInput* currentCameraInput = [_captureSession.inputs objectAtIndex:0];
        [_captureSession removeInput:currentCameraInput];

        //Get new input
        AVCaptureDevice *newCamera = nil;
        if(((AVCaptureDeviceInput*)currentCameraInput).device.position == AVCaptureDevicePositionBack)
        {
          newCamera = [self cameraWithPosition:AVCaptureDevicePositionFront];
          pos = AVCaptureDevicePositionFront;
        }
        else
        {
          newCamera = [self cameraWithPosition:AVCaptureDevicePositionBack];
          pos = AVCaptureDevicePositionBack;
        }
        _captureDevice = newCamera;
        //Add input to session
        NSError *err = nil;
        _captureDeviceInput = [[AVCaptureDeviceInput alloc] initWithDevice:_captureDevice error:&err];
        if(!_captureDeviceInput || err)
        {
          NSLog(@"Error creating capture device input: %@", err.localizedDescription);
        }
        else
        {
          [_captureSession addInput:_captureDeviceInput];
        }
        [_captureSession removeOutput:_videoOutput];
        [self addVideoOutput];
        //Commit all the configuration changes at once
        [_captureSession commitConfiguration];
    }
    return pos;
}


- (void)startSession {
    if(![_captureSession isRunning]) {
        [_captureSession startRunning];
    }
}

- (void)stopSession {
    if([_captureSession isRunning]) {
        [_captureSession stopRunning];
    }
}

//Capture a single frame of the camera input
-(void)capturePhoto:(AVCapturePhotoSettings *)settings  API_AVAILABLE(ios(10.0)){
    if (settings == nil) {
        settings = [AVCapturePhotoSettings photoSettingsWithFormat:
                    @{(NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA) }];
        
        settings.previewPhotoFormat = @{
            (NSString *)kCVPixelBufferPixelFormatTypeKey : settings.availablePreviewPhotoPixelFormatTypes[0],
            (NSString *)kCVPixelBufferHeightKey : @(480),
            (NSString *)kCVPixelBufferWidthKey : @(360)
        };
    }
    
    [_photoOutput capturePhotoWithSettings:settings
                                      delegate:self];
}

#pragma mark - AVCaptureVideoDataOutputSampleBufferDelegate
- (void)captureOutput:(AVCaptureOutput *)captureOutput
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection
{
    auto inputPort = [connection.inputPorts objectAtIndex:0];
    auto input = (AVCaptureDeviceInput *)inputPort.input;
    auto inputPos = input.device.position;

    if (inputPos == AVCaptureDevicePositionUnspecified) {
        NSLog(@"captureOutput, camera position unspecified, drop it!!!!");
        return;
    }
    auto timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer);

    if (_delegate && [_delegate respondsToSelector:@selector(cameraDevice:didCaptureVideo:withPosition:atTimestamp:)]) {
        [_delegate cameraDevice:self
                    didCaptureVideo:sampleBuffer
                       withPosition:inputPos
                        atTimestamp:timestamp];
    }
    
}

-(void)captureOutput:(AVCapturePhotoOutput *)output
 didDropSampleBuffer:(nonnull CMSampleBufferRef)sampleBuffer
      fromConnection:(nonnull AVCaptureConnection *)connection  API_AVAILABLE(ios(10.0)){
    NSLog(@"dropped frame");
}

#pragma mark - AVCapturePhotoCaptureDelegate
- (void)captureOutput:(AVCaptureOutput *)captureOutput
didFinishProcessingPhotoSampleBuffer:(nullable CMSampleBufferRef)photoSampleBuffer
previewPhotoSampleBuffer:(nullable CMSampleBufferRef)previewPhotoSampleBuffer
     resolvedSettings:(nonnull AVCaptureResolvedPhotoSettings *)resolvedSettings
      bracketSettings:(nullable AVCaptureBracketedStillImageSettings *)bracketSettings
                error:(nullable NSError *)error
API_AVAILABLE(ios(10.0)){
    if (error) {
        NSLog(@"captureOutput, error:%@", error.description);
        return;
    }
    if (_delegate && [_delegate respondsToSelector:@selector(cameraDevice:didCapturePhoto:previewImage:)]) {
        [_delegate cameraDevice:self
                    didCapturePhoto:photoSampleBuffer
                       previewImage:previewPhotoSampleBuffer];
    }
    
}

-(id<MTLTexture>)getMTLTexture:(CMSampleBufferRef)sampleBuffer {
    if (!_textureCache || !sampleBuffer) {
        return nil;
    }
    
    auto imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (!imageBuffer) {
        return nil;
    }
    
    return [self getMTLTextureFromImageBuffer:imageBuffer];
}

-(id<MTLTexture>)getMTLTextureFromImageBuffer:(CVImageBufferRef)imageBuffer {
    if (!_textureCache || !imageBuffer) {
        return nil;
    }
    
    auto width = CVPixelBufferGetWidth(imageBuffer);
    auto height = CVPixelBufferGetHeight(imageBuffer);
    
    CVMetalTextureRef texture_ref = nil;
    if (CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                  _textureCache,
                                                  imageBuffer,
                                                  nil,
                                                  //MTLPixelFormatBGRA8Unorm,
                                                  MTLPixelFormatRGBA8Unorm,
                                                  width,
                                                  height,
                                                  0,
                                                  &texture_ref) != kCVReturnSuccess) {
        NSLog(@"Error: CVMetalTextureCacheCreateTextureFromImage could not create a CVMetalTextureRef");
        return nil;
    }
    auto mtl_texture =  CVMetalTextureGetTexture(texture_ref);
    CVBufferRelease(texture_ref);
    return mtl_texture;
}

-(UIImage *)getUIImage:(CMSampleBufferRef)sampleBuffer {
    if (!sampleBuffer) {
        return nil;
    }
    
    auto imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (!imageBuffer) {
        return nil;
    }
    
    auto width = CVPixelBufferGetWidth(imageBuffer);
    auto height = CVPixelBufferGetHeight(imageBuffer);
    auto rect = CGRectMake(0, 0, width, height);
    
    auto ciImage = [CIImage imageWithCVPixelBuffer:imageBuffer];
    auto ciContext = [CIContext contextWithOptions:nil];
    auto cgImage = [ciContext createCGImage:ciImage fromRect:rect];
    if (!cgImage) {
        return nil;
    }
    
    return [UIImage imageWithCGImage:cgImage];
}

@end
