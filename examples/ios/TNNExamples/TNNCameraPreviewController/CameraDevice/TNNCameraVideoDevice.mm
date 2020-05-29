//  Copyright © 2020 tencent. All rights reserved.

#import "TNNCameraVideoDevice.h"
#import <AVFoundation/AVFoundation.h>

@interface TNNCameraVideoDevice ()<AVCaptureVideoDataOutputSampleBufferDelegate>
{
    dispatch_queue_t videoProcessingQueue;
}


@property (nonatomic, strong) AVCaptureSession *captureSession;
@property (nonatomic, strong) AVCaptureDevice *captureDevice;
@property (nonatomic, strong) AVCaptureDeviceInput *captureDeviceInput;
@property (nonatomic, strong) AVCaptureStillImageOutput *stillImageOutput;
@property (nonatomic, strong) AVCaptureVideoDataOutput *videoOutput;
@property (nonatomic, strong) AVCaptureVideoPreviewLayer *videoPreviewLayer;
@property (nonatomic, weak) UIView *previewView;

@property (nonatomic, strong) dispatch_queue_t bufferQueue;

@end

@implementation TNNCameraVideoDevice


- (void)setupCaptureSession {
    // 1.创建会话
    self.captureSession = [[AVCaptureSession alloc] init];
    self.captureSession.sessionPreset = AVCaptureSessionPreset640x480;
    // 2.创建输入设备
    self.captureDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    NSArray *array = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    for (AVCaptureDevice *dev in array) {
        if (dev.position == AVCaptureDevicePositionFront) self.captureDevice = dev;
    }
    
    // 3.创建输入
    NSError *error = nil;
    self.captureDeviceInput = [AVCaptureDeviceInput deviceInputWithDevice:self.captureDevice error:&error];
    // 3.创建输出
    self.stillImageOutput = [[AVCaptureStillImageOutput alloc] init];
    self.stillImageOutput.outputSettings = @{AVVideoCodecKey : AVVideoCodecJPEG};
    // 4.连接输入与会话
    if ([self.captureSession canAddInput:self.captureDeviceInput]) {
        [self.captureSession addInput:self.captureDeviceInput];
    }
    // 5.连接输出与会话
    if ([self.captureSession canAddOutput:self.stillImageOutput]) {
        [self.captureSession addOutput:self.stillImageOutput];
    }
    
    //初始化Queue
    videoProcessingQueue = dispatch_queue_create("com.tencent.tnn.videoProcessingQueue", NULL);
    self.bufferQueue = dispatch_queue_create("com.tencent.tnn.videoBuffer", NULL);
    [self initVideoOutput];
    self.videoPreviewLayer = [AVCaptureVideoPreviewLayer layerWithSession:self.captureSession];
    self.videoPreviewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill;
}

- (id)initWithPreviewView:(UIView *)view
{
    self = [super init];
    if (self) {
        self.previewView = view;
        [self setupCaptureSession];
        [self.previewView.layer addSublayer:self.videoPreviewLayer];
    }
    return self;
}

- (void)startSession {
    if(![self.captureSession isRunning]) {
        [self.captureSession startRunning];
    }
}

- (void)stopSession {
    if([self.captureSession isRunning]) {
        [self.captureSession stopRunning];
    }
}

- (void)initVideoOutput
{
    self.videoOutput = [[AVCaptureVideoDataOutput alloc] init];
    self.videoOutput.alwaysDiscardsLateVideoFrames = YES;
    self.videoOutput.videoSettings = @{ (NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA) };
    [self.videoOutput setSampleBufferDelegate:self queue:videoProcessingQueue];
    if ([self.captureSession canAddOutput:self.videoOutput]) {
        [self.captureSession addOutput:self.videoOutput];
    } else {
        NSLog(@"couldn't add video output");
    }
    AVCaptureConnection *connection =
    [self.videoOutput connectionWithMediaType:AVMediaTypeVideo];
    connection.videoOrientation = AVCaptureVideoOrientationPortrait;
}

- (AVCaptureDevicePosition)rotateCamera
{
    AVCaptureDevicePosition pos = AVCaptureDevicePositionUnspecified;
    //Change camera source
    if(self.captureSession)
    {
        //Indicate that some changes will be made to the session
        [self.captureSession beginConfiguration];

        //Remove existing input
        AVCaptureInput* currentCameraInput = [self.captureSession.inputs objectAtIndex:0];
        [self.captureSession removeInput:currentCameraInput];

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
        self.captureDevice = newCamera;
        //Add input to session
        NSError *err = nil;
        self.captureDeviceInput = [[AVCaptureDeviceInput alloc] initWithDevice:self.captureDevice error:&err];
        if(!self.captureDeviceInput || err)
        {
          NSLog(@"Error creating capture device input: %@", err.localizedDescription);
        }
        else
        {
          [self.captureSession addInput:self.captureDeviceInput];
        }
        [self.captureSession removeOutput:self.videoOutput];
        [self initVideoOutput];
        //Commit all the configuration changes at once
        [self.captureSession commitConfiguration];
    }
    return pos;
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

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary *)change context:(void *)context
{
}

- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection
{
    AVCaptureInputPort *inputPort = [connection.inputPorts objectAtIndex:0];
    AVCaptureDeviceInput *input = (AVCaptureDeviceInput *)inputPort.input;
    AVCaptureDevicePosition inputPos = input.device.position;
    
    if (inputPos == AVCaptureDevicePositionUnspecified) {
        NSLog(@"captureOutput, camera position unspecified, drop it!!!!");
        return;
    }
    static NSInteger nProcessFrame = 0;
    if (nProcessFrame > 0) {
        return;    //last frame not finished
    }
    nProcessFrame++;
    __block CMSampleBufferRef currentBuffer;
    CMSampleBufferCreateCopy(kCFAllocatorDefault, sampleBuffer, &currentBuffer);
    dispatch_async(self.bufferQueue, ^{
        NSDictionary *args = @{@"buffer": @((NSInteger)currentBuffer), @"position": @(inputPos)};
        if (self.delegate && [self.delegate respondsToSelector:@selector(cameraDeviceEvent:withAguments:)]) {
            [self.delegate cameraDeviceEvent:CameraDeviceEvent_FrameReceived withAguments:args];
        }
        CFRelease(currentBuffer);
        nProcessFrame--;
    });
    
}

@end
