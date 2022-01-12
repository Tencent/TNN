// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#import "TNNExamplesListController.h"
#import "TNNCameraPreviewController.h"
#import "TNNFaceDetectorViewModel.h"
#import "TNNBlazeFaceDetectorViewModel.h"
#import "TNNSSDObjectDetectorViewModel.h"
#import "TNNYoloObjectDetectorViewModel.h"
#import "TNNNanodetObjectDetectorViewModel.h"
#import "TNNFaceDetectAlignerViewModel.h"
#import "TNNFaceDetectMeshViewModel.h"
#import "TNNHairSegmentationViewModel.h"
#import "TNNMonoDepthViewModel.h"
#import "TNNPoseDetectLandmarkViewModel.h"
#import "TNNSkeletonDetectorViewModel.h"
#import "TNNOCRViewModel.h"

#import "TNNExamplesListCell.h"

using namespace std;

@interface TNNExampleData : NSObject
@property (strong, nonatomic) NSString *title;
@property (strong, nonatomic) NSString *desc;

@property (strong, nonatomic) NSString *viewControllerID;
@property (strong, nonatomic) TNNViewModel *viewModel;
@end

@implementation TNNExampleData
@end



@interface TNNExamplesListController () {
}
@property (strong, nonatomic) NSArray<TNNExampleData*> *examples;
@end

@implementation TNNExamplesListController

- (void)viewDidLoad {
    [super viewDidLoad];
    [self clearNavigationBarLeft];
    [self forceToOrientation:UIDeviceOrientationPortrait];
    
    [self setupTNNExampleDataSource];
}

- (BOOL)shouldAutorotate {
    return YES;
}

- (UIInterfaceOrientationMask)supportedInterfaceOrientations {
    return UIInterfaceOrientationMaskPortrait;
}

- (void)setupTNNExampleDataSource {
    auto examples = [NSMutableArray array];

    //人脸检测 - Ultra Fast
    {
        auto data = [TNNExampleData new];
        data.title = @"人脸检测 - Ultra Fast";
        data.desc = @"图像类 - 单输入多输出";
        data.viewControllerID = @"TNNFaceDetectorController";
        [examples addObject:data];
    }
    
    //图像分类 - SqueezeNet
    {
        auto data = [TNNExampleData new];
        data.title = @"图像分类 - SqueezeNet";
        data.desc = @"图像类 - 单输入多输出";
        data.viewControllerID = @"TNNImageClassifyController";
        [examples addObject:data];
    }
    
    //灰度图上色
    {
        auto data = [TNNExampleData new];
        data.title = @"灰度图上色 Recolor";
        data.desc = @"图像类 - 单输入（图）多输出（图）";
        data.viewControllerID = @"TNNImageColourController";
        [examples addObject:data];
    }
    
    //人脸检测 - Blazeface
    {
        auto data = [TNNExampleData new];
        data.title = @"人脸检测 - Blazeface";
        data.desc = @"图像类 - 单输入多输出";
        data.viewControllerID = @"TNNBlazefaceDetectorController";
        [examples addObject:data];
    }
    
    //阅读理解 - TinyBert
    {
        auto data = [TNNExampleData new];
        data.title = @"阅读理解 - TinyBert";
        data.desc = @"自然语言类 - 单输入多输出";
        data.viewControllerID = @"TNNReadingComprehensionController";
        [examples addObject:data];
    }

    //人脸检测 - Blazeface
    {
        auto data = [TNNExampleData new];
        data.title = @"人脸检测 - Blazeface";
        data.desc = @"摄像头 - 单输入多输出";
        data.viewControllerID = @"TNNCameraPreviewController";
        {
            data.viewModel = [TNNBlazeFaceDetectorViewModel new];
            data.viewModel.title = data.title;
            data.viewModel.preferFrontCamera = true;
        }
        [examples addObject:data];
    }
    
    //物体检测 - mbv2+SSD
    {
        auto data = [TNNExampleData new];
        data.title = @"物体检测 - mbv2+SSD";
        data.desc = @"摄像头 - 单输入多输出";
        data.viewControllerID = @"TNNCameraPreviewController";
        {
            data.viewModel = [TNNSSDObjectDetectorViewModel new];
            data.viewModel.title = data.title;
        }
        [examples addObject:data];
    }
    
    //物体检测 - yolov5
    {
        auto data = [TNNExampleData new];
        data.title = @"物体检测 - yolov5";
        data.desc = @"摄像头 - 单输入多输出";
        data.viewControllerID = @"TNNCameraPreviewController";
        {
            data.viewModel = [TNNYoloObjectDetectorViewModel new];
            data.viewModel.title = data.title;
        }
        [examples addObject:data];
    }

    //物体检测 - nanodet
    {
        auto data = [TNNExampleData new];
        data.title = @"物体检测 - nanodet";
        data.desc = @"摄像头 - 单输入多输出";
        data.viewControllerID = @"TNNCameraPreviewController";
        {
            data.viewModel = [TNNNanodetObjectDetectorViewModel new];
            data.viewModel.title = data.title;
        }
        [examples addObject:data];
    }
    
    //人脸检测配准 - 腾讯优图
    {
        auto data = [TNNExampleData new];
        data.title = @"人脸检测配准 - 腾讯优图实验室";
        data.desc = @"摄像头 - 单输入多输出";
        data.viewControllerID = @"TNNCameraPreviewController";
        {
            data.viewModel = [TNNFaceDetectAlignerViewModel new];
            data.viewModel.title = @"人脸检测配准 - 腾讯优图实验室";
            data.viewModel.preferFrontCamera = true;
        }
        [examples addObject:data];
    }
    
    //人脸稠密点 - Facemesh
    {
        auto data = [TNNExampleData new];
        data.title = @"人脸稠密点 - Facemesh";
        data.desc = @"摄像头 - 单输入多输出";
        data.viewControllerID = @"TNNCameraPreviewController";
        {
            data.viewModel = [TNNFaceDetectMeshViewModel new];
            data.viewModel.title = @"Facemesh";
            data.viewModel.preferFrontCamera = true;
        }
        [examples addObject:data];
    }

    //头发分割 - HairSegmentation
    {
        auto data = [TNNExampleData new];
        data.title = @"头发分割 - 腾讯光影实验室";
        data.desc = @"摄像头 - 单输入单输出";
        data.viewControllerID = @"TNNCameraPreviewController";
        {
            data.viewModel = [TNNHairSegmentationViewModel new];
            data.viewModel.title = @"头发分割 - 腾讯光影实验室";
            data.viewModel.preferFrontCamera = true;
        }
        [examples addObject:data];
    }
    
    //深度图 - mono depth pydnet
    {
        auto data = [TNNExampleData new];
        data.title = @"深度图 - mono depth - pydnet";
        data.desc = @"摄像头 - 单输入单输出";
        data.viewControllerID = @"TNNCameraPreviewController";
        {
            data.viewModel = [TNNMonoDepthViewModel new];
            data.viewModel.title = @"深度图 - monodepth - pydnet";
            data.viewModel.preferDeviceOrientation = UIDeviceOrientationLandscapeRight;
        }
        [examples addObject:data];
    }
    
    //人体姿势关键点 - BlazePose
    {
        auto data = [TNNExampleData new];
        data.title = @"人体关键点 - BlazePose";
        data.desc = @"摄像头 - 单输入多输出";
        data.viewControllerID = @"TNNCameraPreviewController";
        {
            data.viewModel = [TNNPoseDetectLandmarkViewModel new];
            data.viewModel.title = @"BlazePose";
            data.viewModel.preferFrontCamera = false;
        }
        [examples addObject:data];
    }
    
    //人体关键点 - SkeletonDetector
    {
        auto data = [TNNExampleData new];
        data.title = @"人体关键点 - 腾讯微视";
        data.desc = @"摄像头 - 单输入单输出";
        data.viewControllerID = @"TNNCameraPreviewController";
        {
            data.viewModel = [TNNSkeletonDetectorViewModel new];
            data.viewModel.title = @"人体关键点 - 腾讯微视";
            data.viewModel.preferFrontCamera = false;
        }
        [examples addObject:data];
    }

#if HAS_OPENCV
    //光学字符识别 - OCR
    {
        auto data = [TNNExampleData new];
        data.title = @"OCR";
        data.desc = @"摄像头 - 单输入多输出";
        data.viewControllerID = @"TNNCameraPreviewController";
        {
            data.viewModel = [TNNOCRViewModel new];
            data.viewModel.title = @"OCR";
            data.viewModel.preferFrontCamera = false;
        }
        [examples addObject:data];
    }
#endif

    self.examples = examples;
}

#pragma mark - UITableViewDataSource
- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView {
    return self.examples.count;
}

- (NSInteger)tableView:(UITableView *)tableView
 numberOfRowsInSection:(NSInteger)section {
    return 1;
}

- (UITableViewCell *)tableView:(UITableView *)tableView
 cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    auto cell = (TNNExamplesListCell *)[tableView dequeueReusableCellWithIdentifier:@"TNNExamplesListCell"];
    auto data = self.examples[indexPath.section];
    cell.labelTitle.text = data.title;
    cell.labelDesc.text = data.desc;
    return cell;
}

//#pragma mark - UITableViewDelegate
- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    auto data = self.examples[indexPath.section];
    auto *vc = [self.storyboard instantiateViewControllerWithIdentifier:data.viewControllerID];
    if (![vc isKindOfClass:TNNExamplesController.class]) {
        LOGE("view controller must be subclass of TNNExamplesController\n");
        return;
    }
    
    auto exampleController = (TNNExamplesController *)vc;
    exampleController.viewModel = data.viewModel;

    [self.navigationController pushViewController:vc animated:YES];
}

@end
