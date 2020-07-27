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
#import "TNNSSDObjectDetectorViewModel.h"
#import "TNNYoloObjectDetectorViewModel.h"

using namespace std;

@interface TNNExamplesListController () {
}
@end

@implementation TNNExamplesListController

- (void)viewDidLoad {
    [super viewDidLoad];
}

//#pragma mark - UITableViewDataSource
//- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView {
//    return 1;
//}
//
//- (NSInteger)tableView:(UITableView *)tableView
// numberOfRowsInSection:(NSInteger)section {
//    return 1;
//}
//
//- (UITableViewCell *)tableView:(UITableView *)tableView
// cellForRowAtIndexPath:(NSIndexPath *)indexPath {
//    return [super tableView:tableView cellForRowAtIndexPath:indexPath];
//}

//#pragma mark - UITableViewDelegate
- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    UIViewController *vc = nil;
    if (indexPath.section == 0) {
        vc = [self.storyboard instantiateViewControllerWithIdentifier:@"TNNFaceDetectorController"];
    } else if (indexPath.section == 1) {
        vc = [self.storyboard instantiateViewControllerWithIdentifier:@"TNNImageClassifyController"];
    } else if (indexPath.section == 2){
        vc = [self.storyboard instantiateViewControllerWithIdentifier:@"TNNImageColourController"];
    } else if (indexPath.section == 3){
        vc = [self.storyboard instantiateViewControllerWithIdentifier:@"TNNCameraPreviewController"];
        auto cameraViewController = (TNNCameraPreviewController*)vc;
        cameraViewController.viewModel = [TNNFaceDetectorViewModel new];
        cameraViewController.viewModel.title = @"人脸检测 - mbv2+SSD";
    } else if (indexPath.section == 4) {
        vc = [self.storyboard instantiateViewControllerWithIdentifier:@"TNNCameraPreviewController"];
        auto cameraViewController = (TNNCameraPreviewController*)vc;
        cameraViewController.viewModel = [TNNSSDObjectDetectorViewModel new];
        cameraViewController.viewModel.title = @"物体检测 - mbv2+SSD";
    } else if (indexPath.section == 5) {
        vc = [self.storyboard instantiateViewControllerWithIdentifier:@"TNNCameraPreviewController"];
        auto cameraViewController = (TNNCameraPreviewController*)vc;
        cameraViewController.viewModel = [TNNYoloObjectDetectorViewModel new];
        cameraViewController.viewModel.title = @"物体检测 - yolov5";
    } else if (indexPath.section == 6) {
        vc = [self.storyboard instantiateViewControllerWithIdentifier:@"TNNBlazefaceDetectorController"];
    }
    if (!vc) {
        return;
    }

    [self.navigationController setViewControllers:@[ vc ] animated:YES];
}

@end
