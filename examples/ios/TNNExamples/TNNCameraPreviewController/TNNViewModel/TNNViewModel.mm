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

#import "TNNViewModel.h"
#include <sys/utsname.h>

// utsname.machine has device identifier. For example, identifier for iPhone Xs is "iPhone11,2".
// Since Neural Engine is only available for use on A12 and later, major device version in the
// identifier is checked for these models:
// A12: iPhone XS (11,2), iPad Mini - 5th Gen (11,1)
// A12X: iPad Pro - 3rd Gen (8,1)
// For more information, see https://www.theiphonewiki.com/wiki/Models
bool HasAppleNPU() {
    //check hardware
    struct utsname system_info;
    uname(&system_info);

    if (strncmp("iPad", system_info.machine, 4) == 0) {
    const int major_version = atoi(system_info.machine + 4);
    return major_version >= 8;  // There are no device between iPad 8 and 11.
    }
    else if (strncmp("iPhone", system_info.machine, 6) == 0) {
    const int major_version = atoi(system_info.machine + 6);
    return major_version >= 11;
    }
    else if (strncmp("MacBookPro", system_info.machine, 10) == 0) {
      const int major_version = atoi(system_info.machine + 10);
      return major_version >= 17;
    }
    else if (strncmp("MacBookAir", system_info.machine, 10) == 0) {
      const int major_version = atoi(system_info.machine + 10);
      return major_version >= 10;
    }
    else if (strncmp("iMac", system_info.machine, 4) == 0) {
      const int major_version = atoi(system_info.machine + 4);
      return major_version >= 21;
    }
    else if (strncmp("Macmini", system_info.machine, 7) == 0) {
      const int major_version = atoi(system_info.machine + 7);
      return major_version >= 9;
    }
    return false;
}

@implementation TNNViewModel
- (instancetype)init {
    self = [super init];
    if (self) {
        _preferDeviceOrientation = UIDeviceOrientationPortrait;
    }
    return self;
}

- (NSString *)title {
    if (_title.length <= 0) {
        return NSStringFromClass(self.class);
    } else {
        return _title;
    }
}

- (TNNComputeUnits)preferComputeUnits {
    if (HasAppleNPU()) {
        return TNNComputeUnitsAppleNPU;
    } else {
        return TNNComputeUnitsCPU;
    }
}

-(Status)loadNeuralNetworkModel:(TNNComputeUnits)units {
    [NSException raise:NSInvalidArgumentException format:@"subclass must overide the func loadNeuralNetworkModel"];
    return TNN_OK;
}


-(std::vector<std::shared_ptr<ObjectInfo> >)getObjectList:(std::shared_ptr<TNNSDKOutput>)sdk_output {
    [NSException raise:NSInvalidArgumentException format:@"subclass must overide the func getObjectList"];
    return {};
}

-(ImageInfo)getImage:(std::shared_ptr<TNNSDKOutput>)sdk_output {
    return {};
}

-(NSString*)labelForObject:(std::shared_ptr<ObjectInfo>)object {
    [NSException raise:NSInvalidArgumentException format:@"subclass must overide the func labelForObject"];
    return nil;
}

- (BOOL)showImageAtMinorPreview {
    return NO;
}

- (void)setupCustomView:(UIView *)view layoutHeight:(NSLayoutConstraint *)viewLayoutHeight {
    if (view && viewLayoutHeight) {
        viewLayoutHeight.constant = 0;
    }
}

- (void)adajustStackPrevieView:(UIStackView *)stackView {
}
@end
