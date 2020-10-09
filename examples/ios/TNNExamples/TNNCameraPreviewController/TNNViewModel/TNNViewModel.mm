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

@implementation TNNViewModel
- (NSString *)title {
    if (_title.length <= 0) {
        return NSStringFromClass(self.class);
    } else {
        return _title;
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

- (void)setupCustomView:(UIView *)view layoutHeight:(NSLayoutConstraint *)viewLayoutHeight {
    if (view && viewLayoutHeight) {
        viewLayoutHeight.constant = 0;
    }
}
@end
