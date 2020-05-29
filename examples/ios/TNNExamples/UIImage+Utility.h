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

#import <UIKit/UIKit.h>
#include <memory>
#include <tuple>

namespace utility {
/**
@brief convert uiimage to rgba raw data, resize to height x width
@param image uimage
@param height target image height
@param width target image width
*/
std::shared_ptr<char> UIImageGetData(UIImage *image, int height, int width);

/**
 @brief convert image rgba raw data to uiimage
 @param image_data rgba raw data pointer
 @param height image height
 @param width image width
 */
UIImage *UIImageWithDataRGBA(void *image_data, int height, int width);

/**
@brief convert imageBuffer to rgba raw data, resize to height x width
@param imageBuffer image buffer
@param height target image height
@param width target image width
*/
std::shared_ptr<char> CVImageBuffRefGetData(CVImageBufferRef imageBuffer, int height, int width);

UIImage *UIImageWithCVImageBuffRef(CVImageBufferRef imageBuffer);
}
