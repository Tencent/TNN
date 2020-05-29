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

#ifndef TNN_METAL_MACRO_H_
#define TNN_METAL_MACRO_H_

#include "tnn/core/macro.h"

#if !defined(__APPLE__)
#define TNN_METAL_ENABLED 0
#else
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <float.h>
#endif

#ifdef __OBJC__
#define TNN_OBJC_CLASS(name) @class name
#else
#define TNN_OBJC_CLASS(name) typedef struct objc_object name
#endif //__OBJC__

#if defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#define TNN_TARGET_IPHONE 1
#define TNN_TARGET_OSX 0
#else
#define TNN_TARGET_IPHONE 0
#define TNN_TARGET_OSX 1
#endif
#endif

#ifndef TNN_METAL_ENABLED
#define TNN_METAL_ENABLED (!(TARGET_OS_IPHONE && TARGET_OS_SIMULATOR))
#endif

#ifndef TNN_METAL_DEBUG
#if DEBUG
#define TNN_METAL_DEBUG 1
#else
#define TNN_METAL_DEBUG 0
#endif
#endif

#endif // TNN_METAL_MACRO_H_
