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

#include "test/unit_test/layer_test/layer_test.h"
#if defined(TNN_METAL_DEBUG) && defined(__APPLE__)
#import <Foundation/Foundation.h>
#endif

namespace TNN_NS {

void LayerTest::RunForMetal(std::shared_ptr<AbstractModelInterpreter> interp, Precision precision, DataFormat cpu_input_data_format, DataFormat device_input_data_format) {
#if defined(TNN_METAL_DEBUG) && defined(__APPLE__)
    @autoreleasepool{
#endif

        DoRun(interp, precision, cpu_input_data_format, device_input_data_format);

#if defined(TNN_METAL_DEBUG) && defined(__APPLE__)
    }
#endif
}

}  // namespace TNN_NS
