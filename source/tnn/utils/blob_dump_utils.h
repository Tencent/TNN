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

#ifndef TNN_SOURCE_TNN_UTILS_BLOB_DUMP_UTILS_H_
#define TNN_SOURCE_TNN_UTILS_BLOB_DUMP_UTILS_H_

#include <string>

#include "tnn/core/blob.h"
#include "tnn/core/context.h"
#include "tnn/core/status.h"

#define DUMP_INPUT_BLOB 0
#define DUMP_OUTPUT_BLOB 0

namespace TNN_NS {

#if (DUMP_INPUT_BLOB || DUMP_OUTPUT_BLOB)
extern PUBLIC std::string g_tnn_dump_directory;
#endif

Status DumpDeviceBlob(Blob* dst, Context* context, std::string fname_prefix);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_UTILS_BLOB_DUMP_UTILS_H_
