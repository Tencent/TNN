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

#ifndef TNN_INCLUDE_TNN_UTILS_BLOB_TRANSFER_UTILS_H
#define TNN_INCLUDE_TNN_UTILS_BLOB_TRANSFER_UTILS_H

#include "tnn/core/blob.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"

namespace TNN_NS {

class RawBuffer;

Status CopyToDevice(Blob *dst, Blob *src, void *command_queue);
Status CopyFromDevice(Blob *blob, Blob *src, void *command_queue);

// @brief transfer blob to rawbuffer. The device of blob must be DEVICE_NAIVE
Status Blob2RawBuffer(Blob *blob, std::shared_ptr<RawBuffer> &buffer);
// @brief transfer rawbuffer to blob. The device of blob will be DEVICE_NAIVE
Status RawBuffer2Blob(RawBuffer *buffer, std::shared_ptr<Blob> &blob);
}  // namespace TNN_NS

#endif  // TNN_INCLUDE_TNN_UTILS_BLOB_TRANSFER_UTILS_H
