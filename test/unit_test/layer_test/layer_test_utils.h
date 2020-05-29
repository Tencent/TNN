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

#ifndef TNN_TEST_UNIT_TEST_LAYER_TEST_LAYER_TEST_UTILS_H_
#define TNN_TEST_UNIT_TEST_LAYER_TEST_LAYER_TEST_UTILS_H_

#include "tnn/core/blob.h"

namespace TNN_NS {

std::vector<BlobDesc> CreateInputBlobsDesc(int batch, int channel, int size, int blob_count, DataType data_type);

std::vector<BlobDesc> CreateInputBlobsDesc(int batch, int channel, int height, int width, int blob_count,
                                           DataType data_type);

std::vector<BlobDesc> CreateOutputBlobsDesc(int blob_count, DataType data_type);

int ReadBlobFromFile(Blob* blob, std::string path);
int WriteBlobToFile(Blob* blob, std::string path);
}  // namespace TNN_NS

#endif  // TNN_TEST_UNIT_TEST_LAYER_TEST_LAYER_TEST_UTILS_H_
