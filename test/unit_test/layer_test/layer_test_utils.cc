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

#include "test/unit_test/layer_test/layer_test_utils.h"
#include <fstream>
#include "tnn/utils/data_format_converter.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

std::vector<BlobDesc> CreateInputBlobsDesc(int batch, int channel, int input_size, int blob_count, DataType data_type) {
    // blob desc
    std::vector<BlobDesc> inputs_desc;
    for (int i = 0; i < blob_count; i++) {
        BlobDesc input_desc;
        input_desc.dims.push_back(batch);
        input_desc.dims.push_back(channel);
        input_desc.dims.push_back(input_size);
        input_desc.dims.push_back(input_size);
        input_desc.device_type = DEVICE_NAIVE;
        input_desc.data_type   = data_type;
        inputs_desc.push_back(input_desc);
    }
    return inputs_desc;
}

std::vector<BlobDesc> CreateInputBlobsDesc(int batch, int channel, int height, int width, int blob_count,
                                           DataType data_type) {
    // blob desc
    std::vector<BlobDesc> inputs_desc;
    for (int i = 0; i < blob_count; i++) {
        BlobDesc input_desc;
        input_desc.dims.push_back(batch);
        input_desc.dims.push_back(channel);
        input_desc.dims.push_back(height);
        input_desc.dims.push_back(width);
        input_desc.device_type = DEVICE_NAIVE;
        input_desc.data_type   = data_type;
        inputs_desc.push_back(input_desc);
    }
    return inputs_desc;
}

std::vector<BlobDesc> CreateOutputBlobsDesc(int blob_count, DataType data_type) {
    std::vector<BlobDesc> outputs_desc;
    for (int i = 0; i < blob_count; ++i) {
        BlobDesc output_desc;
        output_desc.data_type   = data_type;
        output_desc.device_type = DEVICE_NAIVE;
        outputs_desc.push_back(output_desc);
    }
    return outputs_desc;
}

int ReadBlobFromFile(Blob *blob, std::string path) {
    BlobDesc blob_desc = blob->GetBlobDesc();
    int data_count     = DimsVectorUtils::Count(blob_desc.dims);
    auto data_ptr      = (float *)blob->GetHandle().base;
    std::ifstream input_stream(path);
    int data_index = 0;
    while (!input_stream.eof() && data_count-- > 0) {
        float tmp;
        input_stream >> tmp;
        data_ptr[data_index++] = tmp;
    }
    input_stream.close();

    return 0;
}

int WriteBlobToFile(Blob *blob, std::string path) {
    BlobDesc blob_desc = blob->GetBlobDesc();
    int data_count     = DimsVectorUtils::Count(blob_desc.dims);
    auto data_ptr      = (float *)blob->GetHandle().base;
    std::ofstream output_stream(path);
    int data_index = 0;
    while (data_index < data_count) {
        output_stream << data_ptr[data_index++] << std::endl;
    }
    output_stream.close();

    return 0;
}

}  // namespace TNN_NS
