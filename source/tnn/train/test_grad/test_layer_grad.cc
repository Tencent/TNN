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

// author: sanerzheng@tencent.com

#include "tnn/train/test_grad/test_layer_grad.h"
#include "tnn/utils/random_data_utils.h"
#include "tnn/train/grad/utils.h"

namespace TNN_NS {
namespace train {

using NameShapes = std::vector<std::pair<std::string, DimsVector>>;
using BlobShapes = std::vector<std::pair<Blob*, DimsVector>>;
Status generate_raw_buffer(std::map<Blob *, std::shared_ptr<RawBuffer>>& buffers, const BlobShapes& shapes, DeviceType device_type, DataFormat data_format, DataType data_type, bool generate_data) {
    for(auto iter=shapes.begin(); iter != shapes.end(); ++iter) {
        std::shared_ptr<RawBuffer> buffer = std::make_shared<RawBuffer>(CalculateElementCount(data_format, iter->second, data_type) * DataTypeUtils::GetBytesSize(data_type), iter->second);
        buffer->SetDataFormat(data_format);
        buffer->SetDataType(data_type);
        if(!buffer->force_to<void *>()) {
            return Status(TNN_TRAIN_TEST_ERROR, "buffer is null, may be allocate error!");
        }
        if(generate_data) {
            float* data_ptr = buffer->force_to<float *>();
            InitRandom(static_cast<float *>(data_ptr), buffer->GetDataCount(), -1.0f, 1.0f);
        }
        buffers[iter->first] = buffer;
    }
    return TNN_OK;
}

Status generate_blob(std::vector<Blob*>& blobs, const NameShapes& shapes, DeviceType device_type, DataFormat data_format, DataType data_type, bool generate_data) {
    for(auto iter=shapes.begin(); iter != shapes.end(); ++iter) {
        BlobDesc desc;
        desc.device_type = device_type;
        desc.data_type   = DATA_TYPE_FLOAT;
        desc.name        = iter->first;
        desc.data_format = data_format;
        desc.dims = iter->second;
        Blob* blob = new Blob(desc, generate_data);
        if(generate_data && blob->GetHandle().base == nullptr) {
            return Status(TNN_TRAIN_TEST_ERROR, "handle is null, may be allocate error!");
        }
        if(generate_data) {
            void* data_ptr = static_cast<void *>(static_cast<char *>(blob->GetHandle().base) + blob->GetHandle().bytes_offset);
            InitRandom(static_cast<float *>(data_ptr), CalculateElementCount(desc), -1.0f, 1.0f);
        }
        
        blobs.push_back(blob);
    }
    return TNN_OK;
}

void free_blobs(std::vector<Blob*>& blobs) {
    for(size_t i = 0; i<blobs.size(); ++i) {
        delete blobs[i];
    }
}

void ouput_data(void* data, const DimsVector dims, const std::string name) {
    float* data_ptr = static_cast<float *>(data);
    int batch = dims[0];
    int channel = dims.size() > 1 ? dims[1] : 1;
    int hw = DimsVectorUtils::Count(dims, 2);
    LOGD("%s shape: n:%d, c:%d, hw:%d", name.c_str(), batch, channel, hw);
    for (int i = 0; i <batch;  ++i) {
        std::string tmp;
        for(int j=0; j<channel*hw; ++j) {
            tmp += std::to_string(j) + ":" + std::to_string(data_ptr[i * channel * hw + j]) + ",";
        }
        LOGD("blob:%s %d n values %s",name.c_str(), i, tmp.c_str());
    }
}

void output_buffer(RawBuffer* buffer, const std::string name) {
    void* data = buffer->force_to<void *>();
    auto dims = buffer->GetBufferDims();
    RawBuffer tmpbuffer;
    std::string print_name = name.empty() ? "rawbuffer_defautl" : name;
    if(buffer->GetDataFormat() == DATA_FORMAT_NC4HW4) {
        ConvertToNCHW(data, tmpbuffer, buffer); 
    }
    ouput_data(data, dims, print_name);
}

void output_blob(Blob* blob, const std::string name) {
    void* data = static_cast<void*>(static_cast<char *>(blob->GetHandle().base) + blob->GetHandle().bytes_offset);
    auto dims = blob->GetBlobDesc().dims;
    std::string print_name = name.empty() ? blob->GetBlobDesc().name : name;
    ouput_data(data, dims, print_name);
}

Status LayerGradTestManager::RunTestGrad() {
    auto& grad_test_map = GetLayerGradTestMap();
    Status status;
    for(auto iter: grad_test_map) {
        status = iter.second->TestGrad();
        RETURN_ON_NEQ(status, TNN_OK);
    }
    return status;
}   
}
}