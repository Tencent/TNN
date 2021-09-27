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

#include <chrono>
#include <random>
#include <string>
#include <vector>

#include "tnn/train/test_grad/test_layer_grad.h"
#include "tnn/train/grad/utils.h"

//TODO: 去掉arm依赖，改成设备无关的，否则后续加入其他设备支持时，编译不过
#include "tnn/device/arm/arm_util.h"

namespace TNN_NS {
namespace train {

template<typename T>
int InitRandom(T* host_data, size_t n, T range_min, T range_max, bool except_zero) {
    static std::mt19937 g(42);
    std::uniform_real_distribution<> rnd(range_min, range_max);

    for (unsigned long long i = 0; i < n; i++) {
        T val = static_cast<T>(rnd(g));
        while(except_zero && (float)val == 0.0f) val = static_cast<T>(rnd(g));
        host_data[i] = val;
    }

    return 0;
}

Status generate_raw_buffer(std::shared_ptr<RawBuffer>& buffer, RawBuffer& src_buffer, DeviceType device_type, DataFormat data_format, DataType data_type) {
    if(src_buffer.GetDataFormat() != DATA_FORMAT_NCHW || (data_format != DATA_FORMAT_NC4HW4 && data_format != DATA_FORMAT_NCHW)) {
        return Status(TNN_TRAIN_TEST_ERROR, "unsupport data format in generate_raw_buffer");
    }
    auto dims = src_buffer.GetBufferDims();
    auto total_size = CalculateElementCount(data_format, dims, data_type) * DataTypeUtils::GetBytesSize(data_type);
    buffer = std::make_shared<RawBuffer>(total_size, dims);
    buffer->SetDataFormat(data_format);
    buffer->SetDataType(data_type);
    if(!buffer->force_to<void *>()) {
        return Status(TNN_TRAIN_TEST_ERROR, "buffer is null, may be allocate error!");
    }
    if(data_format == DATA_FORMAT_NCHW)
        memcpy(buffer->force_to<void *>(), src_buffer.force_to<void *>(), total_size);
    else {
        PackFloatBlob(buffer->force_to<float *>(), src_buffer.force_to<float *>(), GetDim(dims, 0),
                    GetDim(dims, 1), DimsVectorUtils::Count(dims, 2));     
    }

    return TNN_OK;
}

Status generate_raw_buffer(std::map<Blob *, std::shared_ptr<RawBuffer>>& buffers, const BlobShapes& shapes, DeviceType device_type, DataFormat data_format, DataType data_type, bool generate_data, bool except_zero) {
    for(auto iter=shapes.begin(); iter != shapes.end(); ++iter) {
        std::shared_ptr<RawBuffer> buffer = std::make_shared<RawBuffer>(CalculateElementCount(data_format, iter->second, data_type) * DataTypeUtils::GetBytesSize(data_type), iter->second);
        buffer->SetDataFormat(data_format);
        buffer->SetDataType(data_type);
        if(!buffer->force_to<void *>()) {
            return Status(TNN_TRAIN_TEST_ERROR, "buffer is null, may be allocate error!");
        }
        if(generate_data) {
            float* data_ptr = buffer->force_to<float *>();
            InitRandom(static_cast<float *>(data_ptr), buffer->GetDataCount(), -1.0f, 1.0f, except_zero);
        }
        buffers[iter->first] = buffer;
    }
    return TNN_OK;
}

Status generate_blob(std::vector<Blob*>& blobs, NameBuffers& shapes, DeviceType device_type, DataFormat data_format, DataType data_type, bool generate_data, bool except_zero) {
    for(auto iter=shapes.begin(); iter != shapes.end(); ++iter) {
        if(iter->second.GetDataFormat() != DATA_FORMAT_NCHW || (data_format != DATA_FORMAT_NC4HW4 && data_format != DATA_FORMAT_NCHW)) {
            return Status(TNN_TRAIN_TEST_ERROR, "unsupport data format in generate_blob");
        }
        BlobDesc desc;
        desc.device_type = device_type;
        desc.data_type   = DATA_TYPE_FLOAT;
        desc.name        = iter->first;
        desc.data_format = data_format;
        desc.dims = iter->second.GetBufferDims();
        BlobHandle handle;
        Blob* blob = nullptr;
        if(generate_data) {
            auto device = GetDevice(desc.device_type);
            if(device == nullptr) {
                return Status(TNN_TRAIN_TEST_ERROR, "get device error");
            }
            BlobMemorySizeInfo size_info = device->Calculate(desc);
            
            device->Allocate(&handle, size_info);
            blob = new Blob(desc, handle);
            
            if(data_format == DATA_FORMAT_NC4HW4 ) {
                PackFloatBlob(static_cast<float*>(GetBlobHandle(blob)), iter->second.force_to<float *>(), GetDim(desc.dims, 0),
                            GetDim(desc.dims, 1), DimsVectorUtils::Count(desc.dims, 2));
            } else {
                memcpy(GetBlobHandle(blob), iter->second.force_to<float *>(), CalculateElementCount(desc) * DataTypeUtils::GetBytesSize(desc.data_type));
            }
        }  else {
            blob = new Blob(desc, handle);
        }   
        blobs.push_back(blob);
    }
    return TNN_OK;
}

Status generate_blob(std::vector<Blob*>& blobs, const NameShapes& shapes, DeviceType device_type, DataFormat data_format, DataType data_type, bool generate_data, bool except_zero) {
    for(auto iter=shapes.begin(); iter != shapes.end(); ++iter) {
        BlobDesc desc;
        desc.device_type = device_type;
        desc.data_type   = DATA_TYPE_FLOAT;
        desc.name        = iter->first;
        desc.data_format = data_format;
        desc.dims = iter->second;
        BlobHandle handle;
        Blob* blob;
        if(generate_data) {
            auto device = GetDevice(desc.device_type);
            if(device == nullptr) {
                return Status(TNN_TRAIN_TEST_ERROR, "get device error");
            }
            BlobMemorySizeInfo size_info = device->Calculate(desc);
            device->Allocate(&handle, size_info);
        

            blob = new Blob(desc, handle);
            if(blob->GetHandle().base == nullptr) {
                return Status(TNN_TRAIN_TEST_ERROR, "handle is null, may be allocate error!");
            }
            void* data_ptr = GetBlobHandle(blob);
            InitRandom(static_cast<float *>(data_ptr), CalculateElementCount(desc), -1.0f, 1.0f, except_zero);
        } else {
           blob = new Blob(desc, handle); 
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