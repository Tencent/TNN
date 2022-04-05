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

#include "tnn/device/directx/acc/directx_binary_layer_acc.h"

#include <directxmath.h>
#include <directxpackedvector.h>

#include "tnn/core/macro.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/device/directx/directx_memory.h"
#include "tnn/device/directx/directx_util.h"
// #include "tnn/device/opencl/imagebuffer_convertor.h"

namespace TNN_NS {

namespace directx {

Status DirectXBinaryLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Binary Acc\n");

    // set output_dims_size_ here, because ReloadConstant blobs needs this value ORZ.
    output_dims_size_ = outputs[0]->GetBlobDesc().dims.size();
    Status ret        = DirectXLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    auto broadcast_param = dynamic_cast<MultidirBroadcastLayerParam *>(param);
    CHECK_PARAM_NULL(broadcast_param);
    broadcast_param_ = *broadcast_param;


    EltwiseLayerResource *layer_res = dynamic_cast<EltwiseLayerResource *>(resource);
    if (layer_res == nullptr) {
        if (inputs.size() != 2) {
            return Status(TNNERR_PARAM_ERR, "inputs size shound be 2 without binary resource");
        }
    } else {
        param_dims_ = layer_res->element_shape;
        if (inputs.size() != 1) {
            return Status(TNNERR_PARAM_ERR, "input size should be 1");
        }

        float *data_ptr = layer_res->element_handle.force_to<float *>();
        std::shared_ptr<float> data = std::shared_ptr<float>(data_ptr, [](float *){});
        if (layer_res->element_handle.GetDataType() != DATA_TYPE_FLOAT) {
            data = GetFloatFromRawBuffer(layer_res->element_handle);  
            if (data == nullptr) {
                return Status(TNNERR_DX_ACC_INIT_ERR, "convert res to float failed");
            }
        } 
        RETURN_ON_NEQ(ConvertParam(data.get(), param_dims_), TNN_OK);
    }

    data_type_ = inputs[0]->GetBlobDesc().data_type;

    return CalcStrides(inputs, outputs);
}

DirectXBinaryLayerAcc::~DirectXBinaryLayerAcc() {}

Status DirectXBinaryLayerAcc::CalcStrides(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    memset(input_a_stride_, 0u, 6 * sizeof(unsigned int));
    memset(input_b_stride_, 0u, 6 * sizeof(unsigned int));
    unsigned int all_one[6] = {1, 1, 1, 1, 1, 1}; 
    std::swap(output_dim_, all_one);

    auto out_blob_dims = outputs[0]->GetBlobDesc().dims;
    auto in_blob_a_dims = inputs[0]->GetBlobDesc().dims;
    DimsVector in_blob_b_dims;

    if (inputs.size() > 1) {
        in_blob_b_dims = inputs[1]->GetBlobDesc().dims;
    } else {
        in_blob_b_dims = param_dims_;
        if (broadcast_param_.weight_input_index == 0) {
            std::swap(in_blob_a_dims, in_blob_b_dims);
        }
    }

    for(int i=0;i<out_blob_dims.size();i++) {
        output_dim_[i] = out_blob_dims[i];
    }

    for(int i=0;i<in_blob_a_dims.size();i++) {
        if (in_blob_a_dims[i] > 1) {
            input_a_stride_[i] = DimsVectorUtils::Count(in_blob_a_dims, i+1);
        }
    }

    for(int i=0;i<in_blob_b_dims.size();i++) {
        if (in_blob_b_dims[i] > 1) {
            input_b_stride_[i] = DimsVectorUtils::Count(in_blob_b_dims, i+1);
        }
    }

    typedef struct launch_param {
        DirectX::XMUINT4 sa_0;
        DirectX::XMUINT4 sa_3;
        DirectX::XMUINT4 sb_0;
        DirectX::XMUINT4 sb_3;
        DirectX::XMUINT4 od_0;
        DirectX::XMUINT4 od_3;
    } launch_param_t;

    launch_param_t args;
    args.sa_0 = DirectX::XMUINT4(input_a_stride_[0], input_a_stride_[1], input_a_stride_[2], 0);
    args.sa_3 = DirectX::XMUINT4(input_a_stride_[3], input_a_stride_[4], input_a_stride_[5], 0);
    args.sb_0 = DirectX::XMUINT4(input_b_stride_[0], input_b_stride_[1], input_b_stride_[2], 0);
    args.sb_3 = DirectX::XMUINT4(input_b_stride_[3], input_b_stride_[4], input_b_stride_[5], 0);
    args.od_0 = DirectX::XMUINT4(output_dim_[0], output_dim_[1], output_dim_[2], 0);
    args.od_3 = DirectX::XMUINT4(output_dim_[3], output_dim_[4], output_dim_[5], 0);

    Status ret = CreateConstBuffer<launch_param_t>(args, GetID3DDevice(), const_buffer_);
    RETURN_ON_NEQ(ret, TNN_OK);

    return TNN_OK;
}

Status DirectXBinaryLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    auto d3d_context = GetID3DContext();

    auto in_memory = DirectXMemory::CreateRefMemoryFromBlob(inputs[0]); 
    auto out_memory = DirectXMemory::CreateRefMemoryFromBlob(outputs[0]); 

    auto in_srv = in_memory->GetSRV();
    auto out_uav = out_memory->GetUAV();

    std::vector<std::shared_ptr<ID3D11ShaderResourceView>> in_srvs;
    if (broadcast_param_.weight_input_index == 1) {
        in_srvs = {in_srv,  binary_params_->GetSRV() };
    } else {
        in_srvs = {binary_params_->GetSRV(), in_srv};
    }

    std::shared_ptr<ID3D11ComputeShader> cs;
    Status ret = GetShaderByName(kernel_name_, cs);
    RETURN_ON_NEQ(ret, TNN_OK);

    const int THREADS_PER_BLOCK = 128;
    const int ELE_PER_THREAD    = 4;

    const int ele_count = DimsVectorUtils::Count(outputs[0]->GetBlobDesc().dims);

    ret = DispatchShader(cs, in_srvs, {out_uav}, {const_buffer_.get()}, {UP_DIV(ele_count, THREADS_PER_BLOCK * ELE_PER_THREAD)});

    return ret;
}

Status DirectXBinaryLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Binary Acc Reshape\n");
    Status ret = DirectXLayerAcc::Reshape(inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    return CalcStrides(inputs, outputs);
}

Status DirectXBinaryLayerAcc::ConvertParam(float *param_data_ptr, std::vector<int> param_dims) {

    // copy param data into DirectX Buffer
    // TODO: to DirectX Texture2D 
    shared_ptr<DirectXMemory> param_buffer = DirectXMemory::CreateBufferMemoryFromHost(
                                                param_data_ptr, param_dims, DATA_TYPE_FLOAT, DATA_FORMAT_NCHW);
    if (!param_buffer) {
        LOGE("param transfer to GPU failed.");
        return Status(TNNERR_DX_BUFFER_ALOCATE_ERR, "param transfer to GPU failed.");
    }
    binary_params_ = std::move(param_buffer);
    return TNN_OK;
}

Status DirectXBinaryLayerAcc::ReloadConstantBlobs(const std::vector<Blob *> &inputs,
                                                 bool only_reload_shape_differ_blob) {
    auto const_resource      = const_resource_;
    auto const_resource_flag = const_resource_flag_;
    auto const_blob_map      = const_blob_map_;
    for (auto iter : inputs) {
        auto name = iter->GetBlobDesc().name;
        if (const_resource == nullptr || const_resource->find(name) == const_resource->end()) {
            continue;
        }

        if (only_reload_shape_differ_blob && const_resource_flag &&
            const_resource_flag->find(name) == const_resource_flag->end()) {
            continue;
        }

        auto buffer                = (*const_resource)[name];
        std::shared_ptr<Blob> blob = nullptr;
        if (const_blob_map.find(name) != const_blob_map.end()) {
            blob = const_blob_map[name];
        }
        auto buffer_dims = buffer->GetBufferDims();
        if (output_dims_size_ != buffer_dims.size()) {
            std::shared_ptr<RawBuffer> new_buffer(new RawBuffer(*buffer));
            int diff = output_dims_size_ - buffer_dims.size();
            for (int i = 0; i < diff; i++) {
                buffer_dims.insert(buffer_dims.begin(), 1);
            }
            new_buffer->SetBufferDims(buffer_dims);
            buffer = new_buffer;
        }
        auto status = RawBuffer2DirectXBlob(buffer.get(), blob);
        RETURN_ON_NEQ(status, TNN_OK);

        blob->SetFlag(DATA_FLAG_CHANGE_NEVER);
        auto dims            = iter->GetBlobDesc().dims;
        auto data_type_size  = DataTypeUtils::GetBytesSize(iter->GetBlobDesc().data_type);
        const_blob_map[name] = blob;
        iter->SetHandle(blob->GetHandle());
        iter->GetBlobDesc() = blob->GetBlobDesc();
        LOGD("Reload constant blob: %s\n", name.c_str());
    }
    const_blob_map_ = const_blob_map;
    return TNN_OK;
}

#if TNN_PROFILE
double DirectXBinaryLayerAcc::GetBandwidth() {

    auto get_num_elements = [](unsigned int stride[6], unsigned int out_dim[6]) {
        for(int i=0;i<6;i++) {
            if (stride[i] > 0) {
                return stride[i] * out_dim[i];
            }
        }
        return 1u;
    };

    size_t a_size_in_elements = get_num_elements(input_a_stride_, output_dim_);
    size_t b_size_in_elements = get_num_elements(input_b_stride_, output_dim_);
    size_t c_size_in_elements = DimsVectorUtils::Count(output_dims_);
    size_t ele_size_in_bytes = DataTypeUtils::GetBytesSize(data_type_);

    return double( a_size_in_elements + b_size_in_elements + c_size_in_elements ) * ele_size_in_bytes;
}
#endif // TNN_PROFILE

} // namespace directx

}  // namespace TNN_NS
