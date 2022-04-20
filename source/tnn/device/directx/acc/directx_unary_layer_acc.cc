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

#include "tnn/device/directx/acc/directx_unary_layer_acc.h"

namespace TNN_NS {

namespace directx {

Status DirectXUnaryLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                  const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Unary Acc\n");

    // set output_dims_size_ here, because ReloadConstant blobs needs this value ORZ.
    output_dims_size_ = outputs[0]->GetBlobDesc().dims.size();
    Status ret        = DirectXLayerAcc::Init(context, param, resource, inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    data_type_ = inputs[0]->GetBlobDesc().data_type;

    return TNN_OK;
}

DirectXUnaryLayerAcc::~DirectXUnaryLayerAcc() {}

Status DirectXUnaryLayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    auto d3d_context = GetID3DContext();

    std::shared_ptr<DirectXMemory> in_memory, out_memory;
    RETURN_ON_NEQ(DirectXMemoryManager::GetInstance()->GetRefMemoryFromBlob(inputs[0], in_memory), TNN_OK);
    RETURN_ON_NEQ(DirectXMemoryManager::GetInstance()->GetRefMemoryFromBlob(outputs[0], out_memory), TNN_OK);

    auto in_srv = in_memory->GetSRV();
    auto out_uav = out_memory->GetUAV();

//    LOGD("kernel name: %s\n",kernel_name_.c_str());
    std::shared_ptr<ID3D11ComputeShader> cs;
    Status ret = GetShaderByName(kernel_name_, cs);
    RETURN_ON_NEQ(ret, TNN_OK);

    const int batch         = DimsFunctionUtils::GetDim(output_dims_, 0);
    const int output_height = DimsFunctionUtils::GetDim(output_dims_, 2);
    const int output_width  = DimsFunctionUtils::GetDim(output_dims_, 3);
    const int channels      = DimsFunctionUtils::GetDim(output_dims_, 1);

    const int channel_blocks    = UP_DIV(channels, 4);

    ret = DispatchShader(cs, {in_srv}, {out_uav}, {const_buffer_.get()}, {batch * output_height,output_width,channel_blocks});

    return ret;

}

Status DirectXUnaryLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Binary Acc Reshape\n");
    Status ret = DirectXLayerAcc::Reshape(inputs, outputs);
    RETURN_ON_NEQ(ret, TNN_OK);

    return CalcParam(inputs, outputs);
}

Status DirectXUnaryLayerAcc::CalcParam(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {

    typedef struct launch_param {
        DirectX::XMUINT4 id;     //input_dim
        DirectX::XMUINT4 od;     //output_dim
    } launch_param_t;

    launch_param_t args;
    args.id = DirectX::XMUINT4(DimsFunctionUtils::GetDim(input_dims_, 0), DimsFunctionUtils::GetDim(input_dims_, 1),
                              DimsFunctionUtils::GetDim(input_dims_, 2), DimsFunctionUtils::GetDim(input_dims_, 3));
    args.od = DirectX::XMUINT4(DimsFunctionUtils::GetDim(output_dims_, 0), DimsFunctionUtils::GetDim(output_dims_, 1),
                              DimsFunctionUtils::GetDim(output_dims_, 2), DimsFunctionUtils::GetDim(output_dims_, 3));

    Status ret = CreateConstBuffer<launch_param_t>(args, GetID3DDevice(), const_buffer_);
    RETURN_ON_NEQ(ret, TNN_OK);

    return TNN_OK;
}

Status DirectXUnaryLayerAcc::ReloadConstantBlobs(const std::vector<Blob *> &inputs,
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
double DirectXUnaryLayerAcc::GetBandwidth() {

    auto get_num_elements = [](unsigned int stride[6], unsigned int out_dim[6]) {
        for(int i=0;i<6;i++) {
            if (stride[i] > 0) {
                return stride[i] * out_dim[i];
            }
        }
        return 1u;
    };

    unsigned int input_stride_[6];
    unsigned int output_dim_[6];
    memset(input_stride_, 0u, 6 * sizeof(unsigned int));
    unsigned int all_one[6] = {1, 1, 1, 1, 1, 1};
    std::swap(output_dim_, all_one);

    for(int i=0;i<output_dims_.size();i++) {
        output_dim_[i] = output_dims_[i];
    }

    for(int i=0;i<input_dims_.size();i++) {
        if (input_dims_[i] > 1) {
            input_stride_[i] = DimsVectorUtils::Count(input_dims_, i+1);
        }
    }

    size_t a_size_in_elements = get_num_elements(input_stride_, output_dim_);
    size_t c_size_in_elements = DimsVectorUtils::Count(output_dims_);
    size_t ele_size_in_bytes = DataTypeUtils::GetBytesSize(data_type_);

    return double( a_size_in_elements + c_size_in_elements ) * ele_size_in_bytes;
}
#endif // TNN_PROFILE

} // namespace directx

}  // namespace TNN_NS
