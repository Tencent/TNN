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
#include "tnn/train/grad/utils.h"
#include "tnn/core/default_network.h"
#include "tnn/layer/multidir_broadcast_layer.h"
#include "tnn/utils/random_data_utils.h"


namespace TNN_NS {
namespace train {
DECLARE_LAYER_GRAD_TEST_BEGIN(Binary, LAYER_ADD); 
virtual Status TestArmAdd();
virtual Status TestArmSub();
virtual Status TestArmMul();
virtual Status TestNaiveMul();
DECLARE_LAYER_GRAD_TEST_END

int get_out_pos(int offset, DimsVector& in_dims, DimsVector& out_dims, int axis, int axis_offset) {
    DimsVector pos;
    pos.resize(in_dims.size(), 0);
    for(int i=in_dims.size()-1;i>=0;--i) {
        pos[i] = offset % in_dims[i];
        offset = offset / in_dims[i];
    }
    pos[axis] += axis_offset;
    int res = 0;
    for(int i=out_dims.size()-1;i>=0;--i) {
        res += pos[i] * DimsVectorUtils::Count(out_dims, i+1);
    }
    return res;
}

Status check_result(std::shared_ptr<RawBuffer> output_grad, std::shared_ptr<RawBuffer> input_grad, int axis, LayerType layer_type) {
    std::shared_ptr<RawBuffer> out_nchw;
    std::shared_ptr<RawBuffer> in_nchw;
    ConvertToNCHW(output_grad, out_nchw);
    ConvertToNCHW(input_grad, in_nchw);
    float* in_ptr = in_nchw->force_to<float *>();
    float* out_ptr = out_nchw->force_to<float *>();
    auto out_dims = out_nchw->GetBufferDims();
    auto in_dims = in_nchw->GetBufferDims();

    if(out_dims.size() != in_dims.size()) {
        LOGE("in and out dims's size not math");
        return Status(TNN_TRAIN_TEST_ERROR, "in and out dims's size not math");        
    }
    DimsVector cur_in_pos, cur_out_pos;
    cur_in_pos.resize(in_dims.size(), 0);
    cur_out_pos.resize(in_dims.size(), 0);
    int total_in = DimsVectorUtils::Count(in_dims);
    for(int i=0; i<total_in; ++i) {
        int out_pos = get_out_pos(i, in_dims, out_dims, axis, axis_offset);
        if(in_ptr[i] != out_ptr[out_pos]) {
            LOGE("input %d item value not math with output %d", i, out_pos);
            return Status(TNN_TRAIN_TEST_ERROR, "input item value not math with output");              
        }
    }
    return TNN_OK;
}

Status test_concat_base(NameBuffers& input_buffer, NameBuffers& outputs_blobs_shapes, DimsVector& weight_shape, DeviceType device_type,int weight_input_index, LayerType layer_type) {
    TrainContext context;
    context.network = new DefaultNetwork();
    NetworkConfig* config = new NetworkConfig();
    context.config = config;
    config->device_type = device_type;
    config->precision = PRECISION_HIGH;
    DataFormat data_format ;
    if(device_type == DEVICE_ARM)
        data_format = DATA_FORMAT_NC4HW4;
    else if(device_type == DEVICE_NAIVE)
        data_format = DATA_FORMAT_NCHW;
    else 
        return Status(TNN_TRAIN_TEST_ERROR, "not support device in test_concat_base");

    BaseLayer* cur_layer = new MultidirBroadcastLayer(layer_type); //fake for add/sub/mul/div layer
    cur_layer->layer_name_ = "test_binary";


    LayerParam* layer_param = new MultidirBroadcastLayerParam();
    dynamic_cast<MultidirBroadcastLayerParam*>(layer_param)->weight_input_index = weight_input_index;
    cur_layer->param_ = layer_param;

    LayerResource* layer_res = new EltwiseLayerResource();
    int weight_count = DimsVectorUtils::Count(weight_shape);
    if( weight_count> 0) {
        EltwiseLayerResource* res = dynamic_cast<EltwiseLayerResource *>(layer_res);
        res->element_shape = weight_shape;
        res->element_handle = RawBuffer(weight_count * DataTypeUtils::GetBytesSize(DATA_TYPE_FLOAT));
        float* data = res->element_handle.force_to<float *>();
        InitRandom(static_cast<float *>(data), weight_count, -1.0f, 1.0f, true);
    }

    Status status; 
    generate_blob(cur_layer->input_blobs_, input_blobs_shapes, device_type, data_format, DATA_TYPE_FLOAT, true, true);
    generate_blob(cur_layer->output_blobs_, outputs_blobs_shapes, device_type, data_format, DATA_TYPE_FLOAT, false, false);
    BlobShapes buffer_shapes = {
        {cur_layer->output_blobs_[0], outputs_blobs_shapes[0].second}
    };
    generate_raw_buffer(context.backward_grads_blob, buffer_shapes, device_type, data_format, DATA_TYPE_FLOAT, true);
    status = LayerGrad::GetLayerGradMap()[layer_type]->OnGrad(cur_layer, context);
    RETURN_ON_NEQ(status, TNN_OK);
    auto output_grad0 = context.backward_grads_blob[cur_layer->output_blobs_[0]];
    output_buffer(output_grad0.get(), "output_grad");
    int axis_offset = 0;
    for(int i=0; i<cur_layer->input_blobs_.size(); ++i) {
        auto desc = cur_layer->input_blobs_[i]->GetBlobDesc();
        output_buffer(context.backward_grads_blob[cur_layer->input_blobs_[i]].get(), desc.name + "_grad");
        LOGD("test %d %s", i, desc.name.c_str());
        status = check_result(output_grad0, context.backward_grads_blob[cur_layer->input_blobs_[i]], axis, axis_offset);
        RETURN_ON_NEQ(status, TNN_OK);
        axis_offset += desc.dims[axis];
    }

    delete config;
    delete layer_param;
    delete cur_layer;
    delete context.network;
    free_blobs(cur_layer->output_blobs_);
    free_blobs(cur_layer->input_blobs_);
    return TNN_OK; 
}

Status ConcatLayerGradTest::TestArmConcatCommon() {
    NameShapes input_blobs_shapes = {
        {"input0", {2, 3, 2}},
        {"input1", {2, 3, 4}},
        {"input2", {2, 3, 1}}
    };
    NameShapes outputs_blobs_shapes = {
        {"output", {2, 3, 7}},
    };
    LOGD("TestArmConcatCommon start ");
    return test_concat_base(input_blobs_shapes, outputs_blobs_shapes, DEVICE_ARM, 2);
}

Status ConcatLayerGradTest::TestArmConcatChannel() {
    NameShapes input_blobs_shapes = {
        {"input0", {2, 2, 3}},
        {"input1", {2, 4, 3}},
        {"input2", {2, 1, 3}}
    };
    NameShapes outputs_blobs_shapes = {
        {"output", {2, 7, 3}},
    };
    LOGD("TestArmConcatChannel start ");
    return test_concat_base(input_blobs_shapes, outputs_blobs_shapes, DEVICE_ARM, 1);
}

Status ConcatLayerGradTest::TestArmConcatChannelC4() {
    NameShapes input_blobs_shapes = {
        {"input0", {2, 4, 3}},
        {"input1", {2, 4, 3}},
        {"input2", {2, 4, 3}}
    };
    NameShapes outputs_blobs_shapes = {
        {"output", {2, 12, 3}},
    };
    LOGD("TestArmConcatChannelC4 start ");
    return test_concat_base(input_blobs_shapes, outputs_blobs_shapes, DEVICE_ARM, 1);
}

Status ConcatLayerGradTest::TestNaiveNchw() {
    NameShapes input_blobs_shapes = {
        {"input0", {2, 2, 3}},
        {"input1", {2, 4, 3}},
        {"input2", {2, 1, 3}}
    };
    NameShapes outputs_blobs_shapes = {
        {"output", {2, 7, 3}},
    };
    LOGD("TestNaiveNchw start ");
    return test_concat_base(input_blobs_shapes, outputs_blobs_shapes, DEVICE_NAIVE, 1);
}

Status ConcatLayerGradTest::TestGrad() {
    Status status;
    status = TestNaiveNchw();
    RETURN_ON_NEQ(status, TNN_OK);
    status = TestArmConcatCommon();
    RETURN_ON_NEQ(status, TNN_OK);
    status = TestArmConcatChannel();
    RETURN_ON_NEQ(status, TNN_OK);
    status = TestArmConcatChannelC4();
    RETURN_ON_NEQ(status, TNN_OK);
    return TNN_OK;
}
REGISTER_LAYER_GRAD_TEST(Binary, LAYER_ADD);
}
}