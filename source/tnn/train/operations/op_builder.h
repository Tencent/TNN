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

#ifndef TNN_SOURCE_TNN_TRAIN_OPERATIONS_OP_BUILDER_H
#define TNN_SOURCE_TNN_TRAIN_OPERATIONS_OP_BUILDER_H
#include "tnn/train/operations/base_op.h"
#include "tnn/train/operations/op_type.h"
namespace TNN_NS {
namespace train {
ParamWrapper BinaryOpHelper(ParamWrapper& input1,  ParamWrapper& input2, TrainContext& context, ElementOpType element_op_type) {
    if(input1.IsEmpty() || input2.IsEmpty()) 
        return {};
    if(!input1.IsBlobOrRawbuffer() || !input2.IsBlobOrRawbuffer())
        return {};
    std::shared_ptr<RawBuffer> output = std::make_shared<RawBuffer>(input1.GetBlobOrRawbufferSize(), input1.GetBlobOrRawbufferDims());
    output->SetDataType(input1.GetBlobOrRawbufferDatatype());
    output->SetDataFormat(input1.GetBlobOrRawbufferDataformat());
    ParamWrappers outputs = {ParamWrapper(output)};
    Status status = BaseOp::RunOp(OpType::ElementOp, {input1, input2}, {ParamWrapper(output)}, {ParamWrapper(element_op_type)}, context);
    if(status != TNN_OK)
        return {};
    return outputs[0];
};
ParamWrapper UnaryOpHelper(ParamWrapper& input1, TrainContext& context, ElementOpType element_op_type) {
    if(input1.IsEmpty()) 
        return {};
    if(!input1.IsBlobOrRawbuffer())
        return {};
    std::shared_ptr<RawBuffer> output = std::make_shared<RawBuffer>(input1.GetBlobOrRawbufferSize(), input1.GetBlobOrRawbufferDims());
    output->SetDataType(input1.GetBlobOrRawbufferDatatype());
    ParamWrappers outputs = {ParamWrapper(output)};
    Status status = BaseOp::RunOp(OpType::ElementOp, {input1}, {ParamWrapper(output)}, {ParamWrapper(element_op_type)}, context);
    if(status != TNN_OK)
        return {};
    return outputs[0];
};
ParamWrapper _Add(ParamWrapper input1,  ParamWrapper input2, TrainContext& context) {
    return BinaryOpHelper(input1, input2, context, ElementOpType::Add);
}

ParamWrapper _Div(ParamWrapper input1,  ParamWrapper input2, TrainContext& context) {
    return BinaryOpHelper(input1, input2, context, ElementOpType::Div);
}

ParamWrapper _Mul(ParamWrapper input1,  ParamWrapper input2, TrainContext& context) {
    return BinaryOpHelper(input1, input2, context, ElementOpType::Mul);
}

ParamWrapper _Sub(ParamWrapper input1,  ParamWrapper input2, TrainContext& context) {
    return BinaryOpHelper(input1, input2, context, ElementOpType::Sub);
}

ParamWrapper _Neg(ParamWrapper input1, TrainContext& context) {
    return UnaryOpHelper(input1, context, ElementOpType::Neg);
}

ParamWrapper _Log(ParamWrapper input1, TrainContext& context) {
    return UnaryOpHelper(input1, context, ElementOpType::Log);
}

ParamWrapper _Const(ParamWrapper input1, DimsVector dims, DataFormat data_format) {
    std::shared_ptr<RawBuffer> output;
    if(input1.Isint()) {
        output = std::make_shared<RawBuffer>(sizeof(int) * DimsVectorUtils::Count(dims), dims);
        output->SetDataType(DATA_TYPE_INT32);
        output->SetDataFormat(data_format);      
    } else if (input1.Isfloat()){
        output = std::make_shared<RawBuffer>(sizeof(float) * DimsVectorUtils::Count(dims), dims);
        output->SetDataType(DATA_TYPE_FLOAT); 
        output->SetDataFormat(data_format);
    } else if (input1.IsBfp16()) {
        output = std::make_shared<RawBuffer>(sizeof(bfp16_t) * DimsVectorUtils::Count(dims), dims);
        output->SetDataType(DATA_TYPE_BFP16); 
        output->SetDataFormat(data_format);       
    }
    return {};
}

} // train
} //TNN_NS

#endif //TNN_SOURCE_TNN_TRAIN_OPERATIONS_OP_BUILDER_H