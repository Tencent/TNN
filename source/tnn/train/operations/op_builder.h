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

ParamWrapper _Add(ParamWrapper input1, ParamWrapper input2, TrainContext &context);
ParamWrapper _Div(ParamWrapper input1, ParamWrapper input2, TrainContext &context);

ParamWrapper _Mul(ParamWrapper input1, ParamWrapper input2, TrainContext &context);

ParamWrapper _Sub(ParamWrapper input1, ParamWrapper input2, TrainContext &context);

ParamWrapper _Neg(ParamWrapper input1, TrainContext &context);

ParamWrapper _Log(ParamWrapper input1, TrainContext &context);

ParamWrapper _Const(ParamWrapper input1, DimsVector dims, DataFormat data_format);

// for relu, x > 0 return 1, x <=0 return 0;
ParamWrapper _RSign(ParamWrapper input1, TrainContext &context);

} // namespace train
} // namespace TNN_NS

#endif // TNN_SOURCE_TNN_TRAIN_OPERATIONS_OP_BUILDER_H