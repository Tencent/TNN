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

#include "unary_op_layer_interpreter.h"

REGISTER_UNARY_OP_LAYER_INTERPRETER(Cos, LAYER_COS);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Acos, LAYER_ACOS);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Sin, LAYER_SIN);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Asin, LAYER_ASIN);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Tan, LAYER_TAN);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Atan, LAYER_ATAN);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Log, LAYER_LOG);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Reciprocal, LAYER_RECIPROCAL);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Floor, LAYER_FLOOR);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Ceil, LAYER_CEIL);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Softsign, LAYER_SOFTSIGN);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Sigmoid, LAYER_SIGMOID);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Rsqrt, LAYER_RSQRT);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Erf, LAYER_ERF);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Softplus, LAYER_SOFTPLUS);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Neg, LAYER_NEG);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Not, LAYER_NOT);

REGISTER_UNARY_OP_LAYER_INTERPRETER(Swish, LAYER_SWISH);
