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

#include <fstream>
#include <iostream>
#include <sstream>
#include "onnx_converter_multidir_broadcast.h"
#include "onnx_utility.h"

DECLARE_MULTI_BROADCASR_OP_CONVERTER(Less);

string OnnxOpConverterLess::TNNOpType(NodeProto& node, OnnxNetInfo& net_info) {
    return "Less";
}

string OnnxOpConverterLess::TNNLayerParam(NodeProto& node, OnnxNetInfo& net_info) {
    return OnnxOpConverterMultiBrodcast::TNNLayerParam(node, net_info);
}

bool OnnxOpConverterLess::HasLayerResource(NodeProto& node, OnnxNetInfo& net_info) {
    return OnnxOpConverterMultiBrodcast::HasLayerResource(node, net_info);
}

int OnnxOpConverterLess::WriteTNNModel(Serializer* net_writer, NodeProto& node, OnnxNetInfo& net_info) {
    return OnnxOpConverterMultiBrodcast::WriteTNNModel(net_writer, node, net_info);
}

REGISTER_MULTI_BROADCASR_OP_CONVERTER(Less, Less);
