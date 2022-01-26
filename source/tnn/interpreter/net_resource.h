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

#ifndef TNN_SOURCE_TNN_INTERPRETER_NET_RESOURCE_H_
#define TNN_SOURCE_TNN_INTERPRETER_NET_RESOURCE_H_

#include <map>
#include <set>
#include "tnn/interpreter/layer_resource.h"

namespace TNN_NS {

struct NetResource {
    std::map<std::string, std::shared_ptr<LayerResource>> resource_map;
    ConstantResource constant_map;
    
    //data flag of constant blobs
    ConstantResourceFlag constant_blob_flags;
    
    //names of constant layer whose output blob data flag is DATA_FLAG_CHANGE_NEVER or DATA_FLAG_CHANGE_IF_SHAPE_DIFFER
    std::set<std::string> constant_layers;
    
    //names of constant layer whose output blob data flag is DATA_FLAG_CHANGE_IF_SHAPE_DIFFER
    std::set<std::string> shape_differ_layers;
    
    //default shape map, also it is max shape map corresponding to max_inputs_shape in Instance.Init
    BlobShapesMap blob_shapes_map;
    //min shape map, corresponding to min_inputs_shape in Instance.Init
    BlobShapesMap min_blob_shapes_map;
    
    //data type for input and output blobs
    BlobDataTypeMap blob_datatype_map;
    
};

DataType GetNetResourceDataType(NetResource *resource);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_INTERPRETER_NET_RESOURCE_H_
