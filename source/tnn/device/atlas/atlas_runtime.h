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

#ifndef TNN_SOURCE_TNN_DEVICE_ATLAS_ATLAS_RUNTIME_H_
#define TNN_SOURCE_TNN_DEVICE_ATLAS_ATLAS_RUNTIME_H_

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include "tnn/core/status.h"
#include "tnn/device/atlas/atlas_common_types.h"

namespace TNN_NS {

class AtlasRuntime {
public:
    static AtlasRuntime *GetInstance();
    static Status Init();
    static void DecreaseRef();

    ~AtlasRuntime();
    AtlasRuntime(const AtlasRuntime &) = delete;
    AtlasRuntime &operator=(const AtlasRuntime &) = delete;

    Status SetDevice(int device_id);
    Status AddModelInfo(Blob *blob, AtlasModelInfo model_info);
    Status DelModelInfo(Blob *blob);
    std::map<Blob *, AtlasModelInfo> &GetModleInfoMap();

private:
    AtlasRuntime();

private:
    std::set<int> device_list_;
    std::map<Blob *, AtlasModelInfo> model_info_map_;

    static std::shared_ptr<AtlasRuntime> atlas_runtime_singleton_;
    static int ref_count_;
    static bool init_done_;
};

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_OPENCL_OPENCL_RUNTIME_H_
