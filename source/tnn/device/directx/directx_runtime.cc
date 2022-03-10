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

#include "tnn/device/directx/directx_runtime.h"


#include <stdio.h>
#include <sstream>
#include <mutex>
#include <algorithm>

#include <d3dcommon.h>
#include <d3d11.h>
#undef min
#undef max

#include "tnn/core/macro.h"
#include "tnn/utils/md5.h"

namespace TNN_NS {

std::shared_ptr<DirectXRuntime> DirectXRuntime::g_singleton_ = nullptr;
std::mutex DirectXRuntime::g_mutex_;

DirectXRuntime *DirectXRuntime::GetInstance() {
    std::unique_lock<std::mutex> lck(g_mutex_);
    if (!g_singleton_) {
        DirectXRuntime * rt = new DirectXRuntime(); 
        if (rt->Init() != TNN_OK)  {
            return nullptr;
        }
        g_singleton_ = std::shared_ptr<DirectXRuntime>(rt);
    }

    return g_singleton_.get();
}

DirectXRuntime::DirectXRuntime() {
}

Status DirectXRuntime::Init() {
    return TNN_OK;
}

DirectXRuntime::~DirectXRuntime() {
    context_.reset();
    device_.reset();
}

std::shared_ptr<ID3D11Device> DirectXRuntime::Device() {
    return device_;
}


std::shared_ptr<ID3D11DeviceContext> DirectXRuntime::Context() {
    return context_;
}

std::vector<size_t> DirectXRuntime::GetTexture2DMaxSize() {
    return {4096, 4096};
}

}  // namespace TNN_NS
