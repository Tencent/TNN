// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TNN_SOURCE_TNN_DEVICE_SNPE_SNPE_UTILS_H_
#define TNN_SOURCE_TNN_DEVICE_SNPE_SNPE_UTILS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlSystem/IUserBufferFactory.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"

#include "tnn/core/blob.h"

namespace TNN_NS {

zdl::DlSystem::Runtime_t SelectSNPERuntime(std::string prefered_runtime = "GPU");

std::unique_ptr<zdl::SNPE::SNPE> SetBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::RuntimeList runtime_list,
                                                   bool use_user_supplied_buffers,
                                                   zdl::DlSystem::PlatformConfig platform_config,
                                                   bool use_caching,
                                                   zdl::DlSystem::StringList outputs);

void CreateInputBufferMap(zdl::DlSystem::UserBufferMap& input_map,
                          BlobMap& input_blobmap,
                          std::unordered_map<std::string, std::vector<uint8_t>>& application_buffers,
                          std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpe_userbacked_buffers,
                          std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                          bool is_tf8_buffer);

void CreateOutputBufferMap(zdl::DlSystem::UserBufferMap& output_map,
                           BlobMap& output_blobmap,
                           std::unordered_map<std::string, std::vector<uint8_t>>& application_buffers,
                           std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpe_userbacked_buffers,
                           std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                           bool is_tf8_buffer);

void LoadUdoPackages(const std::string& package_dir = "/data/local/tmp/tnn-test/lib");

}  // namespace TNN_NS

#endif  // TNN_SOURCE_TNN_DEVICE_SNPE_SNPE_UTILS_H_
