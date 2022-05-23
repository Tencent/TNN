// Copyright 2019 Tencent. All Rights Reserved

#ifndef TNN_SOURCE_DEVICE_DSP_SNPE_UTILS_HPP_
#define TNN_SOURCE_DEVICE_DSP_SNPE_UTILS_HPP_

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
#include "core/blob.h"

namespace TNN_NS {

std::unique_ptr<zdl::SNPE::SNPE> SetBuilderOptions(
    std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
    zdl::DlSystem::Runtime_t runtime, zdl::DlSystem::RuntimeList runtime_list,
    zdl::DlSystem::UDLBundle udlbundle, bool use_user_supplied_buffers,
    zdl::DlSystem::PlatformConfig platform_config, bool use_caching);

void CreateInputBufferMap(
    zdl::DlSystem::UserBufferMap& input_map, BlobMap& input_blobmap,
    std::unordered_map<std::string, std::vector<uint8_t>>& application_buffers,
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&
        snpe_userbacked_buffers,
    std::unique_ptr<zdl::SNPE::SNPE>& snpe, bool is_tf8_buffer);

void CreateOutputBufferMap(
    zdl::DlSystem::UserBufferMap& output_map, BlobMap& output_blobmap,
    std::unordered_map<std::string, std::vector<uint8_t>>& application_buffers,
    std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&
        snpe_userbacked_buffers,
    std::unique_ptr<zdl::SNPE::SNPE>& snpe, bool is_tf8_buffer);

}  // namespace TNN_NS

#endif  // TNN_SOURCE_DEVICE_DSP_SNPE_UTILS_HPP_
