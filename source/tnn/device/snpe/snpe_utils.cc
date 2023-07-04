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

#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorShape.hpp"
#include "SNPE/SNPEFactory.hpp"

#include "tnn/core/macro.h"
#include "tnn/device/snpe/snpe_utils.h"

namespace TNN_NS {

zdl::DlSystem::Runtime_t SelectSNPERuntime(std::string prefered_runtime) {
    zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;

    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::CPU) &&
        !zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
        LOGE("Error: SNPE cannot run on both CPU and GPU, perhaps you are not running on a Qualcomm Snapdragon SoC.\n");
    }

    if (prefered_runtime == "CPU") {
        if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::CPU)) {
            LOGI("Run TNN SNPE on Selected CPU device.\n");
            return zdl::DlSystem::Runtime_t::CPU;
        }
    }
    if (prefered_runtime == "GPU") {
        if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
            LOGI("Run TNN SNPE on Selected GPU device.\n");
            return zdl::DlSystem::Runtime_t::GPU;
        }
    }
    if (prefered_runtime == "DSP") {
        if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP)) {
            LOGI("Run TNN SNPE on Selected DSP device.\n");
            return zdl::DlSystem::Runtime_t::DSP;
        }
    }

    // Else Select GPU -> CPU
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
        LOGI("Run TNN SNPE on GPU device.\n");
        return zdl::DlSystem::Runtime_t::GPU;
    }

    return runtime;
}

size_t CalcSizeFromDims(const zdl::DlSystem::Dimension* dims,
                        size_t rank,
                        size_t element_size) {
    if (rank == 0) {
        return 0;
    }
    size_t size = element_size;
    while (rank--) {
        (*dims == 0) ? size *= 0 : size *= *dims;
        dims++;
    }
    return size;
}

std::unique_ptr<zdl::SNPE::SNPE> SetBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer>& container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::RuntimeList runtime_list,
                                                   bool use_user_supplied_buffers,
                                                   zdl::DlSystem::PlatformConfig platform_config,
                                                   bool use_caching,
                                                   zdl::DlSystem::StringList outputs) {
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

    if (runtime_list.empty()) {
        runtime_list.add(runtime);
    }

    snpe = snpeBuilder.setOutputLayers(outputs)
               .setRuntimeProcessorOrder(runtime_list)
               .setUseUserSuppliedBuffers(use_user_supplied_buffers)
               .setPlatformConfig(platform_config)
               .setInitCacheMode(use_caching)
               .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
               .build();
    return snpe;
}

void CreateUserBuffer(zdl::DlSystem::UserBufferMap& user_buffer_map,
                      BlobMap& blobmap,
                      std::unordered_map<std::string, std::vector<uint8_t>>& application_buffers,
                      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&snpe_userbacked_buffers,
                      std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                      const char* name,
                      const bool is_tf8_buffer) {
    // get attributes of buffer by name
    auto buffer_attr = snpe->getInputOutputBufferAttributes(name);
    if (!buffer_attr) {
        LOGE("Error while creating user buffer, TNN SNPE GetInputBufferAttributes ERROR.\n");
        throw std::runtime_error(std::string("TNN SNPE: Error obtaining attributes for input tensor ") + name);
    }

    // calculate the size of buffer required by the input tensor
    const zdl::DlSystem::TensorShape& buffer_shape = (*buffer_attr)->getDims();
    
    // Calculate the stride based on buffer strides.
    // Note: Strides = Number of bytes to advance to the next element in each
    // dimension. For example, if a float tensor of dimension 2x4x3 is tightly
    // packed in a buffer of 96 bytes, then the strides would be (48,12,4) Note:
    // Buffer stride is usually known and does not need to be calculated.
    std::vector<size_t> strides(buffer_shape.rank());
    strides[strides.size() - 1] = is_tf8_buffer ? sizeof(uint8_t) : sizeof(float);
    size_t stride = strides[strides.size() - 1];
    for (size_t i = buffer_shape.rank() - 1; i > 0; i--) {
        (buffer_shape[i] == 0) ? stride *= 0 : stride *= buffer_shape[i];
        strides[i - 1] = stride;
    }

    const size_t buffer_element_size = is_tf8_buffer ? sizeof(uint8_t) : sizeof(float);
    size_t buf_size = CalcSizeFromDims(buffer_shape.getDimensions(), buffer_shape.rank(), buffer_element_size);
    
    // set the buffer encoding type
    std::unique_ptr<zdl::DlSystem::UserBufferEncoding> user_buffer_encoding;
    if (buffer_element_size == sizeof(uint8_t)) {
        user_buffer_encoding =
            std::move(std::unique_ptr<zdl::DlSystem::UserBufferEncodingTf8>(
                new zdl::DlSystem::UserBufferEncodingTf8(0, 1.0)));
    } else {
        user_buffer_encoding =
            std::move(std::unique_ptr<zdl::DlSystem::UserBufferEncodingFloat>(
                new zdl::DlSystem::UserBufferEncodingFloat()));
    }

    // create user-backed storage to load input data onto it
    application_buffers.emplace(name, std::vector<uint8_t>(buf_size));

    // create SNPE user buffer from the user-backed buffer
    zdl::DlSystem::IUserBufferFactory& ub_factory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
    snpe_userbacked_buffers.push_back(ub_factory.createUserBuffer(
        application_buffers.at(name).data(), buf_size, strides,
        user_buffer_encoding.get()));
    if (snpe_userbacked_buffers.back() == nullptr) {
        LOGE("TNN SNPE: Error while creating SNPE user buffer.\n");
    }
    // add the user-backed buffer to the inputMap, which is later on fed to the
    // network for execution
    user_buffer_map.add(name, snpe_userbacked_buffers.back().get());

    // add blob
    BlobDesc desc;
    desc.data_format = DATA_FORMAT_NHWC;
    desc.name        = name;
    desc.device_type = DEVICE_DSP;
    for (int i = 0; i<buffer_shape.rank(); i++) {
        desc.dims.push_back(buffer_shape[i]);
    }
    BlobHandle handle;
    handle.base   = application_buffers.at(name).data();
    blobmap[name] = new Blob(desc, handle);
}

void CreateInputBufferMap(zdl::DlSystem::UserBufferMap& input_map,
                          BlobMap& input_blobmap,
                          std::unordered_map<std::string, std::vector<uint8_t>>& application_buffers,
                          std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&snpe_userbacked_buffers,
                          std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                          bool is_tf8_buffer) {
    // get input tensor names of the network that need to be populated
    const auto& input_names_opt = snpe->getInputTensorNames();
    if (!input_names_opt) {
        LOGE("TNN SNPE: Error obtaining input tensor names.\n");
        throw std::runtime_error("TNN SNPE: Error obtaining input tensor names");
    }
    const zdl::DlSystem::StringList& input_names = *input_names_opt;
    assert(input_names.size() > 0);

    // create SNPE user buffers for each application storage buffer
    for (const char* name : input_names) {
        CreateUserBuffer(input_map, input_blobmap, application_buffers,
                         snpe_userbacked_buffers, snpe, name, is_tf8_buffer);
    }
}

void CreateOutputBufferMap(zdl::DlSystem::UserBufferMap& output_map,
                           BlobMap& output_blobmap,
                           std::unordered_map<std::string, std::vector<uint8_t>>& application_buffers,
                           std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>&snpe_userbacked_buffers,
                           std::unique_ptr<zdl::SNPE::SNPE>& snpe,
                           bool is_tf8_buffer) {
    // get input tensor names of the network that need to be populated
    const auto& output_names_opt = snpe->getOutputTensorNames();
    if (!output_names_opt) {
        throw std::runtime_error("Error obtaining output tensor names");
    }
    const zdl::DlSystem::StringList& output_names = *output_names_opt;

    // create SNPE user buffers for each application storage buffer
    for (const char* name : output_names) {
        CreateUserBuffer(output_map, output_blobmap, application_buffers,
                         snpe_userbacked_buffers, snpe, name, is_tf8_buffer);
    }
}

// WARNING: SNPE UDO not fully TESTED.
void LoadUdoPackages(const std::string& package_dir) {
    std::vector<std::string> udo_package_names = {"Selu"};

    for (const auto & name : udo_package_names) {
        std::string full_regpkg_path = package_dir + "/libUdoTNN" + name + "ImplCpu.so";
        if (zdl::SNPE::SNPEFactory::addOpPackage(full_regpkg_path) == false) {
            LOGE("Fail to Add Op Package %s.\n", full_regpkg_path.c_str());
        }
    }
}

}  // namespace TNN_NS
