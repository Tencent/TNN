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

#include "tnn/device/x86/acc/x86_unary2_layer_acc.h"
#include "tnn/device/x86/x86_context.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

std::map<std::string, unary2_kernel_avx_func_t> &X86Unary2LayerAcc::GetUnary2KernelMap() {
    static std::map<std::string, unary2_kernel_avx_func_t> kernel_map;
    return kernel_map;
}

std::string X86Unary2LayerAcc::GetUnaryKernelName(LayerType type, x86_isa_t arch) {
    return ToString(type) + "_" + ToString(arch);
}

Status X86Unary2LayerAcc::RegisterUnary2Kernel(LayerType type, x86_isa_t arch, unary2_kernel_avx_func_t kernel) {
    std::string kernel_name = GetUnaryKernelName(type, arch);
    auto &kernel_map        = GetUnary2KernelMap();
    kernel_map[kernel_name] = kernel;
    return TNN_OK;
}

Status X86Unary2LayerAcc::GetUnary2Kernel(LayerType type, x86_isa_t arch, unary2_kernel_avx_func_t &kernel) {
    const auto &kernel_map  = GetUnary2KernelMap();
    std::string kernel_name = GetUnaryKernelName(type, arch);
    if (kernel_map.find(kernel_name) == kernel_map.end() || kernel_map.at(kernel_name) == nullptr) {
        return Status(TNNERR_PARAM_ERR, "X86Unary2LayerAcc can not find unary kernel");
    }
    kernel = kernel_map.at(kernel_name);
    return TNN_OK;
}

Status X86_UNARY2_CALCULATE(DimsVector &dims, const float *src, float *dst, LayerType type, x86_isa_t arch,
                            LayerParam *param) {
    unary2_kernel_avx_func_t unary2_kernel_func = nullptr;
    RETURN_ON_NEQ(X86Unary2LayerAcc::GetUnary2Kernel(type, arch, unary2_kernel_func), TNN_OK);

    unary2_kernel_func(dims, src, dst, param);

    return TNN_OK;
}

X86Unary2LayerAcc::~X86Unary2LayerAcc() {}

Status X86Unary2LayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                               const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    RETURN_ON_NEQ(X86LayerAcc::Init(context, param, resource, inputs, outputs), TNN_OK);
    return TNN_OK;
}

Status X86Unary2LayerAcc::DoForward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    auto dims = output->GetBlobDesc().dims;

    int count        = DimsVectorUtils::Count(dims);
    auto input_data  = static_cast<float *>(input->GetHandle().base);
    auto output_data = static_cast<float *>(output->GetHandle().base);

    RETURN_ON_NEQ(X86_UNARY2_CALCULATE(dims, input_data, output_data, type_, arch_, param_), TNN_OK);

    return TNN_OK;
}

Status X86Unary2LayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

}  // namespace TNN_NS