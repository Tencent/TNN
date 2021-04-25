/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
* in compliance with the License. You may obtain a copy of the License at
* 
* https://opensource.org/licenses/BSD-3-Clause0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "tnn/network/openvino/layer_builder/compute/gemmbench_dnnl.h"
#include "tnn/network/openvino/layer_builder/compute/gemm_unit.h"

int GemmScan(int m, int n, int k)
{
    engine cpu_engine(engine::kind::cpu, 0);
    stream cpu_stream(cpu_engine);

    std::vector<float> A = std::vector<float> (m*k,1);
    std::vector<float> B = std::vector<float> (k*n,1);
    std::vector<float> C = std::vector<float> (m*n,1);
    std::vector<float> bias = std::vector<float> (n,1.1);

    TNN::openvino::GemmUnit gemmunit;

    // get dnnl_sgemm time
    auto tag_1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        dnnl_sgemm('N', 'N', m, n, k, 1.0, A.data(), k, B.data(), n, 0.0, C.data(), n);
    }
    auto tag_2    = std::chrono::high_resolution_clock::now();
    auto dnnl_sgemm_t = std::chrono::duration<double>(tag_2 - tag_1).count();

    // get InnerProduct time
    tag_1    = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        gemmunit.InnerProduct(cpu_engine, cpu_stream, A.data(), B.data(), bias.data(), C.data(), m, n, k);
    }
    tag_2    = std::chrono::high_resolution_clock::now();
    auto InnerProduct_t = std::chrono::duration<double>(tag_2 - tag_1).count();

    // get Matmul time
    tag_1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        gemmunit.MatMul(cpu_engine, cpu_stream, A.data(), B.data(), bias.data(), C.data(), m, n, k);
    }
    tag_2    = std::chrono::high_resolution_clock::now();
    auto MatMul_t = std::chrono::duration<double>(tag_2 - tag_1).count();

    int result = 0;
    if (dnnl_sgemm_t > InnerProduct_t)
        result = 1;
    else if (InnerProduct_t > MatMul_t)
        result = 2;

    TNN::openvino::GemmUnit::clean();
    return result;
}
