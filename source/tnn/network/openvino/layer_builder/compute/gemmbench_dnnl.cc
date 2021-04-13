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

int padding(int cols) {
    int skip = (16 - cols % 16) % 16;
    int stride = cols + skip;
    if (stride % 256 == 0) {
        stride += 4;
    }
    return stride;
}

int GemmScan(int m, int n, int k)
{
    int result = 0;

    float *A = new float[m*k];
    float *B = new float[k*n];
    float *C = new float[m*n];

    double t_dnnl_sgemm = test_dnnl_sgemm(A, B, C, m, n, k);

    engine eng(engine::kind::cpu, 0);
    engine cpu_engine;
    stream cpu_stream;

    stream stream(eng);
    cpu_engine = eng;
    cpu_stream = stream;

    float *bias = new float[n];
    for (int i = 0; i < n; ++i) {
        bias[i] = 1.1;
    }

    double t_dnnl_ip_ffff  = test_dnnl_inner_product(cpu_engine, cpu_stream, A, B, bias, C, m, n, k);
    double t_dnnl_mm_ffff  = test_dnnl_matmul(cpu_engine, cpu_stream, A, B, bias, C, m, n, k);

    delete[] A;
    delete[] B;
    delete[] C;

    if (t_dnnl_sgemm > t_dnnl_ip_ffff)
        result = 1;
    else if (t_dnnl_ip_ffff > t_dnnl_mm_ffff)
        result = 2;

    return result;
}

double test_dnnl_sgemm(float *A, float *B, float *C, int m, int n, int k)
{
    dnnl_sgemm('N', 'N', m, n, k, 1.0, A, k, B, n, 0.0, C, n);

    auto tag_1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        dnnl_sgemm('N', 'N', m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    }

    auto tag_2 = std::chrono::high_resolution_clock::now();
    auto tag_diff = std::chrono::duration<double>(tag_2 - tag_1).count();
    std::cout << "result: " << C[0] << "," << C[m*n-1] << std::endl;

    return tag_diff;
}

template <typename T_A, typename T_B, typename T_bias, typename T_C>
double test_dnnl_inner_product(engine eng, stream stm, T_A* A_buf, T_B* B_buf, T_bias* bias_buf, T_C* C_buf, int m, int n, int k)
{
    InnerProduct(eng, stm, A_buf, B_buf, bias_buf, C_buf, m, n, k);

    auto tag_1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        InnerProduct(eng, stm, A_buf, B_buf, bias_buf, C_buf, m, n, k);
    }

    auto tag_2 = std::chrono::high_resolution_clock::now();
    auto tag_diff = std::chrono::duration<double>(tag_2 - tag_1).count();
    std::cout << "result: " << C_buf[0] << "," << C_buf[m*n-1] << std::endl;

    return tag_diff;
}

template <typename T_A, typename T_B, typename T_bias, typename T_C>
double test_dnnl_matmul(engine eng, stream stm, T_A* A_buf, T_B* B_buf, T_bias* bias_buf, T_C* C_buf, int m, int n, int k)
{
    MatMul(eng, stm, A_buf, B_buf, bias_buf, C_buf, m, n, k);

    auto tag_1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        MatMul(eng, stm, A_buf, B_buf, bias_buf, C_buf, m, n, k);
    }

    auto tag_2 = std::chrono::high_resolution_clock::now();
    auto tag_diff = std::chrono::duration<double>(tag_2 - tag_1).count();
    std::cout << "result: " << C_buf[0] << "," << C_buf[m*n-1] << std::endl;

    return tag_diff;
}
