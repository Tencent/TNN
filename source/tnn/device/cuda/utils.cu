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

#include <exception>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "tnn/device/cuda/utils.cuh"


namespace TNN_NS {

template<> __device__ float get_float_value<__half>(__half value) { return __half2float(value); }

template<> __device__ __half convert_float_value<__half>(float value) { return __float2half(value); }


CublasMMConfig::CublasMMConfig() {}

CublasMMConfig::CublasMMConfig(const int& in_dtype,
                               const int& in_batch,
                               cublasOperation_t in_transa,
                               cublasOperation_t in_transb,
                               const int& in_m,
                               const int& in_n,
                               const int& in_k,
                               const int& in_lda,
                               const int& in_ldb,
                               const int& in_ldc,
                               const float& f_alpha,
                               const float& f_beta,
                               const int& max_workspace_size) :
                                   dtype(in_dtype), batch(in_batch), m(in_m), n(in_n), k(in_k),
                                   lda(in_lda), ldb(in_ldb), ldc(in_ldc) {
    transa = in_transa == CUBLAS_OP_N ? 0 : 1;
    transb = in_transb == CUBLAS_OP_N ? 0 : 1;
    non_1_alpha = f_alpha == 1.0f ? 0 : 1;
    non_0_beta = f_beta == 0.0f ? 0 : 1;
    need_workspace = max_workspace_size > 0 ? 1 : 0;
}

CublasMMConfig::CublasMMConfig(const int& in_dtype,
                               const int& in_batch,
                               cublasOperation_t in_transa,
                               cublasOperation_t in_transb,
                               const int& in_m,
                               const int& in_n,
                               const int& in_k,
                               const int& in_lda,
                               const int& in_ldb,
                               const int& in_ldc,
                               const half& h_alpha,
                               const half& h_beta,
                               const int& max_workspace_size) :
                                   dtype(in_dtype), batch(in_batch), m(in_m), n(in_n), k(in_k),
                                   lda(in_lda), ldb(in_ldb), ldc(in_ldc) {
    transa = in_transa == CUBLAS_OP_N ? 0 : 1;
    transb = in_transb == CUBLAS_OP_N ? 0 : 1;
    non_1_alpha = float(h_alpha) == 1.0f ? 0 : 1;
    non_0_beta = float(h_beta) == 0.0f ? 0 : 1;
    need_workspace = max_workspace_size > 0 ? 1 : 0;
}

CublasMMConfig::CublasMMConfig(const int& in_dtype,
                               const int& in_batch,
                               cublasOperation_t in_transa,
                               cublasOperation_t in_transb,
                               const int& in_m,
                               const int& in_n,
                               const int& in_k,
                               const int& in_lda,
                               const int& in_ldb,
                               const int& in_ldc,
                               const bool in_non_1_alpha,
                               const bool in_non_0_beta,
                               const bool in_need_workspace) :
                                   dtype(in_dtype), batch(in_batch), m(in_m), n(in_n), k(in_k),
                                   lda(in_lda), ldb(in_ldb), ldc(in_ldc),
                                   non_1_alpha(in_non_1_alpha), non_0_beta(in_non_0_beta), need_workspace(in_need_workspace) {
    transa = in_transa == CUBLAS_OP_N ? 0 : 1;
    transb = in_transb == CUBLAS_OP_N ? 0 : 1;
}






CublasMMWrapper::CublasMMWrapper(cublasHandle_t   cublas_handle,
                                 cublasLtHandle_t cublaslt_handle,
                                 void* cuda_context_workspace) :
    cublas_handle_(cublas_handle),
    cublaslt_handle_(cublaslt_handle),
    cuda_context_workspace_(cublaslt_handle) {
}

CublasMMWrapper::~CublasMMWrapper() {
    FreeCublasLtDesc();
}

void CublasMMWrapper::Gemm(cublasOperation_t transa,
                           cublasOperation_t transb,
                           const int         m,
                           const int         n,
                           const int         k,
                           const void*       alpha,
                           const void*       A,
                           cudaDataType_t    Atype,
                           int               lda,
                           const void*       B,
                           cudaDataType_t    Btype,
                           int               ldb,
                           const void*       beta,
                           void*             C,
                           cudaDataType_t    Ctype,
                           int               ldc,
                           cudaDataType_t    computeType,
                           cublasGemmAlgo_t  algo) {
    CUBLAS_CHECK(cublasGemmEx(cublas_handle_,
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              alpha,
                              A,
                              Atype,
                              lda,
                              B,
                              Btype,
                              ldb,
                              beta,
                              C,
                              Ctype,
                              ldc,
                              computeType,
                              algo));
}

void CublasMMWrapper::Gemm(cublasOperation_t transa,
                           cublasOperation_t transb,
                           const int         m,
                           const int         n,
                           const int         k,
                           const float*      A,
                           const int         lda,
                           const float*      B,
                           const int         ldb,
                           float*            C,
                           const int         ldc,
                           cudaStream_t      stream,
                           const int         max_workspace_size,
                           const bool        use_default_algo,
                           const std::string caller_name) {
    Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, 1.0f, 0.0f, stream, max_workspace_size, use_default_algo, caller_name);
}

void CublasMMWrapper::Gemm(cublasOperation_t transa,
                           cublasOperation_t transb,
                           const int         m,
                           const int         n,
                           const int         k,
                           const half*       A,
                           const int         lda,
                           const half*       B,
                           const int         ldb,
                           half*             C,
                           const int         ldc,
                           cudaStream_t      stream,
                           const int         max_workspace_size,
                           const bool        use_default_algo,
                           const std::string caller_name) {
    Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, half(1.0f), half(0.0f), stream, max_workspace_size, use_default_algo, caller_name);
}

void CublasMMWrapper::Gemm(cublasOperation_t transa,
                           cublasOperation_t transb,
                           const int         m,
                           const int         n,
                           const int         k,
                           const float*      A,
                           const int         lda,
                           const float*      B,
                           const int         ldb,
                           float*            C,
                           const int         ldc,
                           float             f_alpha,
                           float             f_beta,
                           cudaStream_t      stream,
                           const int         max_workspace_size,
                           const bool        use_default_algo,
                           const std::string caller_name) {
    const void* alpha = reinterpret_cast<void*>(&f_alpha);
    const void* beta  = reinterpret_cast<void*>(&f_beta);
    CUBLAS_CHECK(cublasGemmEx(cublas_handle_,
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              alpha,
                              A,
                              CUDA_R_32F,
                              lda,
                              B,
                              CUDA_R_32F,
                              ldb,
                              beta,
                              C,
                              CUDA_R_32F,
                              ldc,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT));
}

void CublasMMWrapper::Gemm(cublasOperation_t transa,
                           cublasOperation_t transb,
                           const int         m,
                           const int         n,
                           const int         k,
                           const half*       A,
                           const int         lda,
                           const half*       B,
                           const int         ldb,
                           half*             C,
                           const int         ldc,
                           float             f_alpha,
                           float             f_beta,
                           cudaStream_t      stream,
                           const int         max_workspace_size,
                           const bool        use_default_algo,
                           const std::string caller_name) {
    Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, half(f_alpha), half(f_beta), stream, max_workspace_size, use_default_algo, caller_name);
}

void CublasMMWrapper::Gemm(cublasOperation_t transa,
                           cublasOperation_t transb,
                           const int         m,
                           const int         n,
                           const int         k,
                           const half*       A,
                           const int         lda,
                           const half*       B,
                           const int         ldb,
                           half*             C,
                           const int         ldc,
                           half              h_alpha,
                           half              h_beta,
                           cudaStream_t      stream,
                           const int         max_workspace_size,
                           const bool        use_default_algo,
                           const std::string caller_name) {
    if (use_default_algo) {
        const void* alpha = reinterpret_cast<void*>(&h_alpha);
        const void* beta  = reinterpret_cast<void*>(&h_beta);
        CublasMMConfig config = CublasMMConfig(int(DATA_TYPE_HALF), 1, transa, transb, m, n, k, lda, ldb, ldc, h_alpha, h_beta, max_workspace_size);
        if (this->algo_map_.find(config) == this->algo_map_.end()) {
            CublasMMAlgoInfo new_algo_info;
            InitCublasLtDesc(config, new_algo_info);
            this->algo_map_[config] = std::move(new_algo_info);
        }
        CublasMMAlgoInfo& algo_info = this->algo_map_[config];
        CUBLAS_CHECK(cublasLtMatmul(cublaslt_handle_,
                                    algo_info.operationDesc,
                                    alpha,
                                    A,
                                    algo_info.Adesc,
                                    B,
                                    algo_info.Bdesc,
                                    beta,
                                    C,
                                    algo_info.Cdesc,
                                    C,
                                    algo_info.Cdesc,
                                    NULL,
                                    NULL,
                                    0,
                                    stream));
        return;
    }

    // ONLY GEMM with caller_name set will be stored in Configure Map.
    if (!caller_name.empty()) {
        std::string map_key = caller_name + "_half";
        Status ret = UpdateMinMaxMapAndFindMMAlgosIfNeeded(map_key, DATA_TYPE_HALF, transa, transb, 1, m, n, k, A, lda, B, ldb, C, ldc,
                                                           float(h_alpha), float(h_beta), stream, max_workspace_size, true, true);
        if (ret != TNN_OK) {
            LOGE("Error in Updating Min Max Configure Map and run FindMMAlgos() for intermediate values, caller_name = (%s).\n", caller_name.c_str());
        }
    }

    CublasMMAlgoInfo best_algo_info;
    const void* alpha = reinterpret_cast<void*>(&h_alpha);
    const void* beta  = reinterpret_cast<void*>(&h_beta);
    cublasStatus_t status = FindBestMMAlgo(transa, transb, 1, m, n, k, A, lda, B, ldb, C, ldc,
                                           alpha, beta, best_algo_info, stream, max_workspace_size);

    if (best_algo_info.is_cublaslt) {
        CUBLAS_CHECK(cublasLtMatmul(cublaslt_handle_,
                                    best_algo_info.operationDesc,
                                    alpha,
                                    A,
                                    best_algo_info.Adesc,
                                    B,
                                    best_algo_info.Bdesc,
                                    beta,
                                    C,
                                    best_algo_info.Cdesc,
                                    C,
                                    best_algo_info.Cdesc,
                                    &(best_algo_info.algo),
                                    this->cuda_context_workspace_,
                                    best_algo_info.workspaceSize,
                                    stream));
    } else {
        CUBLAS_CHECK(cublasGemmEx(cublas_handle_,
                                  transa,
                                  transb,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  A,
                                  CUDA_R_16F,
                                  lda,
                                  B,
                                  CUDA_R_16F,
                                  ldb,
                                  beta,
                                  C,
                                  CUDA_R_16F,
                                  ldc,
                                  CUDA_R_16F,
                                  cublasGemmAlgo_t(best_algo_info.algoId)));
    }
}

Status CublasMMWrapper::InitCublasLtDesc(const CublasMMConfig& config,
                                         CublasMMAlgoInfo& info) {
    DataType dtype = DataType(config.dtype);
    cublasOperation_t transa = config.transa == 0 ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transb = config.transb == 0 ? CUBLAS_OP_N : CUBLAS_OP_T;
    return InitCublasLtDesc(dtype, transa, transb, config.m, config.n, config.k, config.lda, config.ldb, config.ldc, info);
}

Status CublasMMWrapper::InitCublasLtDesc(const DataType    dtype,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         const int         m,
                                         const int         n,
                                         const int         k,
                                         const int         lda,
                                         const int         ldb,
                                         const int         ldc,
                                         CublasMMAlgoInfo& info) {
    bool is_fp16_computeType = dtype == DATA_TYPE_HALF ? 1 : 0;
    LOGD("InitCublasLtDesc: dtype=(%d), m=%d, n=%d, k=%d, lda=%d, ldb=%d, ldc=%d.\n", int(dtype), m, n, k, lda, ldb, ldc);

#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif
    cudaDataType_t scaleType;

    if (is_fp16_computeType) {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_16F;
#else
        computeType = CUDA_R_16F;
#endif
        scaleType = CUDA_R_16F;
    }
    else {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_32F;
#else
        computeType = CUDA_R_32F;
#endif
        scaleType = CUDA_R_32F;
    }

    // Create descriptors for the original matrices
    // "scaleType below should actually be computeType", however, cuda>=11 requires compute type to be cublasComputType_t,
    // so we use scaleType to replace
    cublasLtMatrixLayoutCreate(&(info.Adesc), scaleType, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    cublasLtMatrixLayoutCreate(&(info.Bdesc), scaleType, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
    cublasLtMatrixLayoutCreate(&(info.Cdesc), scaleType, m, n, ldc);

#if (CUDART_VERSION >= 11000)
    cublasLtMatmulDescCreate(&(info.operationDesc), computeType, scaleType);
#else
    cublasLtMatmulDescCreate(&(info.operationDesc), computeType);
#endif
    cublasLtMatmulDescSetAttribute(info.operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(info.operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));
    
    return TNN_OK;
}

void CublasMMWrapper::FreeCublasLtDesc() {
    for (auto& iter : this->algo_map_) {
        if (iter.second.operationDesc) {
            cublasLtMatmulDescDestroy(iter.second.operationDesc);
            iter.second.operationDesc = NULL;
        }
        if (iter.second.Adesc) {
            cublasLtMatrixLayoutDestroy(iter.second.Adesc);
            iter.second.Adesc = NULL;
        }
        if (iter.second.Bdesc) {
            cublasLtMatrixLayoutDestroy(iter.second.Bdesc);
            iter.second.Bdesc = NULL;
        }
        if (iter.second.Cdesc) {
            cublasLtMatrixLayoutDestroy(iter.second.Cdesc);
            iter.second.Cdesc = NULL;
        }
    }
}

void CublasMMWrapper::FreeCublasLtDesc(const CublasMMConfig& config) {
    auto iter = this->algo_map_.find(config);
    if (iter != this->algo_map_.end()) {
        if (iter->second.operationDesc) {
            cublasLtMatmulDescDestroy(iter->second.operationDesc);
            iter->second.operationDesc = NULL;
        }
        if (iter->second.Adesc) {
            cublasLtMatrixLayoutDestroy(iter->second.Adesc);
            iter->second.Adesc = NULL;
        }
        if (iter->second.Bdesc) {
            cublasLtMatrixLayoutDestroy(iter->second.Bdesc);
            iter->second.Bdesc = NULL;
        }
        if (iter->second.Cdesc) {
            cublasLtMatrixLayoutDestroy(iter->second.Cdesc);
            iter->second.Cdesc = NULL;
        }
    }
}

Status CublasMMWrapper::UpdateMinMaxMapAndFindMMAlgosIfNeeded(const std::string& caller_name_key,
                                                              const DataType     dtype,
                                                              cublasOperation_t  transa,
                                                              cublasOperation_t  transb,
                                                              const int          batch,
                                                              const int          m,
                                                              const int          n,
                                                              const int          k,
                                                              const void*        A,
                                                              const int          lda,
                                                              const void*        B,
                                                              const int          ldb,
                                                              void*              C,
                                                              const int          ldc,
                                                              float              f_alpha,
                                                              float              f_beta,
                                                              cudaStream_t       stream,
                                                              const int          max_workspace_size,
                                                              const bool         enable_cublas_algo,
                                                              const bool         enable_cublaslt_algo) {
    auto iter = this->caller_name_minmax_map_.find(caller_name_key);
    if (iter != this->caller_name_minmax_map_.end()) {
        CublasMMMinMaxConfig& caller_cfg = iter->second;
        if (caller_cfg.transa != transa || caller_cfg.transb != transb ||
            std::abs(caller_cfg.f_alpha - f_alpha) > 1e-4 || std::abs(caller_cfg.f_beta - f_beta) > 1e-4) {
            LOGE("Calling Same GEMM with different TransA, TransB, alpha or beta.\n");
            return TNNERR_NET_ERR;
        }

        // Step 1: Update Min Max Config Map, check if the GEMM has one and only one of M, N, K dynamic.
        int diff_count = 0;
        bool diff_batch = false, diff_m = false, diff_n = false, diff_k = false;
        if (batch < caller_cfg.batch_min || m > caller_cfg.batch_max) {
            caller_cfg.batch_min = std::min(batch, caller_cfg.batch_min);
            caller_cfg.batch_max = std::max(batch, caller_cfg.batch_max);
            diff_count++;
            diff_batch = true;
        }
        if (m < caller_cfg.m_min || m > caller_cfg.m_max) {
            caller_cfg.m_min = std::min(m, caller_cfg.m_min);
            caller_cfg.m_max = std::max(m, caller_cfg.m_max);
            if (transa == CUBLAS_OP_N) {
                caller_cfg.lda_min   = std::min(lda, caller_cfg.lda_min);
                caller_cfg.lda_max   = std::max(lda, caller_cfg.lda_max);
                if (caller_cfg.lda_min != caller_cfg.m_min ||
                    caller_cfg.lda_max != caller_cfg.m_max) {
                    LOGD("Input Mat A transa == CUBLAS_OP_N but m != lda, intermediate m, lda in between [Min, Max] Best Algo will run on Init time.\n");
                    return TNN_OK;
                }
            }
            caller_cfg.ldc_min   = std::min(ldc, caller_cfg.ldc_min);
            caller_cfg.ldc_max   = std::max(ldc, caller_cfg.ldc_max);
            if (caller_cfg.ldc_min != caller_cfg.m_min ||
                caller_cfg.ldc_max != caller_cfg.m_max) {
                LOGD("Output Mat C, m != ldc, intermediate m, ldc in between [Min, Max] Best Algo will run on Init time.\n");
                return TNN_OK;
            }
            diff_count++;
            diff_m = true;
        }
        if (n < caller_cfg.n_min || n > caller_cfg.n_max) {
            caller_cfg.n_min = std::min(n, caller_cfg.n_min);
            caller_cfg.n_max = std::max(n, caller_cfg.n_max);
            if (transb == CUBLAS_OP_T) {
                caller_cfg.ldb_min = std::min(ldb, caller_cfg.ldb_min);
                caller_cfg.ldb_max = std::max(ldb, caller_cfg.ldb_max);
                if (caller_cfg.ldb_min != caller_cfg.n_min ||
                    caller_cfg.ldb_max != caller_cfg.n_max) {
                    LOGD("Input Mat B transa == CUBLAS_OP_T but n != ldb, intermediate n, ldb in between [Min, Max] Best Algo will run on Init time.\n");
                    return TNN_OK;
                }
            }
            diff_count++;
            diff_n = true;
        }
        if (k < caller_cfg.k_min || k > caller_cfg.k_max) {
            caller_cfg.k_min = std::min(k, caller_cfg.k_min);
            caller_cfg.k_max = std::max(k, caller_cfg.k_max);
            if (transa == CUBLAS_OP_T) {
                caller_cfg.lda_min = std::min(lda, caller_cfg.lda_min);
                caller_cfg.lda_max = std::max(lda, caller_cfg.lda_max);
                if (caller_cfg.lda_min != caller_cfg.k_min ||
                    caller_cfg.lda_max != caller_cfg.k_max) {
                    LOGD("Input Mat A transa == CUBLAS_OP_T but k != lda, intermediate k, lda in between [Min, Max] Best Algo will run on Init time.\n");
                    return TNN_OK;
                }
            }
            if (transb == CUBLAS_OP_N) {
                caller_cfg.ldb_min = std::min(ldb, caller_cfg.ldb_min);
                caller_cfg.ldb_max = std::max(ldb, caller_cfg.ldb_max);
                if (caller_cfg.ldb_min != caller_cfg.k_min ||
                    caller_cfg.ldb_max != caller_cfg.k_max) {
                    LOGD("Input Mat B transa == CUBLAS_OP_N but k != ldb, intermediate k, ldb in between [Min, Max] Best Algo will run on Init time.\n");
                    return TNN_OK;
                }
            }
            diff_count++;
            diff_k = true;
        }

        if (diff_count >= 2) {
            LOGD("More than one of batch, M, N, K is dynamic, run. This is not supported right NOW, maybe TNN future version will ADD support.\n");
            return TNN_OK;
        }


        // Step 2: Create Vector of MatMul config for FindCublasMMAlgo()
        std::vector<CublasMMConfig> intermediate_cfgs_vec;
        if (diff_batch) {
            // TODO, batch-GEMM now requires all of m, n, k, lda, ldb, ldc to be non-dynamic.
            //       Maybe we can loose the restriction a little bit later.
            if ((m != caller_cfg.m_min || caller_cfg.m_min != caller_cfg.m_max) ||
                (n != caller_cfg.n_min || caller_cfg.n_min != caller_cfg.n_max) ||
                (k != caller_cfg.k_min || caller_cfg.k_min != caller_cfg.k_max) ||
                (lda != caller_cfg.lda_min || caller_cfg.lda_min != caller_cfg.lda_max) ||
                (ldb != caller_cfg.ldb_min || caller_cfg.ldb_min != caller_cfg.ldb_max) ||
                (ldc != caller_cfg.ldc_min || caller_cfg.ldc_min != caller_cfg.ldc_max)) {
                LOGD("More than one of M, N, K, lda, ldb, ldc is dynamic in Batched-GEMM with dynamic batch. This is not supported right NOW, maybe TNN future version will ADD support.\n");
                return TNN_OK;
            }

            for (int batch_i = caller_cfg.batch_min; batch_i < caller_cfg.batch_max; batch_i++) {
                CublasMMConfig cur_config = CublasMMConfig(int(dtype), batch_i, transa, transb, m, n, k,
                                                           lda, ldb, ldc, f_alpha, f_beta, max_workspace_size);
                intermediate_cfgs_vec.push_back(std::move(cur_config));
            }
        }
        if (diff_m) {
            if ((transa == CUBLAS_OP_T && (lda != caller_cfg.lda_min || caller_cfg.lda_min != caller_cfg.lda_max)) ||
                (ldb != caller_cfg.ldb_min || caller_cfg.ldb_min != caller_cfg.ldb_max) ||
                (ldc != caller_cfg.ldc_min || caller_cfg.ldc_min != caller_cfg.ldc_max)) {
                LOGD("More than one of lda, ldb, ldc is dynamic, This is not supported right NOW, maybe TNN future version will ADD support.\n");
                return TNN_OK;
            }
            for (int m_i = caller_cfg.m_min; m_i < caller_cfg.m_max; m_i++) {
                if (this->run_every_intermediate_mnk_ || m_i % 16 == 0) {
                    int lda_i = transa == CUBLAS_OP_N ? m_i : lda;
                    CublasMMConfig cur_config = CublasMMConfig(int(dtype), batch, transa, transb, m_i, n, k,
                                                               lda_i, ldb, ldc, f_alpha, f_beta, max_workspace_size);
                    intermediate_cfgs_vec.push_back(std::move(cur_config));
                }
            }
        }
        if (diff_n) {
            if ((transb == CUBLAS_OP_N && (ldb != caller_cfg.ldb_min || caller_cfg.ldb_min != caller_cfg.ldb_max)) ||
                (lda != caller_cfg.lda_min || caller_cfg.lda_min != caller_cfg.lda_max) ||
                (ldc != caller_cfg.ldc_min || caller_cfg.ldc_min != caller_cfg.ldc_max)) {
                LOGD("More than one of lda, ldb, ldc is dynamic, This is not supported right NOW, maybe TNN future version will ADD support.\n");
                return TNN_OK;
            }
            for (int n_i = caller_cfg.n_min; n_i < caller_cfg.n_max; n_i++) {
                if (this->run_every_intermediate_mnk_ || n_i % 16 == 0) {
                    int ldb_i = transb == CUBLAS_OP_T ? n_i : ldb;
                    CublasMMConfig cur_config = CublasMMConfig(int(dtype), batch, transa, transb, m, n_i, k,
                                                               lda, ldb_i, ldc, f_alpha, f_beta, max_workspace_size);
                    intermediate_cfgs_vec.push_back(std::move(cur_config));
                }
            }
        }
        if (diff_k) {
            if ((transa == CUBLAS_OP_N && (lda != caller_cfg.lda_min || caller_cfg.lda_min != caller_cfg.lda_max)) ||
                (transb == CUBLAS_OP_T && (ldb != caller_cfg.ldb_min || caller_cfg.ldb_min != caller_cfg.ldb_max)) ||
                (ldc != caller_cfg.ldc_min || caller_cfg.ldc_min != caller_cfg.ldc_max)) {
                LOGD("More than one of lda, ldb, ldc is dynamic, This is not supported right NOW, maybe TNN future version will ADD support.\n");
                return TNN_OK;
            }
            for (int k_i = caller_cfg.k_min; k_i < caller_cfg.k_max; k_i++) {
                if (this->run_every_intermediate_mnk_ || k_i % 16 == 0) {
                    int lda_i = transa == CUBLAS_OP_T ? k_i : lda;
                    int ldb_i = transb == CUBLAS_OP_N ? k_i : ldb;
                    CublasMMConfig cur_config = CublasMMConfig(int(dtype), batch, transa, transb, m, n, k_i,
                                                               lda_i, ldb_i, ldc, f_alpha, f_beta, max_workspace_size);
                    intermediate_cfgs_vec.push_back(std::move(cur_config));
                }
            }
        }


        // Step 3: run FindCublasMMAlgo() for all Configs in vec.
        for (int i=0; i<intermediate_cfgs_vec.size(); i++) {
            CublasMMConfig& config = intermediate_cfgs_vec[i];
            CublasMMAlgoInfo cur_best_algo_info;
            if (dtype == DATA_TYPE_FLOAT) {
                if (this->algo_map_.find(config) != this->algo_map_.end()) {
                    const void* alpha = reinterpret_cast<void*>(&f_alpha);
                    const void* beta  = reinterpret_cast<void*>(&f_beta);
                    FindBestMMAlgo<float>(transa, transb, config.batch, config.m, config.n, config.k,
                                          static_cast<const float*>(A), config.lda, static_cast<const float*>(B), config.ldb,
                                          static_cast<float*>(C), config.ldc, alpha, beta, cur_best_algo_info, stream,
                                          max_workspace_size, enable_cublas_algo, enable_cublaslt_algo);
                    this->algo_map_[config] = std::move(cur_best_algo_info);
                }
            } else {
                if (this->algo_map_.find(config) != this->algo_map_.end()) {
                    half h_alpha = half(f_alpha);
                    half h_beta = half(f_beta);
                    const void* alpha = reinterpret_cast<void*>(&h_alpha);
                    const void* beta  = reinterpret_cast<void*>(&h_beta);
                    FindBestMMAlgo<half>(transa, transb, config.batch, config.m, config.n, config.k,
                                         static_cast<const half*>(A), config.lda, static_cast<const half*>(B), config.ldb,
                                         static_cast<half*>(C), config.ldc, alpha, beta, cur_best_algo_info, stream,
                                         max_workspace_size, enable_cublas_algo, enable_cublaslt_algo);
                    this->algo_map_[config] = std::move(cur_best_algo_info);
                }
            }
        }
    } else {
        CublasMMMinMaxConfig caller_cfg;
        caller_cfg.dtype     = int(DATA_TYPE_HALF);
        caller_cfg.transa    = transa;
        caller_cfg.transb    = transb;
        caller_cfg.batch_min = batch;
        caller_cfg.batch_max = batch;
        caller_cfg.m_min     = m;
        caller_cfg.m_max     = m;
        caller_cfg.n_min     = n;
        caller_cfg.n_max     = n;
        caller_cfg.k_min     = k;
        caller_cfg.k_max     = k;
        caller_cfg.lda_min   = lda;
        caller_cfg.lda_max   = lda;
        caller_cfg.ldb_min   = ldb;
        caller_cfg.ldb_max   = ldb;
        caller_cfg.ldc_min   = ldc;
        caller_cfg.ldc_max   = ldc;
        caller_cfg.f_alpha   = f_alpha;
        caller_cfg.f_beta    = f_beta;
        caller_cfg.max_workspace_size = max_workspace_size;

        this->caller_name_minmax_map_[caller_name_key] = std::move(caller_cfg);
    }

    return TNN_OK;
}

std::string CublasMMWrapper::GenAlgoConfigString(const CublasMMConfig& config,
                                                 const std::string  delimiter) {
    std::string uid;
    uid += std::to_string(config.dtype) + delimiter;
    uid += std::to_string(int(config.transa)) + delimiter;
    uid += std::to_string(int(config.transb)) + delimiter;
    uid += std::to_string(config.batch) + delimiter;
    uid += std::to_string(config.m) + delimiter;
    uid += std::to_string(config.n) + delimiter;
    uid += std::to_string(config.k) + delimiter;
    uid += std::to_string(config.lda) + delimiter;
    uid += std::to_string(config.ldb) + delimiter;
    uid += std::to_string(config.ldc) + delimiter;
    uid += std::to_string(int(config.non_1_alpha)) + delimiter;
    uid += std::to_string(int(config.non_0_beta)) + delimiter;
    uid += std::to_string(int(config.need_workspace));

    return uid;
}

std::string CublasMMWrapper::GenAlgoInfoString(const CublasMMAlgoInfo& info,
                                               const std::string delimiter) {
    std::string uid;

    std::stringstream exec_time_stream, waves_count_stream;
    exec_time_stream << std::fixed << std::setprecision(8) << info.exec_time;
    waves_count_stream << std::fixed << std::setprecision(8) << info.wavesCount;
    std::string exec_time_str = exec_time_stream.str();
    std::string waves_count_str = waves_count_stream.str();

    uid += std::to_string(info.dtype) + delimiter;
    uid += std::to_string(info.algoId) + delimiter;
    uid += exec_time_str + delimiter;
    uid += std::to_string(info.batch) + delimiter;
    uid += std::to_string(info.customOption) + delimiter;
    uid += std::to_string(info.tile) + delimiter;
    uid += std::to_string(info.splitK_val) + delimiter;
    uid += std::to_string(info.swizzle) + delimiter;
    uid += std::to_string(info.reductionScheme) + delimiter;
    uid += std::to_string(info.workspaceSize) + delimiter;
    uid += std::to_string(info.stages) + delimiter;
    uid += waves_count_str + delimiter;
    uid += std::to_string(int(info.is_cublaslt));
    
    return uid;
}

Status CublasMMWrapper::GetAlgoConfigFromString(const std::string& config_str,
                                                CublasMMConfig&    config,
                                                const std::string  delimiter) {
    Status ret = TNN_OK;
    std::istringstream cfg_iss(config_str);
    char delim = *(delimiter.c_str());

    auto getlineToInt = [&] (int& target) {
        Status func_ret = TNN_OK;
        std::string substr;
        int target_tmp;
        std::getline(cfg_iss, substr, *(delimiter.c_str()));
        try {
            target_tmp = std::stoi(substr);
        } catch(std::exception &err) {
            LOGE("Unable to Interpret TNN Cublas Algorithm Configure from String.\n");
            func_ret = TNNERR_NET_ERR;
        }
        target = target_tmp;
        return func_ret;
    };

    ret = getlineToInt(config.dtype);
    if (ret != TNN_OK) return ret;

    int transa_int;
    ret = getlineToInt(transa_int);
    if (ret != TNN_OK) return ret;
    config.transa = bool(transa_int);

    int transb_int;
    ret = getlineToInt(transb_int);
    if (ret != TNN_OK) return ret;
    config.transb = bool(transb_int);

    ret = getlineToInt(config.batch);
    if (ret != TNN_OK) return ret;

    ret = getlineToInt(config.m);
    if (ret != TNN_OK) return ret;
    ret = getlineToInt(config.n);
    if (ret != TNN_OK) return ret;
    ret = getlineToInt(config.k);
    if (ret != TNN_OK) return ret;

    ret = getlineToInt(config.lda);
    if (ret != TNN_OK) return ret;
    ret = getlineToInt(config.ldb);
    if (ret != TNN_OK) return ret;
    ret = getlineToInt(config.ldc);
    if (ret != TNN_OK) return ret;

    int non_1_alpha_int, non_0_beta_int, need_workspace_int;
    ret = getlineToInt(non_1_alpha_int);
    if (ret != TNN_OK) return ret;
    config.non_1_alpha = bool(non_1_alpha_int);
    ret = getlineToInt(non_0_beta_int);
    if (ret != TNN_OK) return ret;
    config.non_0_beta = bool(non_0_beta_int);
    ret = getlineToInt(need_workspace_int);
    if (ret != TNN_OK) return ret;
    config.need_workspace = bool(need_workspace_int);

    return ret;
}

Status CublasMMWrapper::GetAlgoInfoFromString(const std::string&    info_str,
                                              const CublasMMConfig& config,
                                              CublasMMAlgoInfo&     info,
                                              const std::string     delimiter) {
    Status ret = TNN_OK;
    std::istringstream info_iss(info_str);
    char delim = *(delimiter.c_str());

    auto getlineToInt = [&] (int& target) {
        Status func_ret = TNN_OK;
        std::string substr;
        int target_tmp;
        std::getline(info_iss, substr, *(delimiter.c_str()));
        try {
            target_tmp = std::stoi(substr);
        } catch(std::exception &err) {
            LOGE("Unable to Interpret TNN Cublas Algorithm Configure from String.\n");
            func_ret = TNNERR_NET_ERR;
        }
        target = target_tmp;
        return func_ret;
    };
    
    auto getlineToFloat = [&] (float& target) {
        Status func_ret = TNN_OK;
        std::string substr;
        float target_tmp;
        std::getline(info_iss, substr, *(delimiter.c_str()));
        try {
            target_tmp = std::stof(substr);
        } catch(std::exception &err) {
            LOGE("Unable to Interpret TNN Cublas Algorithm Configure from String.\n");
            func_ret = TNNERR_NET_ERR;
        }
        target = target_tmp;
        return func_ret;
    };

    ret = getlineToInt(info.dtype);
    if (ret != TNN_OK) return ret;
    ret = getlineToInt(info.algoId);
    if (ret != TNN_OK) return ret;
    ret = getlineToFloat(info.exec_time);
    if (ret != TNN_OK) return ret;

    ret = getlineToInt(info.batch);
    if (ret != TNN_OK) return ret;

    ret = getlineToInt(info.customOption);
    if (ret != TNN_OK) return ret;
    ret = getlineToInt(info.tile);
    if (ret != TNN_OK) return ret;
    ret = getlineToInt(info.splitK_val);
    if (ret != TNN_OK) return ret;
    ret = getlineToInt(info.swizzle);
    if (ret != TNN_OK) return ret;
    ret = getlineToInt(info.reductionScheme);
    if (ret != TNN_OK) return ret;
    ret = getlineToInt(info.workspaceSize);
    if (ret != TNN_OK) return ret;
    ret = getlineToInt(info.stages);
    if (ret != TNN_OK) return ret;

    ret = getlineToFloat(info.wavesCount);
    if (ret != TNN_OK) return ret;

    int is_cublaslt_int;
    ret = getlineToInt(is_cublaslt_int);
    if (ret != TNN_OK) return ret;
    info.is_cublaslt = bool(is_cublaslt_int);

    if (info.is_cublaslt) {
        InitCublasLtAlgo(info);
        InitCublasLtDesc(config, info);
    }

    return ret;
}

Status CublasMMWrapper::InitCublasLtAlgo(CublasMMAlgoInfo& info) {
    if (info.lt_algo_inited) {
        return TNN_OK;
    }

#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif
    cudaDataType_t scaleType;
    cudaDataType_t AType, BType, CType;

    if (info.dtype == DATA_TYPE_HALF) {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_16F;
#else
        computeType = CUDA_R_16F;
#endif
        scaleType   = CUDA_R_16F;
        AType       = CUDA_R_16F;
        BType       = CUDA_R_16F;
        CType       = CUDA_R_16F;
    } else {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_32F;
#else
        computeType = CUDA_R_32F;
#endif
        scaleType   = CUDA_R_32F;
        AType       = CUDA_R_32F;
        BType       = CUDA_R_32F;
        CType       = CUDA_R_32F;
    }

    cublasLtMatmulAlgo_t algo;
    if (!info.is_cublaslt) {
        LOGI("Unable to Generate cublasLt MatMul Algo from Algo Info because it is not a CublasLt algo. Return empty algo.\n");
        cublasLtMatmulAlgoInit(cublaslt_handle_, computeType, scaleType, AType, BType, CType, CType, 0, &algo);
        return TNN_OK;
    }
    cublasLtMatmulAlgoInit(cublaslt_handle_, computeType, scaleType, AType, BType, CType, CType, info.algoId, &algo);
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(info.customOption), sizeof(info.customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &(info.tile), sizeof(info.tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(info.splitK_val), sizeof(info.splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(info.swizzle), sizeof(info.swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(info.reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(info.stages), sizeof(info.stages));
#endif

    // Save Algo back to AlgoInfo Struct.
    info.algo = algo;
    info.lt_algo_inited = true;

    return TNN_OK;
}

std::string CublasMMWrapper::GetCacheFileName(const int device_id) {
#ifdef CUDART_VERSION
    std::string cuda_version = "cu" + std::to_string(CUDART_VERSION);
#else
    std::string cuda_version = "";
#endif

    // DeviceType, save as TNN/source/tnn/network/tensorrt/utils.cc  GetGpuType(int gpu_id)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    int length = strlen(prop.name);
    for (int i = 0; i < length; i++) {
        char c = prop.name[i];
        if (((c >= 'a') && (c<='z')) ||
            ((c >= 'A') && (c<='Z')) ||
            ((c >= '0') && (c<='9'))) {
               continue;
           }
       prop.name[i] = '_';
    }
    std::string gpu_type = std::string(prop.name);

    std::string cache_file_name = "." + cuda_version + "-" + gpu_type + "-cublas_and_cublaslt_matmul_stored_best_algos.cache";
    return cache_file_name;
}

Status CublasMMWrapper::SaveAlgoMapToFile(const std::string file_path, const int device_id) {
    Status ret = TNN_OK;
    std::string cache_file_name = GetCacheFileName(device_id);

    std::ofstream write_stream;
    write_stream.open(file_path + cache_file_name, std::ios::binary);
    if (!write_stream || !write_stream.is_open() || !write_stream.good()) {
        write_stream.close();
        LOGE("invalid mmalgo file path! (%s)\n", file_path.c_str());
        return Status(TNNERR_PACK_MODEL, "TNN cublas MM algorithm file cannot be written.");
    }

    const std::string config_info_separator = ";";

    std::string all_algos_str;
    int count = 0;
    for (const auto& item : this->algo_map_) {
        const CublasMMConfig& config = item.first;
        std::string config_str = GenAlgoConfigString(config, ",");
        std::string algo_info_str = GenAlgoInfoString(item.second, ",");
        all_algos_str += config_str + config_info_separator + algo_info_str + "\n";
        count++;
    }

    write_stream << all_algos_str;
    write_stream.close();
    LOGD("Successfully save [%d] Best Algos to best algo cache file.\n", count);
    
    return ret;
}

Status CublasMMWrapper::LoadAlgoMapFromFile(const std::string file_path, const int device_id) {
    Status ret = TNN_OK;
    std::string cache_file_name = GetCacheFileName(device_id);

    std::ifstream read_stream;
    read_stream.open(file_path + cache_file_name);
    if (!read_stream || !read_stream.is_open() || !read_stream.good()) {
        read_stream.close();
        LOGE("invalid mmalgo file path! (%s)\n", file_path.c_str());
        return Status(TNNERR_PACK_MODEL, "TNN cublas MM algorithm file cannot be read.");
    }

    int count = 0;
    while (read_stream) {
        std::string algo_line;
        std::getline(read_stream, algo_line);
        if (algo_line.empty()) {
            continue;
        }
        std::string algo_config_str, algo_info_str;
        std::stringstream algo_ss(algo_line);
        std::getline(algo_ss, algo_config_str, ';');
        std::getline(algo_ss, algo_info_str);
        if (algo_config_str.empty() || algo_info_str.empty()) {
            continue;
        }

        CublasMMConfig cur_config;
        CublasMMAlgoInfo cur_algo_info;
        ret = GetAlgoConfigFromString(algo_config_str, cur_config);
        if (ret != TNN_OK) {
            LOGE("Unable to Get Current Algorithm Config from TNN cublas Algorithm file. Skip The line.\n");
            continue;
        }
        ret = GetAlgoInfoFromString(algo_info_str, cur_config, cur_algo_info);
        if (ret != TNN_OK) {
            LOGE("Unable to Get Current Algorithm Info from TNN cublas Algorithm file. Skip The line.\n");
            continue;
        }

        this->algo_map_[cur_config] = std::move(cur_algo_info);
        count++;
    }
    read_stream.close();
    LOGD("Successfully load [%d] Best Algos from best algo cache file.\n", count);

    return ret;
}


cublasStatus_t CublasMMWrapper::RunCublasLtMMAlgoPerf(cublasLtMatmulDesc_t        operationDesc,
                                                      const void*                 alpha, // host or device pointer
                                                      const void*                 A,
                                                      cublasLtMatrixLayout_t      Adesc,
                                                      const void*                 B,
                                                      cublasLtMatrixLayout_t      Bdesc,
                                                      const void*                 beta, // host or device pointer
                                                      const void*                 C,
                                                      cublasLtMatrixLayout_t      Cdesc,
                                                      void*                       D,
                                                      cublasLtMatrixLayout_t      Ddesc,
                                                      const cublasLtMatmulAlgo_t& algo,
                                                      int                         kernelRepeats,
                                                      size_t                      workSpaceSizeInBytes,
                                                      CublasMMAlgoInfo&           perfResultInfo,
                                                      cudaStream_t                stream,
                                                      cudaEvent_t&                startEvent,
                                                      cudaEvent_t&                stopEvent) {
    const int WARMUP_ITERS = 10;

    // Check If Algo could run or not.
    cublasLtMatmulHeuristicResult_t heurResult;
    cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck(cublaslt_handle_, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, &algo, &heurResult);

    if (algoStatus == CUBLAS_STATUS_SUCCESS) {
        if (heurResult.workspaceSize <= workSpaceSizeInBytes) {
            cudaError_t err, err1, err2, err3;

            // Uncounted WARMUP iters.
            for (int loop = 0; loop < WARMUP_ITERS; loop++) {
                cublasStatus_t oneRunStatus = cublasLtMatmul(cublaslt_handle_,
                                                             operationDesc,
                                                             alpha,
                                                             A,
                                                             Adesc,
                                                             B,
                                                             Bdesc,
                                                             beta,
                                                             C,
                                                             Cdesc,
                                                             D,
                                                             Ddesc,
                                                             &algo,
                                                             cuda_context_workspace_,
                                                             workSpaceSizeInBytes,
                                                             stream);
                if (oneRunStatus != CUBLAS_STATUS_SUCCESS) {
                    algoStatus = oneRunStatus;
                    break;
                }
            }

            err = cudaEventRecord(startEvent, stream);
            for (int loop = 0; loop < kernelRepeats; loop++) {
                cublasStatus_t oneRunStatus = cublasLtMatmul(cublaslt_handle_,
                                                             operationDesc,
                                                             alpha,
                                                             A,
                                                             Adesc,
                                                             B,
                                                             Bdesc,
                                                             beta,
                                                             C,
                                                             Cdesc,
                                                             D,
                                                             Ddesc,
                                                             &algo,
                                                             cuda_context_workspace_,
                                                             workSpaceSizeInBytes,
                                                             stream);
                if (oneRunStatus != CUBLAS_STATUS_SUCCESS) {
                    algoStatus = oneRunStatus;
                    break;
                }
            }
            err1 = cudaEventRecord(stopEvent, stream);
            err2 = cudaEventSynchronize(stopEvent);
            float time;
            err3 = cudaEventElapsedTime(&time, startEvent, stopEvent);
            if ((err != cudaSuccess) || (err1 != cudaSuccess) || (err2 != cudaSuccess) || (err3 != cudaSuccess)) {
                algoStatus = CUBLAS_STATUS_INTERNAL_ERROR;
            }
            // For the moment only add successful findings
            if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme, stages;
                cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL);
                cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL);
                cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL);
                cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL);
                cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL);
                cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL);
#if (CUDART_VERSION >= 11000)
                cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), NULL);
#else
                stages = 0;
#endif
                LOGD("CublasLt MatMul Perf: AlgoId=[%d], exec_time=[%.4f]ms, tile=%d, customOption=%d, numSplitsK=%d, swizzle=%d, reductionScheme=%d, workspaceSize=%d, stages=%d, wavesCount=%.4f.\n",
                    algoId, time / kernelRepeats, tile, customOption, numSplitsK, swizzle, reductionScheme, heurResult.workspaceSize,
                    stages, heurResult.wavesCount);

                perfResultInfo.algoId            = algoId;
                perfResultInfo.exec_time         = time / kernelRepeats;
                perfResultInfo.batch             = 1;
                perfResultInfo.customOption      = customOption;
                perfResultInfo.tile              = tile;
                perfResultInfo.splitK_val        = numSplitsK;
                perfResultInfo.swizzle           = swizzle;
                perfResultInfo.reductionScheme   = reductionScheme;
                perfResultInfo.workspaceSize     = heurResult.workspaceSize;
                perfResultInfo.stages            = stages;
                perfResultInfo.wavesCount        = heurResult.wavesCount;
                perfResultInfo.algo              = algo;
                //perfResultInfo.operationDesc   = operationDesc; // NOTE: DESCs will not be set here.
                //perfResultInfo.Adesc           = Adesc;
                //perfResultInfo.Bdesc           = Bdesc;
                //perfResultInfo.Cdesc           = Cdesc;
                perfResultInfo.is_cublaslt       = true;
                perfResultInfo.lt_algo_inited    = true;
            }
        } else {
            LOGD("CublasLt MatMul Perf: No enough workspace. Required=%d, Max Size Available=%d.\n", heurResult.workspaceSize, workSpaceSizeInBytes);
            algoStatus = CUBLAS_STATUS_NOT_SUPPORTED;  // Not enough workspace
        }
    }

    return algoStatus;
}

































cublasMMWrapper::cublasMMWrapper(cublasHandle_t   cublas_handle,
                                 cublasLtHandle_t cublaslt_handle) :
    cublas_handle_(cublas_handle),
    cublaslt_handle_(cublaslt_handle) {
}

cublasMMWrapper::~cublasMMWrapper() {
    freeCublasLtDesc();
    cublaslt_inited_ = false;
}

cublasMMWrapper::cublasMMWrapper(const cublasMMWrapper& wrapper):
    cublas_handle_(wrapper.cublas_handle_),
    cublaslt_handle_(wrapper.cublaslt_handle_),
    Atype_(wrapper.Atype_),
    Btype_(wrapper.Btype_),
    Ctype_(wrapper.Ctype_),
    computeType_(wrapper.computeType_),
    cublaslt_inited_(false) {
}

void cublasMMWrapper::Gemm(cublasOperation_t transa,
                           cublasOperation_t transb,
                           const int         m,
                           const int         n,
                           const int         k,
                           const void*       alpha,
                           const void*       A,
                           cudaDataType_t    Atype,
                           int               lda,
                           const void*       B,
                           cudaDataType_t    Btype,
                           int               ldb,
                           const void*       beta,
                           void*             C,
                           cudaDataType_t    Ctype,
                           int               ldc,
                           cudaDataType_t    computeType,
                           cublasGemmAlgo_t  algo) {
    CUBLAS_CHECK(cublasGemmEx(cublas_handle_,
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              alpha,
                              A,
                              Atype,
                              lda,
                              B,
                              Btype,
                              ldb,
                              beta,
                              C,
                              Ctype,
                              ldc,
                              computeType,
                              algo));
}

void cublasMMWrapper::Gemm(cublasOperation_t transa,
                           cublasOperation_t transb,
                           const int         m,
                           const int         n,
                           const int         k,
                           const void*       A,
                           const int         lda,
                           const void*       B,
                           const int         ldb,
                           void*             C,
                           const int         ldc,
                           cudaStream_t      stream) {
    Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, 1.0f, 0.0f, stream);
}

void cublasMMWrapper::Gemm(cublasOperation_t transa,
                           cublasOperation_t transb,
                           const int         m,
                           const int         n,
                           const int         k,
                           const void*       A,
                           const int         lda,
                           const void*       B,
                           const int         ldb,
                           void*             C,
                           const int         ldc,
                           float             f_alpha,
                           float             f_beta,
                           cudaStream_t      stream) {
    half h_alpha = (half)(f_alpha);
    half h_beta  = (half)(f_beta);

    int  is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

    // fp32 use cublas as default
    // fp16 use cublasLt as default
    if (Atype_ == CUDA_R_16F) {
        prepareCublasLtDesc(transa, transb, m, n, k, lda, ldb, ldc);

        CUBLAS_CHECK(cublasLtMatmul(cublaslt_handle_,
                                    operationDesc,
                                    alpha,
                                    A,
                                    Adesc,
                                    B,
                                    Bdesc,
                                    beta,
                                    C,
                                    Cdesc,
                                    C,
                                    Cdesc,
                                    NULL,
                                    NULL,
                                    0,
                                    stream));
    } else {
        cublasGemmAlgo_t algoId = ((Atype_ == CUDA_R_16F) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT);

        CUBLAS_CHECK(cublasGemmEx(cublas_handle_,
                                  transa,
                                  transb,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  A,
                                  Atype_,
                                  lda,
                                  B,
                                  Btype_,
                                  ldb,
                                  beta,
                                  C,
                                  Ctype_,
                                  ldc,
                                  computeType_,
                                  algoId));
    }
}

void cublasMMWrapper::setFP32GemmConfig() {
    Atype_       = CUDA_R_32F;
    Btype_       = CUDA_R_32F;
    Ctype_       = CUDA_R_32F;
    computeType_ = CUDA_R_32F;
}

void cublasMMWrapper::setFP16GemmConfig() {
    Atype_       = CUDA_R_16F;
    Btype_       = CUDA_R_16F;
    Ctype_       = CUDA_R_16F;
    computeType_ = CUDA_R_32F;
}

void cublasMMWrapper::prepareCublasLtDesc(cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          const int         m,
                                          const int         n,
                                          const int         k,
                                          const int         lda,
                                          const int         ldb,
                                          const int         ldc) {
    if (cublaslt_inited_ && transa == cached_transa && transb == cached_transb &&
        m == cached_m && n == cached_n && k == cached_k &&
        lda == cached_lda && ldb == cached_ldb && ldc == cached_ldc) {
        return;
    }

    freeCublasLtDesc();

    int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType;
    cudaDataType_t scaleType;
#else
    cudaDataType_t computeType;
#endif

    if (is_fp16_computeType) {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_16F;
        scaleType = CUDA_R_16F;
#else
        computeType = CUDA_R_16F;
#endif
    }
    else {
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_32F;
        scaleType = CUDA_R_32F;
#else
        computeType = CUDA_R_32F;
#endif
    }

    // --------------------------------------
    // Create descriptors for the original matrices
    cublasLtMatrixLayoutCreate(&Adesc, Atype_, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, Btype_, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, Ctype_, m, n, ldc);

#if (CUDART_VERSION >= 11000)
    cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
#else
    cublasLtMatmulDescCreate(&operationDesc, computeType);
#endif
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t));

    cached_transa = transa;
    cached_transb = transb;
    cached_m = m;
    cached_n = n;
    cached_k = k;
    cached_lda = lda;
    cached_ldb = ldb;
    cached_ldc = ldc;
    cublaslt_inited_ = true;
}

void cublasMMWrapper::freeCublasLtDesc() {
    if (operationDesc) {
        cublasLtMatmulDescDestroy(operationDesc);
        operationDesc = NULL;
    }
    if (Adesc) {
        cublasLtMatrixLayoutDestroy(Adesc);
        Adesc = NULL;
    }
    if (Bdesc) {
        cublasLtMatrixLayoutDestroy(Bdesc);
        Bdesc = NULL;
    }
    if (Cdesc) {
        cublasLtMatrixLayoutDestroy(Cdesc);
        Cdesc = NULL;
    }
}

void cublasMMWrapper::batchedGemm(cublasOperation_t  transa,
                                  cublasOperation_t  transb,
                                  const int          batch_count,
                                  const int          m,
                                  const int          n,
                                  const int          k,
                                  const void* const* A,
                                  const int          lda,
                                  const void* const* B,
                                  const int          ldb,
                                  void* const*       C,
                                  const int          ldc,
                                  cudaStream_t       stream) {
    float f_alpha = static_cast<float>(1.0f);
    float f_beta  = static_cast<float>(0.0f);

    half h_alpha = (half)1.0f;
    half h_beta  = (half)0.0f;

    int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);
    cublasGemmAlgo_t algoId = ((Atype_ == CUDA_R_16F) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT);

    CUBLAS_CHECK(cublasGemmBatchedEx(cublas_handle_,
                                     transa,
                                     transb,
                                     m,
                                     n,
                                     k,
                                     alpha,
                                     A,
                                     Atype_,
                                     lda,
                                     B,
                                     Btype_,
                                     ldb,
                                     beta,
                                     C,
                                     Ctype_,
                                     ldc,
                                     batch_count,
                                     computeType_,
                                     algoId));
}

void cublasMMWrapper::stridedBatchedGemm(cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         const int         batch_count,
                                         const int         m,
                                         const int         n,
                                         const int         k,
                                         const void*       A,
                                         const int         lda,
                                         const int64_t     strideA,
                                         const void*       B,
                                         const int         ldb,
                                         const int64_t     strideB,
                                         void*             C,
                                         const int         ldc,
                                         const int64_t     strideC,
                                         cudaStream_t      stream,
                                         const float       f_alpha,
                                         const float       f_beta) {
    half h_alpha = (half)f_alpha;
    half h_beta  = (half)f_beta;

    int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<const void*>(&f_alpha);
    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<const void*>(&f_beta);
    cublasGemmAlgo_t algoId = ((Atype_ == CUDA_R_16F) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT);

    CUBLAS_CHECK(cublasGemmStridedBatchedEx(cublas_handle_,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            A,
                                            Atype_,
                                            lda,
                                            strideA,
                                            B,
                                            Btype_,
                                            ldb,
                                            strideB,
                                            beta,
                                            C,
                                            Ctype_,
                                            ldc,
                                            strideC,
                                            batch_count,
                                            computeType_,
                                            algoId));
}

}  //  namespace TNN_NS;
