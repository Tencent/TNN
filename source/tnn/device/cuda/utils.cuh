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

#ifndef TNN_SOURCE_TNN_DEVICE_CUDA_UTILS_CUH_
#define TNN_SOURCE_TNN_DEVICE_CUDA_UTILS_CUH_

#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <string>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include "tnn/core/common.h"
#include "tnn/core/macro.h"
#include "tnn/core/status.h"
#include "tnn/device/cuda/cuda_macro.h"



namespace TNN_NS {

template<typename T>
__device__ float get_float_value(T value) { return value; }

template<typename T>
__device__ T convert_float_value(float value) { return T(value); }

// BFP16 Compatible Alias
// Add: 2 Elementes
template<typename T>
inline __device__ T add(T a, T b) {
    return a + b;
}
//#ifdef ENABLE_BF16
//template<>
//inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b) {
//    return bf16hadd2(a, b);
//}
//template<>
//inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
//    return bf16hadd(a, b);
//}
//inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, float b) {
//    return bf16hadd(a, __float2bfloat16(b));
//}
//#endif // ENABLE_BF16

// hadd2, hsub2, hmul2, hexp2: 2 Elements
template<typename T>
inline __device__ T hadd2(T a, T b) {
    return __hadd2(a, b);
}
template<typename T>
inline __device__ T hsub2(T a, T b) {
    return __hsub2(a, b);
}
template<typename T>
inline __device__ T hmul2(T a, T b) {
    return __hmul2(a, b);
}
template<typename T>
inline __device__ T hexp2(T a) {
    return h2exp(a);
}
//#ifdef ENABLE_BF16
//template<>
//inline __device__ __nv_bfloat162 hadd2(__nv_bfloat162 a, __nv_bfloat162 b) {
//    return bf16hadd2(a, b);
//}
//template<>
//inline __device__ __nv_bfloat162 hsub2(__nv_bfloat162 a, __nv_bfloat162 b) {
//    return bf16hsub2(a, b);
//}
//template<>
//inline __device__ __nv_bfloat162 hmul2(__nv_bfloat162 a, __nv_bfloat162 b) {
//    return bf16hmul2(a, b);
//}
//template<>
//inline __device__ __nv_bfloat162 hexp2(__nv_bfloat162 a) {
//    return bf16exp2(a);
//}
//#endif // ENABLE_BF16



// Cuda float -> half/bfp16 data converter
// Cuda float -> half2/bfp162 data converter
template<typename T>
inline __device__ T float2type(float a) {
    return a;
}
template<>
inline __device__ half float2type(float a) {
    return __float2half_rn(a);
}
template<typename T>
inline __device__ T float2type2(float a);
template<>
inline __device__ half2 float2type2(float a) {
    return __float2half2_rn(a);
}
//#ifdef ENABLE_BF16
//template<>
//inline __device__ __nv_bfloat16 float2type(float a) {
//    return __float2bfloat16_rn(a);
//}
//template<>
//inline __device__ __nv_bfloat162 float2type2(float a) {
//    return __float2bfloat162_rn(a);
//}
//#endif // ENABLE_BF16


// Cuda half  -> half2  data converter
// Cuda bfp16 -> bfp162 data converter
template<typename T_IN, typename T_OUT>
inline __device__ T_OUT type2type2(T_IN a);
template<>
inline __device__ half2 type2type2(half a) {
    return __half2half2(a);
}
//#ifdef ENABLE_BF16
//template<>
//inline __device__ __nv_bfloat162 type2type2(__nv_bfloat16 a) {
//    return bf162bf162(a);
//}
//#endif // ENABLE_BF16





// Cuda Type <-> Type2 Type inter converter
// float <-> float
// half  <-> half2
// bfp16 <-> bpf162, TODO
template<typename T>
struct CudaType2InterConverter {using Type = half2;}; // by default half.
template<>
struct CudaType2InterConverter<half2> {using Type = half;};
template<>
struct CudaType2InterConverter<half> {using Type = half2;};
template<>
struct CudaType2InterConverter<float> {using Type = float;};
//#ifdef ENABLE_BF16
//template<>
//struct CudaType2InterConverter<__nv_bfloat162> {using Type = __nv_bfloat16;};
//template<>
//struct CudaType2InterConverter<__nv_bfloat16> {using Type = __nv_bfloat162;};
//#endif // ENABLE_BF16


class CublasMMConfig {
public:
    // Members
    int dtype;
    int batch;
    int m, n, k;
    int lda, ldb, ldc;
    bool transa, transb;
    bool non_1_alpha, non_0_beta;
    bool need_workspace;

public:
    // Constructors
    CublasMMConfig();
    
    CublasMMConfig(const int& in_dtype,
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
                   const int& max_workspace_size);

    CublasMMConfig(const int& in_dtype,
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
                   const int& max_workspace_size);

    CublasMMConfig(const int& in_dtype,
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
                   const bool in_need_workspace);


    // Equal for std::hash
    bool operator==(const CublasMMConfig &other) const {
        return (dtype == other.dtype && batch == other.batch &&
                m == other.m && n == other.n && k == other.k &&
                lda == other.lda && ldb == other.ldb && ldc == other.ldc &&
                transa == other.transa && transb == other.transb &&
                non_1_alpha == other.non_1_alpha && non_0_beta == other.non_0_beta &&
                need_workspace == other.need_workspace);
    }
};

}  // namespace TNN_NS

// Hash Function for Struct CublasMMConfig
namespace std {
    inline void cublas_mm_hash_combine(std::size_t& seed) { }
    
    template <typename T, typename... Rest>
    inline void cublas_mm_hash_combine(std::size_t& seed, const T& v, Rest... rest) {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        cublas_mm_hash_combine(seed, rest...);
    }

    template <>
    struct hash<::TNN_NS::CublasMMConfig> {
        std::size_t operator() (::TNN_NS::CublasMMConfig const& config) const noexcept {
            std::size_t h_transa         = std::hash<bool>{}(config.transa);
            std::size_t h_transb         = std::hash<bool>{}(config.transb);
            std::size_t h_non_1_alpha    = std::hash<bool>{}(config.non_1_alpha);
            std::size_t h_non_0_beta     = std::hash<bool>{}(config.non_0_beta);
            std::size_t h_need_workspace = std::hash<bool>{}(config.need_workspace);
            std::size_t h_booleans       = h_transa + (h_transb << 1) + (h_non_1_alpha << 2) + (h_non_0_beta << 3) + (h_need_workspace << 4);

            std::size_t h_dtype          = std::hash<int>{}(config.dtype);
            std::size_t h_batch          = std::hash<int>{}(config.batch);
            std::size_t h_m              = std::hash<int>{}(config.m);
            std::size_t h_n              = std::hash<int>{}(config.n);
            std::size_t h_k              = std::hash<int>{}(config.k);
            std::size_t h_lda            = std::hash<int>{}(config.lda);
            std::size_t h_ldb            = std::hash<int>{}(config.ldb);
            std::size_t h_ldc            = std::hash<int>{}(config.ldc);

            std::size_t h_all = 0;
            cublas_mm_hash_combine(h_all, h_booleans, h_dtype, h_batch, h_m, h_n, h_k, h_lda, h_ldb, h_ldc);
            return h_all;
        }
    };
}  // namespace std for hash func.



namespace TNN_NS {

class CublasMMWrapper {
public:
    explicit CublasMMWrapper(cublasHandle_t   cublas_handle,
                             cublasLtHandle_t cublaslt_handle,
                             void* cuda_context_workspace);

    ~CublasMMWrapper();

    void Gemm(cublasOperation_t transa,
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
              cublasGemmAlgo_t  algo);

    // @brief fp16 & fp32 simplified GEMM kernel with alpha=1, beta=0
    // @param caller_name: String For GEMM to know the caller. Allow TNN to find best algo for Shapes in between [min, max], better to be called in OP.Init() than in OP.forward().
    void Gemm(cublasOperation_t transa,
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
              const int         max_workspace_size = 0,
              const bool        use_default_algo = true,
              const std::string caller_name = "");
    void Gemm(cublasOperation_t transa,
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
              const int         max_workspace_size = 0,
              const bool        use_default_algo = true,
              const std::string caller_name = "");

    // @brief fp16 & fp32 simplified GEMM kernel with explicit alpha and beta
    // @param caller_name: String For GEMM to know the caller. Allow TNN to find best algo for Shapes in between [min, max], better to be called in OP.Init() than in OP.forward().
    void Gemm(cublasOperation_t transa,
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
              const int         max_workspace_size = 0,
              const bool        use_default_algo = true,
              const std::string caller_name = "");
    void Gemm(cublasOperation_t transa,
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
              const int         max_workspace_size = 0,
              const bool        use_default_algo = true,
              const std::string caller_name = "");
    void Gemm(cublasOperation_t transa,
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
              const int         max_workspace_size = 0,
              const bool        use_default_algo = true,
              const std::string caller_name = "");

    // @brief Free Stored CublasLtDesc of given config if exits.
    //        Free All Stored CublasLtDescs if config_str is not provided (empty)
    void FreeCublasLtDesc();
    void FreeCublasLtDesc(const CublasMMConfig& config);

private:
    // Address of Cuda Context Workspace.
    // This workspace is managed by CUDA Context, CublasMMWrapper is only the USER of the workspace.
    // workspace is set at the time when CublasMMWrapper Class is created.
    void* cuda_context_workspace_     = nullptr;
    cublasHandle_t   cublas_handle_   = nullptr;
    cublasLtHandle_t cublaslt_handle_ = nullptr;

    // Stored Algorithms:
private:
    // @brief Map To Store Algorithm Configurations, key: caller_name.
    //        Caller name can be Layer Name, or Unique Strings containing layer name.
    //        We Keep this Map to get Min / Max size of each of M, N, K, lda, ldb, ldc of Certain GEMM
    struct CublasMMMinMaxConfig {
        int               dtype = -1;
        cublasOperation_t transa;
        cublasOperation_t transb;
        int               batch_min = -1, batch_max = -1;
        int               m_min = -1, m_max = -1;
        int               n_min = -1, n_max = -1;
        int               k_min = -1, k_max = -1;
        int               lda_min = -1, lda_max = -1;
        int               ldb_min = -1, ldb_max = -1;
        int               ldc_min = -1, ldc_max = -1;
        float             f_alpha = 1.0f, f_beta = 0.0f;
        int               max_workspace_size = 0;
    };
    std::unordered_map<std::string, CublasMMMinMaxConfig> caller_name_minmax_map_;
    bool run_every_intermediate_mnk_ = true;

    // @brief Update CublasMMMinMaxConfig Map based on caller_name,
    //        if min shape != max shape (at most one of M, N, K not equal and lda, ldb, ldc == M, N or K)
    //        run FindBestMMAlgo() for every intermediate configure in between min shape and max shape
    //        or run FindBestMMAlgo() only for critical intermediate configures in between min shape and max shape.
    Status UpdateMinMaxMapAndFindMMAlgosIfNeeded(const std::string& caller_name_key,
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
                                                 const bool         enable_cublas_algo = true,
                                                 const bool         enable_cublaslt_algo = true);

    // @brief Map of Best Algorithm Infomation, key: MatMul Configure
    struct CublasMMAlgoInfo {
        int            dtype = -1;                                              // cublas, cublasLt
        int            algoId = -1;                                             // cublas, cublasLt
        float          exec_time = 99999.0f;                                    // cublas, cublasLt
        int            batch = -1;                                              // cublas, cublasLt
        int            customOption = -1, tile = -1, splitK_val = -1;           //         cublasLt
        int            swizzle = -1, reductionScheme = -1, workspaceSize = -1;  //         cublasLt
        int            stages = -1;                                             //         cublasLt >= 11.0
        float          wavesCount = 99999.0f;                                   //         cublasLt
        cublasStatus_t status;                                                  //         cublasLt
        cublasLtMatmulAlgo_t algo;                                              //         cublasLt
        cublasLtMatmulDesc_t   operationDesc = NULL;                            //         cublasLt
        cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;        //         cublasLt
        bool           is_cublaslt;                                             // cublas: false, cublasLt: true
        bool           lt_algo_inited = false;                                  //         cublasLt
    };
    std::unordered_map<CublasMMConfig, CublasMMAlgoInfo> algo_map_;


    // @brief Generate String for algo_map to be save to file, delimiter = ','
    std::string GenAlgoConfigString(const CublasMMConfig& config,
                                    const std::string  delimiter = ",");
    std::string GenAlgoInfoString(const CublasMMAlgoInfo& info,
                                  const std::string delimiter = ",");

    // @brief Generate Config / AlgoInfo from saved algo file string, delimiter = ','
    Status GetAlgoConfigFromString(const std::string& config_str,
                                   CublasMMConfig& config,
                                   const std::string delimiter = ",");
    Status GetAlgoInfoFromString(const std::string& info_str,
                                 const CublasMMConfig& config,
                                 CublasMMAlgoInfo& info,
                                 const std::string delimiter = ",");

    // @brief Init OpDesc, ADesc, BDesc and CDesc for CublasLt in CublasLtAlgoInfo
    Status InitCublasLtDesc(const CublasMMConfig& config, CublasMMAlgoInfo& info);
    Status InitCublasLtDesc(const DataType    dtype,
                            cublasOperation_t transa,
                            cublasOperation_t transb,
                            const int         m,
                            const int         n,
                            const int         k,
                            const int         lda,
                            const int         ldb,
                            const int         ldc,
                            CublasMMAlgoInfo& info);

    // @brief Init Algorithm for CublasLt in CublasLtAlgoInfo
    Status InitCublasLtAlgo(CublasMMAlgoInfo& info);

    // @brief Save / Load Best Algorithm Records to this->algo_map_, to / from a cache file.
    std::string GetCacheFileName(const int device_id);
    Status SaveAlgoMapToFile(const std::string file_path = "./", const int device_id = 0);
    Status LoadAlgoMapFromFile(const std::string file_path = "./", const int device_id = 0);

    // @brief Profile One CublasLt Algorithm, record its time cost
    cublasStatus_t RunCublasLtMMAlgoPerf(cublasLtMatmulDesc_t        operationDesc,
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
                                         cudaEvent_t&                stopEvent);

    // @brief Find Best CublasLt MM Algo
    // @param DType only Support float / half now.
    template <typename dtype>
    cublasStatus_t FindBestCublasLtMMAlgo(cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          const int         batch,
                                          const int         m,
                                          const int         n,
                                          const int         k,
                                          const dtype*      A,
                                          const int         lda,
                                          const dtype*      B,
                                          const int         ldb,
                                          dtype*            C,
                                          const int         ldc,
                                          const void*       alpha,  // host pointer
                                          const void*       beta,   // host pointer
                                          CublasMMAlgoInfo& best_cublaslt_algo,
                                          cudaStream_t      stream,
                                          const int         max_workspace_size = 0);

    // @brief Find Best Classical Cublas MM Algo
    // @param DType only Support float / half now.
    template <typename dtype>
    cublasStatus_t FindBestCublasMMAlgo(cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        const int         batch,
                                        const int         m,
                                        const int         n,
                                        const int         k,
                                        const dtype*      A,
                                        const int         lda,
                                        const dtype*      B,
                                        const int         ldb,
                                        dtype*            C,
                                        const int         ldc,
                                        const void*       alpha,  // host pointer
                                        const void*       beta,   // host pointer
                                        CublasMMAlgoInfo& best_cublas_algo,
                                        cudaStream_t      stream);

    // @brief Find Best Classical Cublas and CublasLt MM Algo
    // @param DType only Support float / half now.
    template <typename dtype>
    cublasStatus_t FindBestMMAlgo(cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  const int         batch,
                                  const int         m,
                                  const int         n,
                                  const int         k,
                                  const dtype*      A,
                                  const int         lda,
                                  const dtype*      B,
                                  const int         ldb,
                                  dtype*            C,
                                  const int         ldc,
                                  const void*       alpha,  // host pointer
                                  const void*       beta,   // host pointer
                                  CublasMMAlgoInfo& best_algo_info,
                                  cudaStream_t      stream,
                                  const int         max_workspace_size = 0,
                                  const bool        enable_cublas_algo = true,
                                  const bool        enable_cublaslt_algo = true);

};  // class CublasMMWrapper



// Template Func Definitions of Class CublasMMWrapper
template <typename dtype>
cublasStatus_t CublasMMWrapper::FindBestCublasLtMMAlgo(cublasOperation_t transa,
                                                       cublasOperation_t transb,
                                                       const int         batch,
                                                       const int         m,
                                                       const int         n,
                                                       const int         k,
                                                       const dtype*      A,
                                                       const int         lda,
                                                       const dtype*      B,
                                                       const int         ldb,
                                                       dtype*            C,
                                                       const int         ldc,
                                                       const void*       alpha,  // host pointer
                                                       const void*       beta,   // host pointer
                                                       CublasMMAlgoInfo& best_cublaslt_algo,
                                                       cudaStream_t      stream,
                                                       const int         max_workspace_size) {
    const int MAX_NUM_ALGO_IDS = 100;
    const int MAX_NUM_ALGO_COMBINATIONS = 1000;
    const int NUM_KERNEL_REPEATS = 100;
    const int NUM_MAX_TRAVERSAL = 50;
    const int splitKSequenceArray[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};
    CublasMMAlgoInfo perfResults[NUM_MAX_TRAVERSAL];

    int                  AlgoCount            = 0;
    int                  AlgoCountNoWorkspace = 0;
    cublasLtMatmulAlgo_t algos[MAX_NUM_ALGO_COMBINATIONS];
    cublasLtMatmulAlgo_t algosNoWorkspace[MAX_NUM_ALGO_COMBINATIONS];

    int algoIdArray[MAX_NUM_ALGO_IDS];
    int numAlgosAvailable = 0;

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    cudaEvent_t    startEvent;
    cudaEvent_t    stopEvent;

    cublasLtMatmulDesc_t   operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cudaDataType_t Atype, Btype, Ctype, scaleType;
#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif

    if (std::is_same<dtype, float>::value) {
        Atype = CUDA_R_32F, Btype = CUDA_R_32F, Ctype = CUDA_R_32F, scaleType = CUDA_R_32F;
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_32F;
#else
        computeType = CUDA_R_32F;
#endif
        for (int i=0; i<NUM_MAX_TRAVERSAL; i++) {
            perfResults[i].dtype = int(DATA_TYPE_FLOAT);
        }
    } else if (std::is_same<dtype, half>::value) {
        Atype = CUDA_R_16F, Btype = CUDA_R_16F, Ctype = CUDA_R_16F, scaleType = CUDA_R_16F;
#if (CUDART_VERSION >= 11000)
        computeType = CUBLAS_COMPUTE_16F;
#else
        computeType = CUDA_R_16F;
#endif
        for (int i=0; i<NUM_MAX_TRAVERSAL; i++) {
            perfResults[i].dtype = int(DATA_TYPE_HALF);
        }
    } else {
        return status;
    }

    // Create operation descriptor; see cublasLtMatmulDescAttributes_t for
    // details about defaults; here we just need to set the transforms for A and B
    // creates a matrix multiply descriptor
#if (CUDART_VERSION >= 11000)
    status = cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
#else
    status = cublasLtMatmulDescCreate(&operationDesc, computeType);
#endif
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }

    // Create matrix descriptors. We are good with the details here so no need
    // to set any extra attributes
    status = cublasLtMatrixLayoutCreate(&Adesc, Atype, m, k, m);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    status = cublasLtMatrixLayoutCreate(&Bdesc, Btype, k, n, k);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }
    status = cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, m);
    if (status != CUBLAS_STATUS_SUCCESS) {
        goto CLEANUP;
    }

    // Create CUDA event to time the execution time of each algo
    if (cudaEventCreate(&startEvent, cudaEventBlockingSync) != cudaSuccess) {
        goto CLEANUP;
    }
    if (cudaEventCreate(&stopEvent, cudaEventBlockingSync) != cudaSuccess) {
        goto CLEANUP;
    }


    // Request the 100 first AlgoId available
    status = cublasLtMatmulAlgoGetIds(cublaslt_handle_, computeType, scaleType, Atype, Btype, Ctype, Ctype, MAX_NUM_ALGO_IDS, algoIdArray, &numAlgosAvailable);
    if (status != CUBLAS_STATUS_SUCCESS && numAlgosAvailable == 0) {
        goto CLEANUP;
    }
    LOGD("CublasLt FindAlgo: Found [%d] AlgoIDs available.\n", numAlgosAvailable);


    // Main Algorithm Loops
    for (int i_algo=0; i_algo < numAlgosAvailable; i_algo++) {
        cublasLtMatmulAlgo_t algo;
        size_t               sizeWritten = 0;

        // Initialize algo structure with given Algp ID
        status = cublasLtMatmulAlgoInit(cublaslt_handle_, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdArray[i_algo], &algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            continue;
        }

        // Query the tiles enums supported by that algo
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sizeWritten);
        int  nbTiles = int(sizeWritten / sizeof(int));
        int* tileArray   = new int[nbTiles == 0 ? 1 : nbTiles];
        if (nbTiles == 0) {
            tileArray[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles  = 1;
        }

#if (CUDART_VERSION >= 11000)
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, NULL, 0, &sizeWritten);
        int nbStages = int(sizeWritten / sizeof(int));
        std::vector<int> stagesA(nbStages == 0 ? 1 : nbStages);
        if (nbStages == 0) {
            stagesA[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
            nbStages   = 1;
        } else {
            cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stagesA.data(), sizeof(int) * nbStages, &sizeWritten);
        }
#endif

        int splitkSupport, redMask, swizzlingMax, customOptionMax;
        // Retrieve Algo Capabilities attributes to be able to setup loop over
        // the different combinations
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileArray, sizeof(int) * nbTiles, &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten);

        // Loop over the different tiles
        for (int tileIdx = 0; tileIdx < nbTiles; tileIdx++) {
#if (CUDART_VERSION >= 11000)
            // Loop over different stages count
            for (int stagesIdx = 0; stagesIdx < nbStages; stagesIdx++) {
                cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesA[stagesIdx], sizeof(stagesA[stagesIdx]));
#endif
                // Loop over the different custom option if any
                for (int customOption = 0; customOption <= customOptionMax; customOption++) {
                    cublasLtMatmulAlgoConfigSetAttribute(
                        &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption));
                    // Loop over the CTAs swizzling support
                    for (int k = 0; k <= swizzlingMax; k++) {
                        int splitK_trial = 0;
                        if (splitkSupport) {
                            splitK_trial += sizeof(splitKSequenceArray) / sizeof(splitKSequenceArray[0]);
                        }
                        // Loop over the splitK value over a fixed sequence
                        // splitKSequenceArray in addition to the case where splitK
                        // is not enabled
                        for (int l = 0; (l < (1 + splitK_trial)) && (AlgoCount < MAX_NUM_ALGO_COMBINATIONS); l++) {
                            // Setup attribute of the algo to run
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileArray[tileIdx], sizeof(tileArray[tileIdx]));
                            int splitK_val = 0;
                            int redScheme  = CUBLASLT_REDUCTION_SCHEME_NONE;
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val));
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k));
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int));

                            if (l > 0) {  // Split-K case
                                splitK_val = splitKSequenceArray[l - 1];
                                cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                                     CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                                                     &splitKSequenceArray[l - 1],
                                                                     sizeof(splitKSequenceArray[l - 1]));
                                // Going over all the reduction scheme
                                for (redScheme = 1;
                                     redScheme < (int)CUBLASLT_REDUCTION_SCHEME_MASK && (AlgoCount < MAX_NUM_ALGO_COMBINATIONS);
                                     redScheme = redScheme << 1) {
                                    if (redScheme & redMask) {
                                        cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                                             CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                                             &redScheme,
                                                                             sizeof(redScheme));

                                        cublasLtMatmulHeuristicResult_t heurResult;
                                        cublasStatus_t                  algoStatus = cublasLtMatmulAlgoCheck(
                                            cublaslt_handle_, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, &algo, &heurResult);
                                        if (heurResult.workspaceSize > max_workspace_size) {
                                            LOGD("CublasLt MatMul Perf: No enough workspace. Required=%d, Max Size Available=%d.\n", heurResult.workspaceSize, max_workspace_size);
                                            algoStatus = CUBLAS_STATUS_NOT_SUPPORTED;  // Not enough workspace
                                        } else if (heurResult.workspaceSize == 0) {
                                            if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                                                algosNoWorkspace[AlgoCountNoWorkspace++] = algo;
                                            }
                                        }
                                        if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                                            algos[AlgoCount++] = algo;
                                        }
                                    }  // end if
                                }      // end for
                            } else {  // Non-splitK case
                                // if user preference is ok with workspace
                                if (AlgoCount < MAX_NUM_ALGO_COMBINATIONS) {
                                    cublasLtMatmulHeuristicResult_t heurResult;
                                    cublasStatus_t                  algoStatus = cublasLtMatmulAlgoCheck(
                                        cublaslt_handle_, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, &algo, &heurResult);
                                    if (heurResult.workspaceSize > max_workspace_size) {
                                        LOGD("CublasLt MatMul Perf: No enough workspace. Required=%d, Max Size Available=%d.\n", heurResult.workspaceSize, max_workspace_size);
                                        algoStatus = CUBLAS_STATUS_NOT_SUPPORTED;  // Not Enough Workspace
                                    } else if (heurResult.workspaceSize == 0) {
                                        if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                                            algosNoWorkspace[AlgoCountNoWorkspace++] = algo;
                                        }
                                    }
                                    if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                                        algos[AlgoCount++] = algo;
                                    }
                                }
                            }
                        }  // end l
                    }      // end k
                }          // end customOption
#if (CUDART_VERSION >= 11000)
            }  // end stagesIdx
#endif
        }  // end tileIdx
        delete[] tileArray;
    }  // Algo i loop
    LOGD("CublasLt FindAlgo: Found [%d] Algos total, include [%d] Algos without workspace.\n", AlgoCount, AlgoCountNoWorkspace);

    if (AlgoCount < NUM_MAX_TRAVERSAL) {
        // 0 <= workspacesize <= 32MB
        for (int i_algo = 0; i_algo < AlgoCount; i_algo++) {
            status = RunCublasLtMMAlgoPerf(operationDesc,
                                           alpha, // host or device pointer
                                           A,
                                           Adesc,
                                           B,
                                           Bdesc,
                                           beta, // host or device pointer
                                           C,
                                           Cdesc,
                                           C,
                                           Cdesc,
                                           algos[i_algo],
                                           NUM_KERNEL_REPEATS,
                                           max_workspace_size,
                                           perfResults[i_algo],
                                           stream,
                                           startEvent,
                                           stopEvent);
            perfResults[i_algo].status = status;
        }
    } else {
        // Heuristic + workspacesize==0
        AlgoCount = 0;
        numAlgosAvailable = 0;
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_size, sizeof(max_workspace_size)); // (Recommend: 32MB)
        cublasLtMatmulHeuristicResult_t heuristicResultsArray[NUM_MAX_TRAVERSAL];

        cublasLtMatmulAlgoGetHeuristic(cublaslt_handle_,
                                       operationDesc,
                                       Adesc,
                                       Bdesc,
                                       Cdesc,
                                       Cdesc,
                                       pref,
                                       NUM_MAX_TRAVERSAL,
                                       heuristicResultsArray,
                                       &numAlgosAvailable);
        cublasLtMatmulPreferenceDestroy(pref);
        LOGD("CublasLt MatMul Perf: heuristic suggest return %d algo IDs.\n", numAlgosAvailable);

        for (int i_algo = 0; i_algo < numAlgosAvailable; i_algo++) {
            if (heuristicResultsArray[i_algo].state == CUBLAS_STATUS_SUCCESS) {
                status = RunCublasLtMMAlgoPerf(operationDesc,
                                               alpha, // host or device pointer
                                               A,
                                               Adesc,
                                               B,
                                               Bdesc,
                                               beta, // host or device pointer
                                               C,
                                               Cdesc,
                                               C,
                                               Cdesc,
                                               heuristicResultsArray[i_algo].algo,
                                               NUM_KERNEL_REPEATS,
                                               max_workspace_size,
                                               perfResults[AlgoCount],
                                               stream,
                                               startEvent,
                                               stopEvent);
                perfResults[AlgoCount].status = status;
                if (status == CUBLAS_STATUS_SUCCESS) {
                    AlgoCount++;
                }
            }
        }

        // workspacesize==0
        LOGD("CublasLt MatMul Perf: Run %d Algos with NO Workspace.\n", AlgoCountNoWorkspace);
        for (int i_algo = 0; i_algo < AlgoCountNoWorkspace && i_algo < (NUM_MAX_TRAVERSAL - numAlgosAvailable); i_algo++) {
            status = RunCublasLtMMAlgoPerf(operationDesc,
                                           alpha, // host or device pointer
                                           A,
                                           Adesc,
                                           B,
                                           Bdesc,
                                           beta, // host or device pointer
                                           C,
                                           Cdesc,
                                           C,
                                           Cdesc,
                                           algosNoWorkspace[i_algo],
                                           NUM_KERNEL_REPEATS,
                                           0,
                                           perfResults[AlgoCount],
                                           stream,
                                           startEvent,
                                           stopEvent);
            perfResults[AlgoCount].status = status;
            if (status == CUBLAS_STATUS_SUCCESS) {
                AlgoCount++;
            }
        }
    }

    // Sort the results per run duration
    std::sort(perfResults, perfResults + AlgoCount,
        [](const CublasMMAlgoInfo& perf_a, const CublasMMAlgoInfo& perf_b) {
            return ((perf_a.status == CUBLAS_STATUS_SUCCESS) && (perf_a.exec_time < perf_b.exec_time));} );

    if (perfResults[0].exec_time > 0.0f) {
        best_cublaslt_algo = perfResults[0];
        InitCublasLtDesc(DataType(perfResults[0].dtype), transa, transb, m, n, k, lda, ldb, ldc, best_cublaslt_algo);

        LOGD("CublasLt MatMul Perf: Best Algo: AlgoId=[%d], exec_time=[%.4f]ms, tile=%d, customOption=%d, numSplitsK=%d, swizzle=%d, reductionScheme=%d, workspaceSize=%d, stages=%d, wavesCount=%.4f.\n",
             best_cublaslt_algo.algoId, best_cublaslt_algo.exec_time,
             best_cublaslt_algo.tile, best_cublaslt_algo.customOption, best_cublaslt_algo.splitK_val,
             best_cublaslt_algo.swizzle, best_cublaslt_algo.reductionScheme, best_cublaslt_algo.workspaceSize,
             best_cublaslt_algo.stages, best_cublaslt_algo.wavesCount);
    }


CLEANUP:
    // Descriptors are no longer needed as all GPU work was already enqueued
    if (Cdesc) {
        cublasLtMatrixLayoutDestroy(Cdesc);
    }
    if (Bdesc) {
        cublasLtMatrixLayoutDestroy(Bdesc);
    }
    if (Adesc) {
        cublasLtMatrixLayoutDestroy(Adesc);
    }
    if (operationDesc) {
        cublasLtMatmulDescDestroy(operationDesc);
    }
    if (startEvent) {
        cudaEventDestroy(startEvent);
    }
    if (stopEvent) {
        cudaEventDestroy(stopEvent);
    }
    return status;
}


template <typename dtype>
cublasStatus_t CublasMMWrapper::FindBestCublasMMAlgo(cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     const int         batch,
                                                     const int         m,
                                                     const int         n,
                                                     const int         k,
                                                     const dtype*      A,
                                                     const int         lda,
                                                     const dtype*      B,
                                                     const int         ldb,
                                                     dtype*            C,
                                                     const int         ldc,
                                                     const void*       alpha,  // host pointer
                                                     const void*       beta,   // host pointer
                                                     CublasMMAlgoInfo& best_cublas_algo,
                                                     cudaStream_t      stream) {
    const int NUM_KERNEL_REPEATS = 100;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    cudaDataType_t AType;
    cudaDataType_t BType;
    cudaDataType_t CType;
    cudaDataType_t computeType;
    int            startAlgo, endAlgo;

    if (std::is_same<dtype, float>::value) {
        AType       = CUDA_R_32F;
        BType       = CUDA_R_32F;
        CType       = CUDA_R_32F;
        computeType = CUDA_R_32F;
        startAlgo   = (int)CUBLAS_GEMM_DEFAULT;
        endAlgo     = (int)CUBLAS_GEMM_ALGO23;
        best_cublas_algo.dtype = int(DATA_TYPE_FLOAT);
    } else if (std::is_same<dtype, half>::value) {
        AType       = CUDA_R_16F;
        BType       = CUDA_R_16F;
        CType       = CUDA_R_16F;
        computeType = CUDA_R_16F;
        startAlgo   = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        endAlgo     = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
        best_cublas_algo.dtype = int(DATA_TYPE_HALF);
    }

    float best_time = 99999.0f;
    int   best_algo = 0;
    for (int i_algo = startAlgo; i_algo <= endAlgo; i_algo++) {
        cublasStatus_t status;
        cudaDeviceSynchronize();
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i_iter = 0; i_iter<NUM_KERNEL_REPEATS; i_iter++) {
            status = cublasGemmEx(cublas_handle_,
                                  transa,
                                  transb,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  A,
                                  AType,
                                  lda,
                                  B,
                                  BType,
                                  ldb,
                                  beta,
                                  C,
                                  CType,
                                  ldc,
                                  computeType,
                                  static_cast<cublasGemmAlgo_t>(i_algo));
        } // i_iter
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        float avg_time = std::chrono::duration<float, std::milli>(end_time - start_time).count() / NUM_KERNEL_REPEATS;
        if (status == CUBLAS_STATUS_SUCCESS) {
            //LOGD("algo_%d costs %.3fms \n", i_algo, avg_time);
            if (avg_time < best_time) {
                best_time = avg_time;
                best_algo = i_algo;
            }
        }
    }  // i_algo
    LOGD("CublasLt MatMul Perf: Best Algo id = %d, time = %.4f ms\n", best_algo, best_time);

    if (best_time != 99999.0f) {
        best_cublas_algo.algoId      = best_algo;
        best_cublas_algo.exec_time   = best_time;
        best_cublas_algo.batch       = 1;
        best_cublas_algo.is_cublaslt = false;
    }

    return status;
}

template <typename dtype>
cublasStatus_t CublasMMWrapper::FindBestMMAlgo(cublasOperation_t transa,
                                               cublasOperation_t transb,
                                               const int         batch,
                                               const int         m,
                                               const int         n,
                                               const int         k,
                                               const dtype*      A,
                                               const int         lda,
                                               const dtype*      B,
                                               const int         ldb,
                                               dtype*            C,
                                               const int         ldc,
                                               const void*       alpha,  // host pointer
                                               const void*       beta,   // host pointer
                                               CublasMMAlgoInfo& best_algo_info,
                                               cudaStream_t      stream,
                                               const int         max_workspace_size,
                                               const bool        enable_cublas_algo,
                                               const bool        enable_cublaslt_algo) {
    if (!enable_cublas_algo && !enable_cublaslt_algo) {
        LOGD("Choose At least one of Classical Cublas and CublasLt to find Best Algo.\n");
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    // Try to find out if Best Also has already been placed in this->algo_map_;
    CublasMMConfig config;
    if (std::is_same<dtype, float>::value) {
        float f_alpha = reinterpret_cast<const float*>(alpha)[0];
        float f_beta  = reinterpret_cast<const float*>(beta)[0];
        config = CublasMMConfig(int(DATA_TYPE_FLOAT), batch, transa, transb, m, n, k, lda, ldb, ldc, f_alpha, f_beta, max_workspace_size);
    } else if (std::is_same<dtype, half>::value) {
        half h_alpha = reinterpret_cast<const half*>(alpha)[0];
        half h_beta  = reinterpret_cast<const half*>(beta)[0];
        config = CublasMMConfig(int(DATA_TYPE_HALF), batch, transa, transb, m, n, k, lda, ldb, ldc, h_alpha, h_beta, max_workspace_size);
    } else {
        LOGE("Unable to find Best Cublas MM Algo, data type not supported.");
        return status;
    }

    auto iter = this->algo_map_.find(config);
    if (iter != this->algo_map_.end()) {
        best_algo_info = iter->second;
        return status;
    }

    // If best algo is not set yet.
    if (enable_cublas_algo) {
        cublasStatus_t status = FindBestCublasMMAlgo(transa, transb, batch, m, n, k, A, lda, B, ldb, C, ldc,
                                                     alpha, beta, best_algo_info, stream);
    }
    
    if (enable_cublaslt_algo) {
        CublasMMAlgoInfo best_lt_algo_info;
        status = status = FindBestCublasLtMMAlgo(transa, transb, batch, m, n, k, A, lda, B, ldb, C, ldc,
                                                       alpha, beta, best_lt_algo_info, stream, max_workspace_size);
        if (best_lt_algo_info.exec_time < best_algo_info.exec_time) {
            best_algo_info = best_lt_algo_info;
        }
    }

    this->algo_map_[config] = best_algo_info;

    LOGD("Cublas MatMul Perf: Best Algo: AlgoId=[%d], exec_time=[%.4f]ms, tile=%d, customOption=%d, numSplitsK=%d, swizzle=%d, reductionScheme=%d, workspaceSize=%d, stages=%d, wavesCount=%.4f, is_cublaslt=[%d].\n",
             best_algo_info.algoId, best_algo_info.exec_time,
             best_algo_info.tile, best_algo_info.customOption, best_algo_info.splitK_val,
             best_algo_info.swizzle, best_algo_info.reductionScheme, best_algo_info.workspaceSize,
             best_algo_info.stages, best_algo_info.wavesCount, int(best_algo_info.is_cublaslt));

    /////////////////////////////////
    // TEST Save/Load Module. TO BE REMOVED
    //SaveAlgoMapToFile();
    //LoadAlgoMapFromFile();
    /////////////////////////////////

    return status;
}







class cublasMMWrapper {
public:
    cublasMMWrapper(cublasHandle_t   cublas_handle,
                    cublasLtHandle_t cublaslt_handle);

    ~cublasMMWrapper();

    cublasMMWrapper(const cublasMMWrapper& wrapper);

    void Gemm(cublasOperation_t transa,
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
              cublasGemmAlgo_t  algo);

    void Gemm(cublasOperation_t transa,
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
              cudaStream_t      stream);

    void Gemm(cublasOperation_t transa,
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
              cudaStream_t      stream);

    void batchedGemm(cublasOperation_t  transa,
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
                     cudaStream_t       stream);

    void stridedBatchedGemm(cublasOperation_t transa,
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
                            const float       f_alpha = 1.0f,
                            const float       f_beta  = 0.0f);

    void setFP32GemmConfig();
    void setFP16GemmConfig();

    void prepareCublasLtDesc(cublasOperation_t transa,
                             cublasOperation_t transb,
                             const int         m,
                             const int         n,
                             const int         k,
                             const int         lda,
                             const int         ldb,
                             const int         ldc);

    void freeCublasLtDesc();

private:
    cublasHandle_t   cublas_handle_;
    cublasLtHandle_t cublaslt_handle_;

    cudaDataType_t Atype_;
    cudaDataType_t Btype_;
    cudaDataType_t Ctype_;
    cudaDataType_t computeType_;

    cublasLtMatmulDesc_t   operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;

    bool cublaslt_inited_ = false;
    cublasOperation_t cached_transa;
    cublasOperation_t cached_transb;
    int cached_m, cached_n, cached_k;
    int cached_lda, cached_ldb, cached_ldc;
};



}  //  namespace TNN_NS;

#endif  //  TNN_SOURCE_TNN_DEVICE_CUDA_UTILS_CUH_

