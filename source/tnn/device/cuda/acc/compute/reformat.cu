#include "tnn/device/cuda/acc/compute/reformat.h"
#include "tnn/device/cuda/fastdiv.h"

#include <algorithm>
#include <cuda.h>
#include <stdint.h>

#define THREADS_PER_BLOCK 128
#define BLOCKS_Y_PER_GRID 65535 //Maximum Y dimension for grids
#define BLOCKS_Z_PER_GRID 65535 //Maximum Z dimension for grids
#define UNROLL_COUNT 4

template <typename T>
__device__ __forceinline__ float toFloat(T x)
{
    return float(x);
}

template <typename T>
__device__ T fromFloat(float x);

template <>
__device__ __forceinline__ int8_t fromFloat<int8_t>(float x)
{
    // The order of the next two statements matters when x is a NaN,
    // because IEEE max/min return the non-NaN operand when one operand
    // is a NaN and the other is not.
    x = fmaxf(x, INT8_MIN);
    x = fminf(x, INT8_MAX);
    return __float2int_rn(x);
}

inline __host__ __device__ constexpr int divUp(int x, int n)
{
    return (x + n - 1) / n;
}

inline __host__ __device__ constexpr int roundUp(int m, int n) { return divUp(m, n) * n; }

struct DivisorParams {
    fastdiv divCHW;
    fastdiv divHW;
    fastdiv divW;
};

void SetDivisorParams(DivisorParams &div_param, int32_t N, int32_t C, int32_t H, int32_t W) {
    div_param.divCHW.init(C*H*W);
    div_param.divHW.init(H*W);
    div_param.divW.init(W);
}

template<typename Func>
void metaLaunch(dim3 gridDim, Func func)
{
    for (uint32_t y = 0; y < gridDim.y; y += BLOCKS_Y_PER_GRID - 1) // Offset needs to be even number to avoid alignment issue.
    {
        for (uint32_t z = 0; z < gridDim.z; z += BLOCKS_Z_PER_GRID - 1)
        {
            dim3 gridDimSub(gridDim.x, std::min((unsigned int) BLOCKS_Y_PER_GRID, gridDim.y - y), std::min((unsigned int) BLOCKS_Z_PER_GRID, gridDim.z - z));
            dim3 gridDimOffset(0, y, z);
            func(gridDimSub, gridDimOffset);
        }
    }
}

static __device__ __forceinline__ int32_t getCoords(int32_t index, const DivisorParams& divParams, int32_t &n, int32_t &c, int32_t &h, int32_t &w, int32_t nStride)
{
    n = index / divParams.divCHW;
    index = index % divParams.divCHW;
    int32_t stridedIndex = n * nStride + index;
    c = index / divParams.divHW;
    index = index % divParams.divHW;
    h = index / divParams.divW;
    w = index % divParams.divW;
    return stridedIndex;
}

// static __device__ __forceinline__ int32_t ncqhw4Addr(int32_t n, int32_t c, int32_t h, int32_t w, int32_t N, int32_t C, int32_t H, int32_t W, int32_t nStride, int32_t &indexNoStride)
// {
//     int32_t part = ((c / 4)*H*W + h*W + w) * 4 + (c & 3);
//     indexNoStride = n * roundUp(C, 4)*H*W + part;
//     return n * nStride + part;
// }

// __global__ void ncqhw4ToNchw(int8_t const *ncqhw4, float *nchw, int32_t N, int32_t C, int32_t H, int32_t W, int32_t srcNStride, int32_t dstNStride, const float* scale, DivisorParams divParams)
// {
//     int32_t D = roundUp(C, 4);

//     int32_t index = threadIdx.x + THREADS_PER_BLOCK *blockIdx.x;
//     if (index < N*C*H*W)
//     {
//         int32_t n, c, h, w;
//         index = getCoords(index, divParams, n, c, h, w, dstNStride);
//         assert(n < N && c < C && h < H && w < W);

//         int32_t inIndexNoStride;
//         int32_t inIndex = ncqhw4Addr(n, c, h, w, N, D, H, W, srcNStride, inIndexNoStride);
//         assert(inIndexNoStride < N*D*H*W);
//         float s{1.0f};
//         if (scale) s = scale[c];
//         float val{toFloat(ncqhw4[inIndex])};
//         nchw[index] = s * val;
//     }
// }

template<int PACK>
__global__ void ncxhwxToNchw(int8_t const *ncxhwx, float *nchw, int32_t N, int32_t C, int32_t H, int32_t W, int32_t srcNStride, int32_t dstNStride, const float scale, DivisorParams divParams)
{
    int32_t D = roundUp(C, PACK);

    int32_t index = threadIdx.x + THREADS_PER_BLOCK *blockIdx.x;
    if (index < N*C*H*W)
    {
        int32_t n, c, h, w;
        index = getCoords(index, divParams, n, c, h, w, dstNStride);
        // assert(n < N && c < C && h < H && w < W);

        int32_t part = ((c / PACK)*H*W + h*W + w) * PACK + (c & (PACK-1));
        int32_t inIndex = n * srcNStride + part;

        // assert(inIndex < N*D*H*W);
        // float s{1.0f};
        // if (scale) s = scale[c];
        float val{toFloat(ncxhwx[inIndex])};
        nchw[index] = scale * val;
    }   
}

__global__ void nchwToNcqhw4(const float* nchw, uint32_t* ncqhw4, int32_t N, int32_t C, int32_t H, int32_t W, int32_t cStride, int32_t srcNStride, int32_t dstNStride, const float scale, DivisorParams divParams)
{
    int32_t const n = blockIdx.y;
    int32_t threadNum = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    int32_t c = threadNum / divParams.divHW;
    threadNum = threadNum % divParams.divHW;
    int32_t h = threadNum / divParams.divW;
    int32_t w = threadNum % divParams.divW;
    // n,c,h,w filled in, c is [0, C/4]

    int32_t const blockBase = n * srcNStride + c * 4 * cStride;

    if (n < N && c < divUp(C, 4))
    {
        uint32_t a[4];
        #pragma unroll
        for (int32_t i = 0, index = blockBase + h * W + w; i < 4; i++, index += cStride)
        {
            bool validElement = (c * 4 + i) < C;
            float val = validElement ? nchw[index] : 0.0f;

            // If the element is not valid, then the scale is not valid.
            // Have to have this check here, otherwise a NaN can be introduced.
            // if (scale != nullptr && validElement)
            if (validElement)
            {
                val = val / scale;
            }

            a[i] = uint32_t(uint8_t(fromFloat<int8_t>(val)));
            // assert(a[i] <= 255);
        }

        // assert(dstNStride % 4 == 0);
        uint32_t combinedVal = a[0] | a[1] << 8 | a[2] << 16 | a[3] << 24;
        int32_t outIndex = n * (dstNStride >> 2) + c * cStride + h * W + w;
        ncqhw4[outIndex] = combinedVal;
    }
}

__global__ void nchwToNcxhwx(const float* nchw, uint32_t* ncxhwx, int32_t N, int32_t C, int32_t H, int32_t W, int32_t cStride, int32_t srcNStride, int32_t dstNStride, const float scale, DivisorParams divParams, fastdiv divPacked)
{
    int32_t const n = blockIdx.y;
    int32_t threadNum = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    int32_t c = threadNum / divParams.divHW;
    threadNum = threadNum % divParams.divHW;
    int32_t h = threadNum / divParams.divW;
    int32_t w = threadNum % divParams.divW;
    // n,c,h,w filled in, c is [0, C/4]

    int32_t const blockBase = n * srcNStride + c * 4 * cStride;
    int32_t c_major = c / divPacked;
    int32_t c_minor = c % divPacked;

    if (n < N && c < divUp(C, 4))
    {
        uint32_t a[4];
        #pragma unroll
        for (int32_t i = 0, index = blockBase + h * W + w; i < 4; i++, index += cStride)
        {
            bool validElement = (c * 4 + i) < C;
            float val = validElement ? nchw[index] : 0.0f;

            // If the element is not valid, then the scale is not valid.
            // Have to have this check here, otherwise a NaN can be introduced.
            // if (scale != nullptr && validElement)
            if (validElement)
            {
                val = val / scale;
            }

            a[i] = uint32_t(uint8_t(fromFloat<int8_t>(val)));
            // assert(a[i] <= 255);
        }

        uint32_t combinedVal = a[0] | a[1] << 8 | a[2] << 16 | a[3] << 24;
        // int32_t outIndex = n * (dstNStride >> 2) + c * cStride + h * W + w;
        int32_t outIndex = n * (dstNStride >> 2) + (c_major * H * W + h * W + w) * int(divPacked) + c_minor;
        ncxhwx[outIndex] = combinedVal;
    }
}

void NC4HW4ToNCHW(int8_t const *ncqhw4, float *nchw, int32_t N, int32_t C, int32_t H, int32_t W, int32_t srcNStride, int32_t dstNStride, const float scale, cudaStream_t stream)
{
    DivisorParams divParams;
    SetDivisorParams(divParams, N, C, H, W);

    // ncqhw4ToNchw <<< divUp(N*C*H*W, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream >>>(ncqhw4, nchw, N, C, H, W, srcNStride, dstNStride, scale, divParams);
    ncxhwxToNchw<4> <<< divUp(N*C*H*W, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream >>>(ncqhw4, nchw, N, C, H, W, srcNStride, dstNStride, scale, divParams);
}

void NC32HW32ToNCHW(int8_t const *nc32hw32, float *nchw, int32_t N, int32_t C, int32_t H, int32_t W, int32_t srcNStride, int32_t dstNStride, const float scale, cudaStream_t stream)
{
    DivisorParams divParams;
    SetDivisorParams(divParams, N, C, H, W);

    ncxhwxToNchw<32> <<< divUp(N*C*H*W, THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream >>>(nc32hw32, nchw, N, C, H, W, srcNStride, dstNStride, scale, divParams);
}

void NCHWToNC4HW4(const float* nchw, int8_t* ncqhw4, int32_t N, int32_t C, int32_t H, int32_t W, int32_t srcNStride, int32_t dstNStride, const float scale, cudaStream_t stream)
{
    DivisorParams divParams;
    SetDivisorParams(divParams, N, divUp(C, 4), H, W);

    int32_t chwq = divUp(C, 4) * H * W;
    auto func = [&](dim3 gridDimSub, dim3 gridDimOffset) {
        int32_t srcOffset = srcNStride * gridDimOffset.y;
        int32_t dstOffset = (dstNStride >> 2) * gridDimOffset.y;
        nchwToNcqhw4<<<gridDimSub, THREADS_PER_BLOCK, 0, stream>>>(nchw + srcOffset, (uint32_t*) ncqhw4 + dstOffset, N, C, H, W, /* CStride= */H * W, srcNStride, dstNStride, scale, divParams);
    };
    dim3 gridDim(divUp(chwq, THREADS_PER_BLOCK), N);
    metaLaunch(gridDim, func);
}

template<int PACK>
void NCHWToNCxHWx(const float *nchw, int8_t *ncxhwx, int32_t N, int32_t C, int32_t H, int32_t W, int32_t srcNStride, int32_t dstNStride, const float scale, cudaStream_t stream) {
    DivisorParams divParams;
    SetDivisorParams(divParams, N, divUp(C, 4), H, W);
    fastdiv divPacked;
    divPacked.init(PACK / 4);

    int32_t chwq = divUp(C, 4) * H * W;
    auto func = [&](dim3 gridDimSub, dim3 gridDimOffset) {
        int32_t srcOffset = srcNStride * gridDimOffset.y;
        int32_t dstOffset = (dstNStride >> 2) * gridDimOffset.y;
        nchwToNcxhwx<<<gridDimSub, THREADS_PER_BLOCK, 0, stream>>>(nchw + srcOffset, (uint32_t*) ncxhwx + dstOffset, N, C, H, W, /* CStride= */H * W, srcNStride, dstNStride, scale, divParams, divPacked);
    };
    dim3 gridDim(divUp(chwq, THREADS_PER_BLOCK), N);
    metaLaunch(gridDim, func);
}


void NCHWToNC32HW32(const float* nchw, int8_t* nc32hw32, int32_t N, int32_t C, int32_t H, int32_t W, int32_t srcNStride, int32_t dstNStride, const float scale, cudaStream_t stream) {
    NCHWToNCxHWx<32>(nchw, nc32hw32, N, C, H, W, srcNStride, dstNStride, scale, stream);
}