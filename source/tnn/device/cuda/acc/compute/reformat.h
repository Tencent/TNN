#ifndef _TNN_CUDA_REFORMAT_H_
#define _TNN_CUDA_REFORMAT_H_

void NC4HW4ToNCHW(int8_t const *ncqhw4, float *nchw, int32_t N, int32_t C, int32_t H, int32_t W, int32_t srcNStride, int32_t dstNStride, const float scale, cudaStream_t stream);

void NC32HW32ToNCHW(int8_t const *nc32hw32, float *nchw, int32_t N, int32_t C, int32_t H, int32_t W, int32_t srcNStride, int32_t dstNStride, const float scale, cudaStream_t stream);

void NCHWToNC4HW4(const float* nchw, int8_t* ncqhw4, int32_t N, int32_t C, int32_t H, int32_t W, int32_t srcNStride, int32_t dstNStride, const float scale, cudaStream_t stream);

void NCHWToNC32HW32(const float* nchw, int8_t* nc32hw32, int32_t N, int32_t C, int32_t H, int32_t W, int32_t srcNStride, int32_t dstNStride, const float scale, cudaStream_t stream);

#endif