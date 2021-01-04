#ifndef TNN_ARM_WINOGRAD_INT8_FUNCTION_H_
#define TNN_ARM_WINOGRAD_INT8_FUNCTION_H_

#include <stdint.h>
#include <stdio.h>

#include "tnn/device/arm/arm_common.h"

#if defined(TNN_USE_NEON) && defined(__aarch64__)

void kernel4x4(const int cin, const int hin, const int win, const int cout,
               const int hout, const int wout, const int8_t *sa,
               const int8_t *sb, int8_t *sc, const float *scale,
               const int32_t *bias, const int pad, int8_t *src_pad_buf,
               int8_t *src_wino_buf, int32_t *dst_wino_buf,
               long relu, const int8_t* add_input, const float* add_scale);

void weight_convert(const int8_t *src, int8_t *dst, int cin, int cout);

#endif

#endif /* WinogradOptFunction_hpp */
