#include "base.inc"

__kernel void Reorg(GLOBAL_SIZE_1_DIMS
                    const __global FLOAT *input, 
                    __global FLOAT *output,
                    int w,
                    int h,
                    int c,
                    int batch,
                    int stride, 
                    int stride_pow2,
                    int forward,
                    int mode
                    ) {
      int i = get_global_id(0);
      DEAL_NON_UNIFORM_DIM1(i);
      int in_index = i;
      int in_w = i%w;
      i = i/w;
      int in_h = i%h;
      i = i/h;
      int in_c = i%c;
      i = i/c;
      int b = i%batch;
      int out_c = c/(stride*stride);

      int c2, offset;
      c2 = select(in_c % out_c, in_c / stride_pow2, mode);
      offset = select(in_c / out_c, in_c % stride_pow2, mode);
      int w2 = in_w*stride + offset % stride;
      int h2 = in_h*stride + offset / stride;
      int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

      if(forward) {
          output[out_index] = input[in_index];
      }
      else {
          output[in_index] = input[out_index];
      }

}
