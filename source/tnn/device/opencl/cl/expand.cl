#include "base.inc"

__kernel void Expand(GLOBAL_SIZE_1_DIMS
                     __global float* input,
                     __global float* output,
                     shape_6d output_dims,
                     shape_6d input_step) {
    int index = get_global_id(0);
    DEAL_NON_UNIFORM_DIM1(index);
    int inner_idx = index;
    int input_idx = 0;
    for(int i=INNER_DIMS;i>0;i--) {
        int pos = inner_idx % output_dims[i];
        inner_idx /= output_dims[i];
        input_idx += pos * input_step[i];
    }

    output[index] = input[output_idx];    
}
