#include "base.inc"

__kernel void Expand(GLOBAL_SIZE_1_DIMS
                     __global float* input,
                     __global float* output,
                     shape_6d output_dims,
                     shape_6d expand_input_dims,
                     shape_6d input_step) {
    int index = get_global_id(0);
    DEAL_NON_UNIFORM_DIM1(index);
    int inner_idx = index;
    int input_idx = 0;
    for(int i =  INNER_DIMS - 1; i >= 0 ; i--) {
        int pos = ((inner_idx % output_dims.data[i]) % expand_input_dims.data[i]);
        inner_idx /= output_dims.data[i];
        input_idx += pos * input_step.data[i];
    }

    output[index] = input[input_idx];
}
