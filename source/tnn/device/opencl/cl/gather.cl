#include "base.inc"

__kernel void GatherCommon(GLOBAL_SIZE_3_DIMS
                     __global float* input,
                     __global int* indices, 
                     __global float* output,
                     int inner_size, 
                     int input_outer_step,
                     int output_outer_step) {
    int inner_id = get_global_id(0);
    int indice_id = get_global_id(1);
    int outer_id = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(inner_id, indice_id, outer_id);
    int indice = indices[indice_id];
    int input_index = mad24(outer_id, input_outer_step, inner_id);
    input_index = mad24(indice, inner_size, input_index);
    int output_index = mad24(outer_id, output_outer_step, inner_id);
    output_index = mad24(indice_id, inner_size, output_index);
    output[output_index] = input[input_index];    
}
