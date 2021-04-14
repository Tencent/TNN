#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define GLOBAL_SIZE_1_DIMS __private const int global_size_dim0,   

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM1(input1)       \
    if (input1 >= global_size_dim0) {       \
        return;                             \
    }

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define CALCULATE_OUTPUT(i)                  \
    out##i = mad(in##i.x, weights0, out##i); \
    out##i = mad(in##i.y, weights1, out##i); \
    out##i = mad(in##i.z, weights2, out##i); \
    out##i = mad(in##i.w, weights3, out##i);

#define CALCULATE_VEC16_OUTPUT(i)                  \
    out##i = mad(in##i.x, weights.s0123, out##i); \
    out##i = mad(in##i.y, weights.s4567, out##i); \
    out##i = mad(in##i.z, weights.s89ab, out##i); \
    out##i = mad(in##i.w, weights.scdef, out##i);

#define CALCULATE_SLICE_OUTPUT(s_idx) \
    out_w0_s##s_idx += weights_c0_s##s_idx * in0.x; \
    out_w1_s##s_idx += weights_c0_s##s_idx * in1.x; \
    out_w2_s##s_idx += weights_c0_s##s_idx * in2.x; \
    out_w3_s##s_idx += weights_c0_s##s_idx * in3.x; \
    out_w0_s##s_idx += weights_c1_s##s_idx * in0.y; \
    out_w1_s##s_idx += weights_c1_s##s_idx * in1.y; \
    out_w2_s##s_idx += weights_c1_s##s_idx * in2.y; \
    out_w3_s##s_idx += weights_c1_s##s_idx * in3.y; \
    out_w0_s##s_idx += weights_c2_s##s_idx * in0.z; \
    out_w1_s##s_idx += weights_c2_s##s_idx * in1.z; \
    out_w2_s##s_idx += weights_c2_s##s_idx * in2.z; \
    out_w3_s##s_idx += weights_c2_s##s_idx * in3.z; \
    out_w0_s##s_idx += weights_c3_s##s_idx * in0.w; \
    out_w1_s##s_idx += weights_c3_s##s_idx * in1.w; \
    out_w2_s##s_idx += weights_c3_s##s_idx * in2.w; \
    out_w3_s##s_idx += weights_c3_s##s_idx * in3.w; \

#define CALCULATE_OUTPUT_DOT(i)       \
    out##i.x += dot(in##i, weights0); \
    out##i.y += dot(in##i, weights1); \
    out##i.z += dot(in##i, weights2); \
    out##i.w += dot(in##i, weights3);
