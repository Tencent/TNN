#include "base.inc"

#define REDUCE_INPUTS GLOBAL_SIZE_2_DIMS __read_only image2d_t input,               \
                         __write_only image2d_t output,                             \
                         __private const int input_n,                               \
                         __private const int input_c,                               \
                         __private const int input_h,                               \
                         __private const int input_w,                               \
                         __private const int c4_n,                                  \
                         __private const int c4_r,                                  \
                         __private const int cw4,                                   \
                         __private const int axis_n                                 \

#define REDUCE_MULTI_AXIS_INPUTS GLOBAL_SIZE_2_DIMS __read_only image2d_t input,    \
                            __write_only image2d_t output,                          \
                            __private const int input_n,                            \
                            __private const int input_c,                            \
                            __private const int input_h,                            \
                            __private const int input_w,                            \
                            __private const int c4_n,                               \
                            __private const int c4_r,                               \
                            __private const int cw4,                                \
                            __private const int axis_n,                             \
                            __private const int4 axis_nhwc                          \

#define REDUCE_LOCAL_INPUTS GLOBAL_SIZE_2_DIMS __read_only image2d_t input,         \
                            __write_only image2d_t output,                          \
                            __private const int input_n,                            \
                            __private const int input_c,                            \
                            __private const int input_h,                            \
                            __private const int input_w,                            \
                            __private const int c4_n,                               \
                            __private const int c4_r,                               \
                            __private const int cw4,                                \
                            __private const int axis_n,                             \
                            __private const int local_block_size,                   \
                            __local FLOAT4* local_output                            \

#define REDUCE_WRITE_LOCAL_OUTPUT(local_id, group_size, local_output)               \
    barrier(CLK_LOCAL_MEM_FENCE);                                                   \
    for (unsigned short stride = (group_size >> 1); stride > 0; stride >>= 1) {     \
        if (local_id < stride) {                                                    \
            REDUCEOPERATOR(local_output[local_id], local_output[local_id + stride]) \
        }                                                                           \
        barrier(CLK_LOCAL_MEM_FENCE);                                               \
    }                                                                               \
                                                                                    \
    if (local_id == 0) {                                                            \
        local_output[local_id] = POSTOPERATOR(local_output[local_id]);              \
        WI_F(output, (int2)(cw, bh), local_output[local_id]);                       \
    }                                                                               \

__kernel void ReduceMultiAxis(REDUCE_MULTI_AXIS_INPUTS) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    FLOAT4 t;
    FLOAT4 r = (FLOAT4)(DATAINIT);
    int n_reduce_len = select(1, input_n, axis_nhwc.x);
    int h_reduce_len = select(1, input_h, axis_nhwc.y);
    int w_reduce_len = select(1, input_w, axis_nhwc.z);

    for (unsigned short n = 0; n < n_reduce_len; n++) {
        for (unsigned short h = 0; h < h_reduce_len; h++) {
            for (unsigned short w = 0; w < w_reduce_len; w++) {
                for (unsigned short c_4 = 0; c_4 < select(1, c4_n, axis_nhwc.w); c_4++) {
                    t = RI_F(input, SAMPLER, (int2)(input_w * c_4 + w + cw * w_reduce_len, input_h * n + h + bh * h_reduce_len));
                    OPERATOR(r, t)
                }

                if (axis_nhwc.w) {
                    if (c4_r == 1) {
                        t = RI_F(input, SAMPLER, (int2)(cw4 + w + cw, input_h * n + h + bh));
                        OPERATOR(r.x, t.x)
                    } else if (c4_r == 2) {
                        t = RI_F(input, SAMPLER, (int2)(cw4 + w + cw, input_h * n + h + bh));
                        OPERATOR(r.x, t.x)
                        OPERATOR(r.y, t.y)
                    } else if (c4_r == 3) {
                        t = RI_F(input, SAMPLER, (int2)(cw4 + w + cw, input_h * n + h + bh));
                        OPERATOR(r.x, t.x)
                        OPERATOR(r.y, t.y)
                        OPERATOR(r.z, t.z)
                    }
                }
            }
        }
    }

    if (axis_nhwc.w) {
        r.x = INNEROPERATOR(r);
    }
    r = POSTOPERATOR(r);

    WI_F(output, (int2)(cw, bh), r);
}

__kernel void ReduceC0(REDUCE_INPUTS) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    FLOAT4 t;
    FLOAT4 r = (FLOAT4)(DATAINIT);

    for (unsigned short i = 0; i < input_n; i++) {
        t = RI_F(input, SAMPLER, (int2)(cw, input_h * i + bh));
        OPERATOR(r, t)
    }
    r = POSTOPERATOR(r);

    WI_F(output, (int2)(cw, bh), r);
}

__kernel void ReduceC1(REDUCE_INPUTS) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    FLOAT4 t;
    FLOAT4 r = (FLOAT4)(DATAINIT);

    for (unsigned short i = 0; i < c4_n; i++) {
        t = RI_F(input, SAMPLER, (int2)(input_w * i + cw, bh));
        OPERATOR(r, t)
    }
    if (c4_r == 1) {
        t = RI_F(input, SAMPLER, (int2)(cw4 + cw, bh));
        OPERATOR(r.x, t.x)
    } else if (c4_r == 2) {
        t = RI_F(input, SAMPLER, (int2)(cw4 + cw, bh));
        OPERATOR(r.x, t.x)
        OPERATOR(r.y, t.y)
    } else if (c4_r == 3) {
        t = RI_F(input, SAMPLER, (int2)(cw4 + cw, bh));
        OPERATOR(r.x, t.x)
        OPERATOR(r.y, t.y)
        OPERATOR(r.z, t.z)
    }
    
    r.x = INNEROPERATOR(r);
    r = POSTOPERATOR(r);

    WI_F(output, (int2)(cw, bh), r);
}

__kernel void ReduceC2(REDUCE_INPUTS) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    FLOAT4 t;
    FLOAT4 r = (FLOAT4)(DATAINIT);

    for (unsigned short i = 0; i < input_h; i++) {
        t = RI_F(input, SAMPLER, (int2)(cw, input_h * bh + i));
        OPERATOR(r, t)
    }
    r = POSTOPERATOR(r);

    WI_F(output, (int2)(cw, bh), r);
}

__kernel void ReduceC3(REDUCE_INPUTS) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    FLOAT4 t;
    FLOAT4 r = (FLOAT4)(DATAINIT);

    for (unsigned short i = 0; i < input_w; i++) {
        t = RI_F(input, SAMPLER, (int2)(cw * input_w + i, bh));
        OPERATOR(r, t)
    }
    r = POSTOPERATOR(r);

    WI_F(output, (int2)(cw, bh), r);
}

__kernel void ReduceC0Local(REDUCE_LOCAL_INPUTS) {
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int global_id = get_global_id(0);
    const int cw = global_id / group_size;
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(global_id, bh);

    FLOAT4 t;
    local_output[local_id] = (FLOAT4)(DATAINIT);
    int pos = local_id;

    for (unsigned short i = 0; i < local_block_size; i++) {
        if (pos >= input_n) break;
        t = RI_F(input, SAMPLER, (int2)(cw, input_h * pos + bh));
        OPERATOR(local_output[local_id], t)
        pos += group_size;
    }

    REDUCE_WRITE_LOCAL_OUTPUT(local_id, group_size, local_output)
}

__kernel void ReduceC1Local(REDUCE_LOCAL_INPUTS) {
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int global_id = get_global_id(0);
    const int cw = global_id / group_size;
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(global_id, bh);

    FLOAT4 t;
    local_output[local_id] = (FLOAT4)(DATAINIT);
    int pos = local_id;

    for (unsigned short i = 0; i < local_block_size; i++) {
        if (pos >= c4_n) break;
        t = RI_F(input, SAMPLER, (int2)(input_w * pos + cw, bh));
        OPERATOR(local_output[local_id], t)
        pos += group_size;
    }

    if (local_id == 0) {
        if (c4_r == 1) {
            t = RI_F(input, SAMPLER, (int2)(cw4 + cw, bh));
            OPERATOR(local_output[local_id].x, t.x)
        } else if (c4_r == 2) {
            t = RI_F(input, SAMPLER, (int2)(cw4 + cw, bh));
            OPERATOR(local_output[local_id].x, t.x)
            OPERATOR(local_output[local_id].y, t.y)
        } else if (c4_r == 3) {
            t = RI_F(input, SAMPLER, (int2)(cw4 + cw, bh));
            OPERATOR(local_output[local_id].x, t.x)
            OPERATOR(local_output[local_id].y, t.y)
            OPERATOR(local_output[local_id].z, t.z)
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned short stride = (group_size >> 1); stride > 0; stride >>= 1) {
        if (local_id < stride) {
            REDUCEOPERATOR(local_output[local_id], local_output[local_id + stride])
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        local_output[local_id].x = INNEROPERATOR(local_output[local_id]);
        local_output[local_id] = POSTOPERATOR(local_output[local_id]);
        WI_F(output, (int2)(cw, bh), local_output[local_id]);
    }
}

__kernel void ReduceC2Local(REDUCE_LOCAL_INPUTS) {
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int global_id = get_global_id(0);
    const int cw = global_id / group_size;
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(global_id, bh);

    FLOAT4 t;
    local_output[local_id] = (FLOAT4)(DATAINIT);
    int pos = local_id;

    for (unsigned short i = 0; i < local_block_size; i++) {
        if (pos >= input_h) break;
        t = RI_F(input, SAMPLER, (int2)(cw, input_h * bh + pos));
        OPERATOR(local_output[local_id], t)
        pos += group_size;
    }

    REDUCE_WRITE_LOCAL_OUTPUT(local_id, group_size, local_output)
}

__kernel void ReduceC3Local(REDUCE_LOCAL_INPUTS) {
    const int local_id = get_local_id(0);
    const int group_size = get_local_size(0);
    const int global_id = get_global_id(0);
    const int cw = global_id / group_size;
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(global_id, bh);

    FLOAT4 t;
    local_output[local_id] = (FLOAT4)(DATAINIT);
    int pos = local_id;

    for (unsigned short i = 0; i < local_block_size; i++) {
        if (pos >= input_w) break;
        t = RI_F(input, SAMPLER, (int2)(cw * input_w + pos, bh));
        OPERATOR(local_output[local_id], t)
        pos += group_size;
    }

    REDUCE_WRITE_LOCAL_OUTPUT(local_id, group_size, local_output)
}