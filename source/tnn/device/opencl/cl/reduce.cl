#include "base.inc"

#define REDUCE_INPUTS GLOBAL_SIZE_2_DIMS __read_only image2d_t input,   \
                         __write_only image2d_t output,                 \
                         __private const int input_n,                   \
                         __private const int input_c,                   \
                         __private const int input_h,                   \
                         __private const int input_w,                   \
                         __private const int c4_n,                      \
                         __private const int c4_r,                      \
                         __private const int cw4,                       \
                         __private const int axis_n                     \

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
    r = POSTOPERATOR;

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
    
    r.x = INNEROPERATOR;
    r = POSTOPERATOR;

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
    r = POSTOPERATOR;

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
    r = POSTOPERATOR;

    WI_F(output, (int2)(cw, bh), r);
}