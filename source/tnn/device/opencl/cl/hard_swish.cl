#include "base.inc"

__kernel void BinaryElementWise_HardSwish(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input0,
    __read_only image2d_t input1, __write_only image2d_t output,
    __private const float alpha, __private const float beta) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    FLOAT4 in0 = RI_F(input0, SAMPLER, (int2)(cw, bh));
    FLOAT4 in1 = RI_F(input1, SAMPLER, (int2)(cw, bh));
    FLOAT4 out = OPERATOR;

    WI_F(output, (int2)(cw, bh), out);
}

__kernel void BinarySingle_HardSwish(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t param,
    __write_only image2d_t output, __private const float alpha,
    __private const float beta) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    FLOAT4 in0 = RI_F(input, SAMPLER, (int2)(cw, bh));
    FLOAT in1  = RI_F(param, SAMPLER, (int2)(0, 0)).x;
    FLOAT4 out = OPERATOR;

    WI_F(output, (int2)(cw, bh), out);
}

__kernel void BinaryChannel_HardSwish(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t param,
    __private const int width, __write_only image2d_t output,
    __private const float alpha, __private const float beta) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    const int c_idx = cw / width;
    FLOAT4 in0      = RI_F(input, SAMPLER, (int2)(cw, bh));
    FLOAT4 in1      = RI_F(param, SAMPLER, (int2)(c_idx, 0));
    FLOAT4 out = OPERATOR;

    WI_F(output, (int2)(cw, bh), out);
}

__kernel void BinaryCHW_HardSwish(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t param,
    __private const int height, __write_only image2d_t output,
    __private const float alpha, __private const float beta) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    const int h_idx = bh % height;
    FLOAT4 in0      = RI_F(input, SAMPLER, (int2)(cw, bh));
    FLOAT4 in1      = RI_F(param, SAMPLER, (int2)(cw, h_idx));
    FLOAT4 out = OPERATOR;

    WI_F(output, (int2)(cw, bh), out);
}

__kernel void BinaryHW_HardSwish(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                                 __read_only image2d_t param,
                                 __private const int height,
                                 __private const int width,
                                 __write_only image2d_t output,
                                 __private const float alpha,
                                 __private const float beta) {
    const int cw = get_global_id(0);
    const int bh = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(cw, bh);

    const int h_idx = bh % height;
    const int w_idx = cw % width;
    FLOAT4 in0      = RI_F(input, SAMPLER, (int2)(cw, bh));
    FLOAT in1       = RI_F(param, SAMPLER, (int2)(w_idx, h_idx)).x;
    FLOAT4 out = OPERATOR;

    WI_F(output, (int2)(cw, bh), out);
}
