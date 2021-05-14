#include "base.inc"
#include "activation.inc"
#include "io.inc"

__kernel void Conv2D1x1_S1(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input, /* [w,h] [c%4 * w * c/4, h * b] */
    __read_only image2d_t weights,                  /* [w,h] [cout%4 * cin, cout/4] */
    __read_only image2d_t bias,                     /* [w,h] [cout%4 * cout/4, 1]   */
    __write_only image2d_t output, __private const int2 wh,
    __private const int input_c_blocks,
    __private const int output_w_updiv_4,
    __private const int activation_type) {
    const int output_cw_idx = get_global_id(0);
    const int bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_idx, bh_idx);

    const int output_c_block_idx = output_cw_idx / output_w_updiv_4;
    const int output_w_block_idx = output_cw_idx % output_w_updiv_4;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(output_c_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int input_w_idx0 = output_w_block_idx << 2;
    int input_w_idx1 = input_w_idx0 + 1;
    int input_w_idx2 = input_w_idx0 + 2;
    int input_w_idx3 = input_w_idx0 + 3;

    input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= wh.x);
    input_w_idx1 = select(input_w_idx1, INT_MIN, input_w_idx1 >= wh.x);
    input_w_idx2 = select(input_w_idx2, INT_MIN, input_w_idx2 >= wh.x);
    input_w_idx3 = select(input_w_idx3, INT_MIN, input_w_idx3 >= wh.x);

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    int input_w_base   = 0;
    int weights_w_base = 0;
    for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {
        in0 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx0, bh_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx1, bh_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx2, bh_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx3, bh_idx));

        weights0 = RI_F(weights, SAMPLER, (int2)(weights_w_base, output_c_block_idx));
        weights1 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 1, output_c_block_idx));
        weights2 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 2, output_c_block_idx));
        weights3 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 3, output_c_block_idx));

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);

        input_w_base   += wh.x;
        weights_w_base += 4;
    }

    out0 = ActivationProcess(out0, activation_type);
    out1 = ActivationProcess(out1, activation_type);
    out2 = ActivationProcess(out2, activation_type);
    out3 = ActivationProcess(out3, activation_type);

    const int out_x_base = mul24(output_c_block_idx, wh.x);
    int out_x_idx        = output_w_block_idx << 2;

    const int remain = wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               bh_idx, remain);
}

__kernel void Conv2D1x1(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input, /* [w,h] [c%4 * w * c/4, h * b] */
    __read_only image2d_t weights,                  /* [w,h] [cout%4 * cin, cout/4] */
    __read_only image2d_t bias,                     /* [w,h] [cout%4 * cout/4, 1]   */
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int input_c_blocks, __private const int2 output_wh,
    __private const int2 stride_wh, __private const int output_w_updiv_4,
    __private const int activation_type) {
    const int output_cw_idx = get_global_id(0);
    const int output_bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

    const int output_c_block_idx = output_cw_idx / output_w_updiv_4;
    const int output_w_block_idx = output_cw_idx % output_w_updiv_4;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(output_c_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int input_w_idx0 = mul24(output_w_block_idx, stride_wh.x << 2);
    int input_w_idx1 = input_w_idx0 + stride_wh.x;
    int input_w_idx2 = input_w_idx1 + stride_wh.x;
    int input_w_idx3 = input_w_idx2 + stride_wh.x;

    input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= input_wh.x);
    input_w_idx1 = select(input_w_idx1, INT_MIN, input_w_idx1 >= input_wh.x);
    input_w_idx2 = select(input_w_idx2, INT_MIN, input_w_idx2 >= input_wh.x);
    input_w_idx3 = select(input_w_idx3, INT_MIN, input_w_idx3 >= input_wh.x);

    int b_idx = output_bh_idx / output_wh.y;
    int input_bh_idx = mad24(output_bh_idx % output_wh.y, stride_wh.y, b_idx * input_wh.y);

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {
        int input_w_base   = input_c_block_idx * input_wh.x;
        int weights_w_base = input_c_block_idx << 2;

        in0 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx0, input_bh_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx1, input_bh_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx2, input_bh_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx3, input_bh_idx));

        weights0 = RI_F(weights, SAMPLER, (int2)(weights_w_base, output_c_block_idx));
        weights1 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 1, output_c_block_idx));
        weights2 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 2, output_c_block_idx));
        weights3 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 3, output_c_block_idx));

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);
    }

    out0 = ActivationProcess(out0, activation_type);
    out1 = ActivationProcess(out1, activation_type);
    out2 = ActivationProcess(out2, activation_type);
    out3 = ActivationProcess(out3, activation_type);

    const int out_x_base = mul24(output_c_block_idx, output_wh.x);
    int out_x_idx        = output_w_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
}
