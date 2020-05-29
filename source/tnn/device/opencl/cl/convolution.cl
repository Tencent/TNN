#include "base.inc"
#include "activation.inc"
#include "io.inc"

__kernel void Conv2D1x1_S1_MIX(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, 
                          __global const FLOAT *weights_ptr,
                          __global const FLOAT *bias_ptr,
                          __write_only image2d_t output, __private const int2 wh,
                          __private const int input_c_blocks,
                          __private const int output_w_updiv_4) {

    const int output_cw_idx = get_global_id(0); //c/4 w/4
    const int bh_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(output_cw_idx, bh_idx);

    const int output_c_block_idx = output_cw_idx / output_w_updiv_4;
    const int output_w_block_idx = output_cw_idx % output_w_updiv_4;

    FLOAT4 out0 = vload4(output_c_block_idx, (__global FLOAT *)bias_ptr);
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;

    const int out_x_idx = output_w_block_idx << 2;

    int input_w_idx0 = out_x_idx;
    int input_w_idx1 = out_x_idx + 1;
    int input_w_idx2 = out_x_idx + 2;
    int input_w_idx3 = out_x_idx + 3;

    input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= wh.x);
    input_w_idx1 = select(input_w_idx1, INT_MIN, input_w_idx1 >= wh.x);
    input_w_idx2 = select(input_w_idx2, INT_MIN, input_w_idx2 >= wh.x);
    input_w_idx3 = select(input_w_idx3, INT_MIN, input_w_idx3 >= wh.x);

    int input_w_base   = 0;
    int weights_offset = mul24(output_c_block_idx, input_c_blocks << 2);
    for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {
        in0 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx0, bh_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx1, bh_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx2, bh_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx3, bh_idx));

        weights0 = vload4(weights_offset, (__global FLOAT *)weights_ptr);
        weights1 = vload4(weights_offset + 1, (__global FLOAT *)weights_ptr);
        weights2 = vload4(weights_offset + 2, (__global FLOAT *)weights_ptr);
        weights3 = vload4(weights_offset + 3, (__global FLOAT *)weights_ptr);

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);

        input_w_base   += wh.x;
        weights_offset += 4;
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(output_c_block_idx, wh.x);

    const int remain = wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               bh_idx, remain);
}

__kernel void Conv2D1x1_S1(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input, /* [w,h] [c%4 * w * c/4, h * b] */
    __read_only image2d_t weights,                  /* [w,h] [cout%4 * cin, cout/4] */
    __read_only image2d_t bias,                     /* [w,h] [cout%4 * cout/4, 1]   */
    __write_only image2d_t output, __private const int2 wh,
    __private const int input_c_blocks,
    __private const int output_w_updiv_4) {
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

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

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
    __private const int2 stride_wh, __private const int output_w_updiv_4) {
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

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(output_c_block_idx, output_wh.x);
    int out_x_idx        = output_w_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
}

__kernel void Conv2D1x1GS3D_S1(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input, /* [w,h] [c%4 * w * c/4, h * b] */
    __read_only image2d_t weights,                  /* [w,h] [cout%4 * cin, cout/4] */
    __read_only image2d_t bias,                     /* [w,h] [cout%4 * cout/4, 1]   */
    __write_only image2d_t output, __private const int2 wh,
    __private const int input_c_blocks,
    __private const int output_w_updiv_4) {
    const int output_c_block_idx = get_global_id(0);
    const int output_w_block_idx = get_global_id(1);
    const int bh_idx      = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(output_c_block_idx, output_w_block_idx, bh_idx);

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

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(output_c_block_idx, wh.x);
    int out_x_idx        = output_w_block_idx << 2;

    const int remain = wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               bh_idx, remain);
}

__kernel void Conv2D1x1GS3D(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input, /* [w,h] [c%4 * w * c/4, h * b] */
    __read_only image2d_t weights,                  /* [w,h] [cout%4 * cin, cout/4] */
    __read_only image2d_t bias,                     /* [w,h] [cout%4 * cout/4, 1]   */
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int input_c_blocks, __private const int2 output_wh,
    __private const int2 stride_wh, __private const int output_w_updiv_4) {
    const int output_c_block_idx = get_global_id(0);
    const int output_w_block_idx = get_global_id(1);
    const int output_bh_idx      = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(output_c_block_idx, output_w_block_idx, output_bh_idx);

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
    int input_bh_idx = mul24((output_bh_idx % output_wh.y), stride_wh.y) + b_idx * input_wh.y;

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    int input_w_base   = 0;
    int weights_w_base = 0;
    for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {
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

        input_w_base += input_wh.x;
        weights_w_base += 4;
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(output_c_block_idx, output_wh.x);
    int out_x_idx        = output_w_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
}

__kernel void Conv2D(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
    __read_only image2d_t weights, __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks) {
    // deal with 2 dim image : dim0 = channel + width | dim1 = batch + height
    const int output_cw_idx = get_global_id(0);
    const int output_bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

    const int out_channel_block_idx = output_cw_idx / out_width_blocks;
    const int out_width_block_idx   = output_cw_idx % out_width_blocks;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int in_width0 = mad24(out_width_block_idx, stride_wh.x << 2, -padding_wh.x);
    int in_width1 = in_width0 + stride_wh.x;
    int in_width2 = in_width0 + stride_wh.x * 2;
    int in_width3 = in_width0 + stride_wh.x * 3;

    const int height_start = mad24((output_bh_idx % output_wh.y), stride_wh.y, -padding_wh.y);
    int in_height_start = mad24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0), dilation_wh.y, height_start);
    int in_height_end = min(mad24(kernel_wh.y, dilation_wh.y, height_start), input_wh.y);

    const int batch_idx = mul24((output_bh_idx / output_wh.y), input_wh.y);
    const int weights_h_idx = mul24(out_channel_block_idx, mul24(kernel_wh.x, kernel_wh.y)) + 
                              mul24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0), kernel_wh.x);

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
        const int in_idx  = mul24(input_c_block_idx, input_wh.x);
        int weights_x_idx = input_c_block_idx << 2;
        int weights_y_idx = weights_h_idx;
        for (int iy = in_height_start; iy < in_height_end; iy += dilation_wh.y) {
            int in_hb_value = iy + batch_idx;
            for (int w = 0; w < kernel_wh.x; w++) {
                int input_w_base = mul24(w, dilation_wh.x);
                READ_INPUT_IMAGE(0, input_w_base);
                READ_INPUT_IMAGE(1, input_w_base);
                READ_INPUT_IMAGE(2, input_w_base);
                READ_INPUT_IMAGE(3, input_w_base);

                weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx));
                weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx));
                weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx));
                weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx++));

                CALCULATE_OUTPUT(0);
                CALCULATE_OUTPUT(1);
                CALCULATE_OUTPUT(2);
                CALCULATE_OUTPUT(3);
            }
        }
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
}

__kernel void Conv2DGS3D(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
    __read_only image2d_t weights, __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks) {
    // deal with 2 dim image : dim0 = channel + width | dim1 = batch + height
    const int out_channel_block_idx = get_global_id(0);
    const int out_width_block_idx   = get_global_id(1);
    const int output_bh_idx         = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(out_channel_block_idx, out_width_block_idx, output_bh_idx);

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int in_width0 = mad24(out_width_block_idx, stride_wh.x << 2, -padding_wh.x);
    int in_width1 = in_width0 + stride_wh.x;
    int in_width2 = in_width0 + stride_wh.x * 2;
    int in_width3 = in_width0 + stride_wh.x * 3;

    const int height_start = mad24((output_bh_idx % output_wh.y), stride_wh.y, -padding_wh.y);
    int in_height_start = mad24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0),
                                dilation_wh.y, height_start);
    int in_height_end = min(mad24(kernel_wh.y, dilation_wh.y, height_start), input_wh.y);

    const int batch_idx = mul24((output_bh_idx / output_wh.y), input_wh.y);
    const int weights_h_idx = mul24(out_channel_block_idx, mul24(kernel_wh.x, kernel_wh.y)) +
                              mul24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0), kernel_wh.x);

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
        const int in_idx  = mul24(input_c_block_idx, input_wh.x);
        int weights_x_idx = input_c_block_idx << 2;
        int weights_y_idx = weights_h_idx;
        for (int iy = in_height_start; iy < in_height_end; iy += dilation_wh.y) {
            int in_hb_value = iy + batch_idx;
            for (int w = 0; w < kernel_wh.x; w++) {
                int input_w_base = mul24(w, dilation_wh.x);
                READ_INPUT_IMAGE(0, input_w_base);
                READ_INPUT_IMAGE(1, input_w_base);
                READ_INPUT_IMAGE(2, input_w_base);
                READ_INPUT_IMAGE(3, input_w_base);

                weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx));
                weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx));
                weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx));
                weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx++));

                CALCULATE_OUTPUT(0);
                CALCULATE_OUTPUT(1);
                CALCULATE_OUTPUT(2);
                CALCULATE_OUTPUT(3);
            }
        }
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
}

__kernel void DepthwiseConv2DS1(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                                __read_only image2d_t filter,
                                __read_only image2d_t bias,
                                __write_only image2d_t output,
                                __private const int2 input_wh,
                                __private const int2 output_wh,
                                __private const int2 kernel_wh,
                                __private const int2 padding_wh) {
    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightBlockIdx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightBlockIdx);
    int ow4                      = (output_wh.x + 3) / 4;
    const int outChannelBlockIdx = outChannelWidthIdx / ow4;
    const int outWidthBlockidx   = outChannelWidthIdx % ow4;

    const int inChannelBlockIdx = outChannelBlockIdx;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    const int outWidthBlockidx4 = outWidthBlockidx << 2;
    const int in_width0         = outWidthBlockidx4 - padding_wh.x;
    const int in_width1         = in_width0 + 1;
    const int in_width2         = in_width0 + 2;
    const int in_width3         = in_width0 + 3;

    int heightIdx = outHeightBlockIdx % output_wh.y - padding_wh.y;
    const int outBatchIdx =
        mul24((outHeightBlockIdx / output_wh.y), input_wh.y);
    const int in_idx = mul24(inChannelBlockIdx, input_wh.x);

    const int inWidthIdx0 = select(in_idx + in_width0, -1, (in_width0 < 0 || in_width0 >= input_wh.x));
    const int inWidthIdx1 = select(in_idx + in_width1, -1, (in_width1 < 0 || in_width1 >= input_wh.x));
    const int inWidthIdx2 = select(in_idx + in_width2, -1, (in_width2 < 0 || in_width2 >= input_wh.x));

    FLOAT4 in0, in1, in2, in3;
    for (int kh = 0; kh < kernel_wh.y; kh++) {
        int in_hb_value = select(heightIdx + outBatchIdx, -1, (heightIdx < 0 || heightIdx >= input_wh.y));
        heightIdx++;
        in1 = RI_F(input, SAMPLER, (int2)(inWidthIdx0, in_hb_value));
        in2 = RI_F(input, SAMPLER, (int2)(inWidthIdx1, in_hb_value));
        in3 = RI_F(input, SAMPLER, (int2)(inWidthIdx2, in_hb_value));
        for (int kw = 0; kw < kernel_wh.x; kw++) {
            int filterIdx = mad24(kh, kernel_wh.x, kw);

            in0 = in1;
            in1 = in2;
            in2 = in3;

            int inWidthIdx = in_width3 + kw;
            inWidthIdx     = select(in_idx + inWidthIdx, -1, (inWidthIdx < 0 || inWidthIdx >= input_wh.x));
            in3 = RI_F(input, SAMPLER, (int2)(inWidthIdx, in_hb_value));

            FLOAT4 weights = RI_F(filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));

            out0 = mad(in0, weights, out0);
            out1 = mad(in1, weights, out1);
            out2 = mad(in2, weights, out2);
            out3 = mad(in3, weights, out3);
        }
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int remain = output_wh.x - outWidthBlockidx4;
    int outWidthIdx = mul24(outChannelBlockIdx, output_wh.x) + outWidthBlockidx4;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, outWidthIdx,
                               outHeightBlockIdx, remain);
}

__kernel void DepthwiseConv2D(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
    __read_only image2d_t filter, __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int2 output_wh, __private const int2 kernel_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int2 stride_wh) {
    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightIdx       = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightIdx);

    int ow4                      = (output_wh.x + 3) / 4;
    const int outChannelBlockIdx = outChannelWidthIdx / ow4;
    const int outWidthBlockidx   = outChannelWidthIdx % ow4;

    const int inChannelBlockIdx = outChannelBlockIdx;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    const int in_width0 = mad24(outWidthBlockidx, stride_wh.x << 2, -padding_wh.x);
    const int in_width1 = in_width0 + stride_wh.x;
    const int in_width2 = in_width1 + stride_wh.x;
    const int in_width3 = in_width2 + stride_wh.x;
    int heightIdx = mad24(outHeightIdx % output_wh.y, stride_wh.y, -padding_wh.y);

    const int outBatchIdx = mul24((outHeightIdx / output_wh.y), input_wh.y);

    const int in_idx = mul24(inChannelBlockIdx, input_wh.x);
    for (int kh = 0; kh < kernel_wh.y; kh++) {
        int in_hb_value = select(heightIdx + outBatchIdx, -1, (heightIdx < 0 || heightIdx >= input_wh.y));
        heightIdx += dilation_wh.y;
        for (int kw = 0; kw < kernel_wh.x; kw++) {
            int filterIdx = mad24(kh, kernel_wh.x, kw);
            FLOAT4 in0, in1, in2, in3;
            int inWidthIdx = mul24(kw, dilation_wh.x);

            READ_INPUT_IMAGE(0, inWidthIdx);
            READ_INPUT_IMAGE(1, inWidthIdx);
            READ_INPUT_IMAGE(2, inWidthIdx);
            READ_INPUT_IMAGE(3, inWidthIdx);

            FLOAT4 weights = RI_F(filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));

            out0 = mad(in0, weights, out0);
            out1 = mad(in1, weights, out1);
            out2 = mad(in2, weights, out2);
            out3 = mad(in3, weights, out3);
        }
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int outWidthBlockidx4 = outWidthBlockidx << 2;
    const int remain            = output_wh.x - outWidthBlockidx4;
    int outWidthIdx = mul24(outChannelBlockIdx, output_wh.x) + outWidthBlockidx4;

    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, outWidthIdx,
                               outHeightIdx, remain);
}
