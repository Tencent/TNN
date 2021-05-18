#include "base.inc"
#include "activation.inc"
#include "io.inc"

__kernel void Conv2D1x1GS3D(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input, /* [w,h] [c%4 * w * c/4, h * b] */
    __read_only image2d_t weights,                  /* [w,h] [cout%4 * cin, cout/4] */
    __read_only image2d_t bias,                     /* [w,h] [cout%4 * cout/4, 1]   */
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int input_c_blocks, __private const int2 output_wh,
    __private const int2 stride_wh, __private const int output_w_updiv_4,
    __private const int activation_type) {
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

__kernel void Conv2D1x1GS3D_S1(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input, /* [w,h] [c%4 * w * c/4, h * b] */
    __read_only image2d_t weights,                  /* [w,h] [cout%4 * cin, cout/4] */
    __read_only image2d_t bias,                     /* [w,h] [cout%4 * cout/4, 1]   */
    __write_only image2d_t output, __private const int2 wh,
    __private const int input_c_blocks,
    __private const int output_w_updiv_4,
    __private const int activation_type) {
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

__kernel void Conv2D1x1GS3D_S1_CB2(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input, /* [w,h] [c%4 * w * c/4, h * b] */
    __read_only image2d_t weights,                  /* [w,h] [cout%4 * cin, cout/4] */
    __read_only image2d_t bias,                     /* [w,h] [cout%4 * cout/4, 1]   */
    __write_only image2d_t output, __private const int2 wh,
    __private const int in_channel_block_length, __private const int out_channel_block_length,
    __private const int out_width_blocks,
    __private const int activation_type) {
    // deal with 2 dim image : dim0 = channel + width | dim1 = batch + height
    const int out_channel_slice_idx = get_global_id(0);
    const int out_width_block_idx   = get_global_id(1);
    const int output_bh_idx         = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(out_channel_slice_idx, out_width_block_idx, output_bh_idx);
    const int out_channel_block_idx = out_channel_slice_idx << 1;

    FLOAT4 out_w0_s0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out_w1_s0 = out_w0_s0;
    FLOAT4 out_w2_s0 = out_w0_s0;
    FLOAT4 out_w3_s0 = out_w0_s0;

    FLOAT4 out_w0_s1 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx + 1, 0));
    FLOAT4 out_w1_s1 = out_w0_s1;
    FLOAT4 out_w2_s1 = out_w0_s1;
    FLOAT4 out_w3_s1 = out_w0_s1;

    int in_width0 = out_width_block_idx << 2;
    int in_width1 = in_width0 + 1;
    int in_width2 = in_width0 + 2;
    int in_width3 = in_width0 + 3;
    int4 in_width = {in_width0, in_width1, in_width2, in_width3};
    int4 weights_x_idx = {0, 1, 2, 3};
    int out_channel_block_idx_s1 = out_channel_block_idx + 1;

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights_c0_s0, weights_c1_s0, weights_c2_s0, weights_c3_s0;
    FLOAT4 weights_c0_s1, weights_c1_s1, weights_c2_s1, weights_c3_s1;
    for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
        in0 = RI_F(input, SAMPLER, (int2)(in_width.x, output_bh_idx));
        in1 = RI_F(input, SAMPLER, (int2)(in_width.y, output_bh_idx));
        in2 = RI_F(input, SAMPLER, (int2)(in_width.z, output_bh_idx));
        in3 = RI_F(input, SAMPLER, (int2)(in_width.w, output_bh_idx));

        weights_c0_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, out_channel_block_idx));
        weights_c1_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, out_channel_block_idx));
        weights_c2_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, out_channel_block_idx));
        weights_c3_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, out_channel_block_idx));

        weights_c0_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, out_channel_block_idx_s1));
        weights_c1_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, out_channel_block_idx_s1));
        weights_c2_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, out_channel_block_idx_s1));
        weights_c3_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, out_channel_block_idx_s1));

        CALCULATE_SLICE_OUTPUT(0);
        CALCULATE_SLICE_OUTPUT(1);

        weights_x_idx += 4;
        in_width += wh.x;
    }

    out_w0_s0 = ActivationProcess(out_w0_s0, activation_type);
    out_w1_s0 = ActivationProcess(out_w1_s0, activation_type);
    out_w2_s0 = ActivationProcess(out_w2_s0, activation_type);
    out_w3_s0 = ActivationProcess(out_w3_s0, activation_type);

    out_w0_s1 = ActivationProcess(out_w0_s1, activation_type);
    out_w1_s1 = ActivationProcess(out_w1_s1, activation_type);
    out_w2_s1 = ActivationProcess(out_w2_s1, activation_type);
    out_w3_s1 = ActivationProcess(out_w3_s1, activation_type);

    const int out_x_base = mul24(out_channel_block_idx, wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = wh.x - out_x_idx;
    int output_w_idx_s0 = out_x_base + out_x_idx;
    WriteSliceOutputAntiOutOfBounds(output, out_w0_s0, out_w1_s0,
                                    out_w2_s0, out_w3_s0, output_w_idx_s0,
                                    output_bh_idx, remain);

    bool is_s1_in_boundary = (out_channel_block_idx + 1 < out_channel_block_length);
    if (!is_s1_in_boundary) return;
    int output_w_idx_s1 = output_w_idx_s0 + wh.x;
    WriteSliceOutputAntiOutOfBounds(output, out_w0_s1, out_w1_s1,
                                    out_w2_s1, out_w3_s1, output_w_idx_s1,
                                    output_bh_idx, remain);
}

__kernel void Conv2D1x1GS3D_CB2(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input, /* [w,h] [c%4 * w * c/4, h * b] */
    __read_only image2d_t weights,                  /* [w,h] [cout%4 * cin, cout/4] */
    __read_only image2d_t bias,                     /* [w,h] [cout%4 * cout/4, 1]   */
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int out_channel_block_length,
    __private const int2 output_wh,
    __private const int2 stride_wh, __private const int out_width_blocks,
    __private const int activation_type) {
    // deal with 2 dim image : dim0 = channel + width | dim1 = batch + height
    const int out_channel_slice_idx = get_global_id(0);
    const int out_width_block_idx   = get_global_id(1);
    const int output_bh_idx         = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(out_channel_slice_idx, out_width_block_idx, output_bh_idx);
    const int out_channel_block_idx = out_channel_slice_idx << 1;

    FLOAT4 out_w0_s0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out_w1_s0 = out_w0_s0;
    FLOAT4 out_w2_s0 = out_w0_s0;
    FLOAT4 out_w3_s0 = out_w0_s0;

    FLOAT4 out_w0_s1 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx + 1, 0));
    FLOAT4 out_w1_s1 = out_w0_s1;
    FLOAT4 out_w2_s1 = out_w0_s1;
    FLOAT4 out_w3_s1 = out_w0_s1;

    int in_width0 = mul24(out_width_block_idx, stride_wh.x << 2);
    int in_width1 = in_width0 + stride_wh.x;
    int in_width2 = in_width1 + stride_wh.x;
    int in_width3 = in_width2 + stride_wh.x;
    int4 in_width = {in_width0, in_width1, in_width2, in_width3};
    int4 weights_x_idx = {0, 1, 2, 3};

    const int batch_idx     = mul24((output_bh_idx / output_wh.y), input_wh.y);
    const int in_hb_value   = mad24(output_bh_idx % output_wh.y, stride_wh.y, batch_idx);
    int out_channel_block_idx_s1 = out_channel_block_idx + 1;

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights_c0_s0, weights_c1_s0, weights_c2_s0, weights_c3_s0;
    FLOAT4 weights_c0_s1, weights_c1_s1, weights_c2_s1, weights_c3_s1;
    for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
        in0 = RI_F(input, SAMPLER, (int2)(in_width.x, in_hb_value));
        in1 = RI_F(input, SAMPLER, (int2)(in_width.y, in_hb_value));
        in2 = RI_F(input, SAMPLER, (int2)(in_width.z, in_hb_value));
        in3 = RI_F(input, SAMPLER, (int2)(in_width.w, in_hb_value));

        weights_c0_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, out_channel_block_idx));
        weights_c1_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, out_channel_block_idx));
        weights_c2_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, out_channel_block_idx));
        weights_c3_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, out_channel_block_idx));

        weights_c0_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, out_channel_block_idx_s1));
        weights_c1_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, out_channel_block_idx_s1));
        weights_c2_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, out_channel_block_idx_s1));
        weights_c3_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, out_channel_block_idx_s1));

        CALCULATE_SLICE_OUTPUT(0);
        CALCULATE_SLICE_OUTPUT(1);

        weights_x_idx += 4;
        in_width += input_wh.x;
    }

    out_w0_s0 = ActivationProcess(out_w0_s0, activation_type);
    out_w1_s0 = ActivationProcess(out_w1_s0, activation_type);
    out_w2_s0 = ActivationProcess(out_w2_s0, activation_type);
    out_w3_s0 = ActivationProcess(out_w3_s0, activation_type);

    out_w0_s1 = ActivationProcess(out_w0_s1, activation_type);
    out_w1_s1 = ActivationProcess(out_w1_s1, activation_type);
    out_w2_s1 = ActivationProcess(out_w2_s1, activation_type);
    out_w3_s1 = ActivationProcess(out_w3_s1, activation_type);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx_s0 = out_x_base + out_x_idx;
    WriteSliceOutputAntiOutOfBounds(output, out_w0_s0, out_w1_s0,
                                    out_w2_s0, out_w3_s0, output_w_idx_s0,
                                    output_bh_idx, remain);

    bool is_s1_in_boundary = (out_channel_block_idx + 1 < out_channel_block_length);
    if (!is_s1_in_boundary) return;
    int output_w_idx_s1 = output_w_idx_s0 + output_wh.x;
    WriteSliceOutputAntiOutOfBounds(output, out_w0_s1, out_w1_s1,
                                    out_w2_s1, out_w3_s1, output_w_idx_s1,
                                    output_bh_idx, remain);
}