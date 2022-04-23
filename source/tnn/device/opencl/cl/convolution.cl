#include "base.inc"
#include "activation.inc"
#include "io.inc"

__kernel void Conv2D(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
    __read_only image2d_t weights, __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks,
    __private const int activation_type) {
    // deal with 2 dim image : dim0 = channel + width | dim1 = batch + height
    const int output_cw_idx = get_global_id(0);
    const int output_bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

#ifdef CHECK_INPUT_COOR
    int2 input_dims = get_image_dim(input);
#endif

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

    out0 = ActivationProcess(out0, activation_type);
    out1 = ActivationProcess(out1, activation_type);
    out2 = ActivationProcess(out2, activation_type);
    out3 = ActivationProcess(out3, activation_type);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
}

__kernel void Conv2D_CB2(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
    __read_only image2d_t weights, __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int out_channel_block_length,
    __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int kernel_size,
    __private const int out_width_blocks,
    __private const int activation_type) {
    // deal with 2 dim image : dim0 = channel + width | dim1 = batch + height
    const int output_channel_slice_w_idx = get_global_id(0);
    const int output_bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_channel_slice_w_idx, output_bh_idx);

#ifdef CHECK_INPUT_COOR
    int2 input_dims = get_image_dim(input);
#endif

    const int out_channel_slice_idx = output_channel_slice_w_idx / out_width_blocks;
    const int out_channel_block_idx = out_channel_slice_idx << 1;
    const int out_width_block_idx   = output_channel_slice_w_idx % out_width_blocks;

    FLOAT4 out_w0_s0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out_w1_s0 = out_w0_s0;
    FLOAT4 out_w2_s0 = out_w0_s0;
    FLOAT4 out_w3_s0 = out_w0_s0;

    FLOAT4 out_w0_s1 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx + 1, 0));
    FLOAT4 out_w1_s1 = out_w0_s1;
    FLOAT4 out_w2_s1 = out_w0_s1;
    FLOAT4 out_w3_s1 = out_w0_s1;

    int in_width0 = mad24(out_width_block_idx, stride_wh.x << 2, -padding_wh.x);
    int in_width1 = in_width0 + stride_wh.x;
    int in_width2 = in_width1 + stride_wh.x;
    int in_width3 = in_width2 + stride_wh.x;

    const int height_start = mad24((output_bh_idx % output_wh.y), stride_wh.y, -padding_wh.y);
    int in_height_start = mad24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0),
                                dilation_wh.y, height_start);
    int in_height_end = min(mad24(kernel_wh.y, dilation_wh.y, height_start), input_wh.y);

    const int batch_idx = mul24((output_bh_idx / output_wh.y), input_wh.y);
    int weights_y_idx_s0 = mad24(out_channel_block_idx, kernel_size,
                                 mul24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0), kernel_wh.x));
    int weights_y_idx_s1 = weights_y_idx_s0 + kernel_size;
    int2 weights_y_idx = {weights_y_idx_s0, weights_y_idx_s1};

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights_c0_s0, weights_c1_s0, weights_c2_s0, weights_c3_s0;
    FLOAT4 weights_c0_s1, weights_c1_s1, weights_c2_s1, weights_c3_s1;
    for (int iy = in_height_start; iy < in_height_end; iy += dilation_wh.y) {
        int in_hb_value = iy + batch_idx;
        int4 in_width = {in_width0, in_width1, in_width2, in_width3};
        for (int w = 0; w < kernel_wh.x; w++) {
            int4 weights_x_idx = {0, 1, 2, 3};
            for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
                const int in_idx  = mul24(input_c_block_idx, input_wh.x);
                int4 is_w_in_boundary = (in_width >= 0 && in_width < input_wh.x);
                int4 in_cw_value = in_width + in_idx;
        
                in0 = RI_F(input, SAMPLER, (int2)(select(-1, in_cw_value.x, is_w_in_boundary.x), in_hb_value));
                in1 = RI_F(input, SAMPLER, (int2)(select(-1, in_cw_value.y, is_w_in_boundary.y), in_hb_value));
                in2 = RI_F(input, SAMPLER, (int2)(select(-1, in_cw_value.z, is_w_in_boundary.z), in_hb_value));
                in3 = RI_F(input, SAMPLER, (int2)(select(-1, in_cw_value.w, is_w_in_boundary.w), in_hb_value));

#ifdef CHECK_INPUT_COOR
                if (!InRange((int2)(select(-1, in_cw_value.x, is_w_in_boundary.x), in_hb_value), input_dims)) {
                    in0 = (FLOAT4)0;
                }
                if (!InRange((int2)(select(-1, in_cw_value.y, is_w_in_boundary.y), in_hb_value), input_dims)) {
                    in1 = (FLOAT4)0;
                }
                if (!InRange((int2)(select(-1, in_cw_value.z, is_w_in_boundary.z), in_hb_value), input_dims)) {
                    in2 = (FLOAT4)0;
                }
                if (!InRange((int2)(select(-1, in_cw_value.w, is_w_in_boundary.w), in_hb_value), input_dims)) {
                    in3 = (FLOAT4)0;
                }
#endif

                weights_c0_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.x));
                weights_c1_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, weights_y_idx.x));
                weights_c2_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, weights_y_idx.x));
                weights_c3_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, weights_y_idx.x));

                weights_c0_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.y));
                weights_c1_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, weights_y_idx.y));
                weights_c2_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, weights_y_idx.y));
                weights_c3_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, weights_y_idx.y));

                CALCULATE_SLICE_OUTPUT(0);
                CALCULATE_SLICE_OUTPUT(1);

                weights_x_idx += 4;
            }
            weights_y_idx++;
            in_width += dilation_wh.x;
        }
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

