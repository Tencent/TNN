#include "base.inc"
#include "activation.inc"
#include "io.inc"

__kernel void Deconv2D(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t weights,
                       __read_only image2d_t bias,
                       __write_only image2d_t output,
                       __private const int2 input_wh,
                       __private const int2 output_wh,
                       __private const int2 stride_wh,
                       __private const int2 align_wh,
                       __private const int2 padding_wh,
                       __private const int2 kernel_wh,
                       __private const int kernel_size,
                       __private const int in_channel_blocks,
                       __private const int activation_type) {

    const int output_cw_idx = get_global_id(0);
    const int output_bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

#ifdef CHECK_INPUT_COOR
    int2 input_dims = get_image_dim(input);
#endif

    const int out_channel_blocks_idx    = output_cw_idx / output_wh.x;
    const int out_width_idx             = output_cw_idx % output_wh.x;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_blocks_idx, 0));

    const int out_batch_idx  = output_bh_idx / output_wh.y;
    const int out_height_idx = output_bh_idx % output_wh.y;

    int kernel_start_x = (out_width_idx + align_wh.x) / stride_wh.x;
    int kernel_start_y = max(0, (out_height_idx + align_wh.y) / stride_wh.y);

    int deal_kernel_width  = kernel_wh.x - mad24(kernel_start_x, stride_wh.x, padding_wh.x) + out_width_idx - 1;
    int deal_kernel_height = kernel_wh.y - mad24(kernel_start_y, stride_wh.y, padding_wh.y) + out_height_idx - 1;

    int kernel_x_0, kernel_x_1, kernel_x_2, kernel_x_3, kernel_y;
    FLOAT4 in0;
    FLOAT4 weights0, weights1, weights2, weights3;
    for (int k_y = deal_kernel_height, idx_h = kernel_start_y; k_y >= 0; k_y -= stride_wh.y, idx_h++) {
        int in_idy      = mad24(out_batch_idx, input_wh.y, idx_h);
        int in_hb_value = select(in_idy, -1, idx_h < 0 || idx_h >= input_wh.y);
        int in_width0   = kernel_start_x;
        for (int k_x = deal_kernel_width; k_x >= 0; k_x -= stride_wh.x) {
            kernel_y = mad24(k_y, kernel_wh.x, k_x);
            kernel_y = mad24(out_channel_blocks_idx, kernel_size, kernel_y);
            for (int ic = 0; ic < in_channel_blocks; ic++) {
                kernel_x_0 = ic << 2;
                kernel_x_1 = kernel_x_0 + 1;
                kernel_x_2 = kernel_x_0 + 2;
                kernel_x_3 = kernel_x_0 + 3;

                int in_idx = mul24(ic, input_wh.x);
                int in_width_value0 =
                    select(in_idx + in_width0, -1, (in_width0 < 0 || in_width0 >= input_wh.x));
                in0 = RI_F(input, SAMPLER, (int2)(in_width_value0, in_hb_value));

#ifdef CHECK_INPUT_COOR
                if (!InRange((int2)(in_width_value0, in_hb_value), input_dims)) {
                    in0 = (FLOAT4)0;
                }
#endif

                weights0 = RI_F(weights, SAMPLER, (int2)(kernel_x_0, kernel_y));
                out0 = mad(in0.x, weights0, out0);
                weights1 = RI_F(weights, SAMPLER, (int2)(kernel_x_1, kernel_y));
                out0 = mad(in0.y, weights1, out0);
                weights2 = RI_F(weights, SAMPLER, (int2)(kernel_x_2, kernel_y));
                out0 = mad(in0.z, weights2, out0);
                weights3 = RI_F(weights, SAMPLER, (int2)(kernel_x_3, kernel_y));
                out0 = mad(in0.w, weights3, out0);
            }
            in_width0++;
        }
    }

    out0 = ActivationProcess(out0, activation_type);

    WI_F(output, (int2)(output_cw_idx, output_bh_idx), out0);
}

__kernel void Deconv2D4x4s2p1wb4(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t weights,
                                 __read_only image2d_t bias,
                                 __write_only image2d_t output,
                                 __private const int2 input_wh,
                                 __private const int2 output_wh,
                                 __private const int out_width_blocks,
                                 __private const int in_channel_blocks,
                                 __private const int activation_type) {

    const int output_cw_blocks_idx = get_global_id(0);
    const int output_bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_blocks_idx, output_bh_idx);

#ifdef CHECK_INPUT_COOR
    int2 input_dims = get_image_dim(input);
#endif

    const int out_channel_blocks_idx    = output_cw_blocks_idx / out_width_blocks;
    const int out_width_blocks_idx      = output_cw_blocks_idx % out_width_blocks;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_blocks_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    const int out_batch_idx  = output_bh_idx / output_wh.y;
    const int out_height_idx = output_bh_idx % output_wh.y;

    const int in_width_b1       = out_width_blocks_idx << 1;
    const int in_width_b0       = in_width_b1 - 1;
    const int in_width_b2       = in_width_b0 + 2;
    const int in_width_b3       = in_width_b0 + 3;
    const int kernel_start_y    = (out_height_idx - 1) / 2;

    int deal_kernel_height = out_height_idx - (kernel_start_y << 1) + 1;

    int kernel_x_c0, kernel_x_c1, kernel_x_c2, kernel_x_c3;
    int kernel_y_b0, kernel_y_b1, kernel_y_b2, kernel_y_b3;
    FLOAT4 in_b0, in_b1, in_b2, in_b3;
    FLOAT4 weights_c0_b0, weights_c0_b1, weights_c0_b2, weights_c0_b3;
    FLOAT4 weights_c1_b0, weights_c1_b1, weights_c1_b2, weights_c1_b3;
    FLOAT4 weights_c2_b0, weights_c2_b1, weights_c2_b2, weights_c2_b3;
    FLOAT4 weights_c3_b0, weights_c3_b1, weights_c3_b2, weights_c3_b3;
    for (int k_y = deal_kernel_height, idx_h = kernel_start_y; k_y >= 0; k_y -= 2, idx_h++) {
        int in_idy      = mad24(out_batch_idx, input_wh.y, idx_h);
        int in_hb_value = select(in_idy, -1, idx_h < 0 || idx_h >= input_wh.y);
        kernel_y_b0 = (out_channel_blocks_idx * 16) + (k_y * 4);
        kernel_y_b1 = kernel_y_b0 + 1;
        kernel_y_b2 = kernel_y_b0 + 2;
        kernel_y_b3 = kernel_y_b0 + 3;
        for (int ic = 0; ic < in_channel_blocks; ic++) {
            kernel_x_c0 = ic << 2;
            kernel_x_c1 = kernel_x_c0 + 1;
            kernel_x_c2 = kernel_x_c0 + 2;
            kernel_x_c3 = kernel_x_c0 + 3;

            weights_c0_b0 = RI_F(weights, SAMPLER, (int2)(kernel_x_c0, kernel_y_b0));
            weights_c1_b0 = RI_F(weights, SAMPLER, (int2)(kernel_x_c1, kernel_y_b0));
            weights_c2_b0 = RI_F(weights, SAMPLER, (int2)(kernel_x_c2, kernel_y_b0));
            weights_c3_b0 = RI_F(weights, SAMPLER, (int2)(kernel_x_c3, kernel_y_b0));

            weights_c0_b1 = RI_F(weights, SAMPLER, (int2)(kernel_x_c0, kernel_y_b1));
            weights_c1_b1 = RI_F(weights, SAMPLER, (int2)(kernel_x_c1, kernel_y_b1));
            weights_c2_b1 = RI_F(weights, SAMPLER, (int2)(kernel_x_c2, kernel_y_b1));
            weights_c3_b1 = RI_F(weights, SAMPLER, (int2)(kernel_x_c3, kernel_y_b1));

            weights_c0_b2 = RI_F(weights, SAMPLER, (int2)(kernel_x_c0, kernel_y_b2));
            weights_c1_b2 = RI_F(weights, SAMPLER, (int2)(kernel_x_c1, kernel_y_b2));
            weights_c2_b2 = RI_F(weights, SAMPLER, (int2)(kernel_x_c2, kernel_y_b2));
            weights_c3_b2 = RI_F(weights, SAMPLER, (int2)(kernel_x_c3, kernel_y_b2));

            weights_c0_b3 = RI_F(weights, SAMPLER, (int2)(kernel_x_c0, kernel_y_b3));
            weights_c1_b3 = RI_F(weights, SAMPLER, (int2)(kernel_x_c1, kernel_y_b3));
            weights_c2_b3 = RI_F(weights, SAMPLER, (int2)(kernel_x_c2, kernel_y_b3));
            weights_c3_b3 = RI_F(weights, SAMPLER, (int2)(kernel_x_c3, kernel_y_b3));

            int in_idx = mul24(ic, input_wh.x);
            int in_width_value_b0 =
                select(in_idx + in_width_b0, -1, (in_width_b0 < 0 || in_width_b0 >= input_wh.x));
            int in_width_value_b1 =
                select(in_idx + in_width_b1, -1, (in_width_b1 < 0 || in_width_b1 >= input_wh.x));
            int in_width_value_b2 =
                select(in_idx + in_width_b2, -1, (in_width_b2 < 0 || in_width_b2 >= input_wh.x));
            int in_width_value_b3 =
                select(in_idx + in_width_b3, -1, (in_width_b3 < 0 || in_width_b3 >= input_wh.x));
            in_b0 = RI_F(input, SAMPLER, (int2)(in_width_value_b0, in_hb_value));
            in_b1 = RI_F(input, SAMPLER, (int2)(in_width_value_b1, in_hb_value));
            in_b2 = RI_F(input, SAMPLER, (int2)(in_width_value_b2, in_hb_value));
            in_b3 = RI_F(input, SAMPLER, (int2)(in_width_value_b3, in_hb_value));

#ifdef CHECK_INPUT_COOR
            if (!InRange((int2)(in_width_value_b0, in_hb_value), input_dims)) {
                in_b0 = (FLOAT4)0;
            }
            if (!InRange((int2)(in_width_value_b1, in_hb_value), input_dims)) {
                in_b1 = (FLOAT4)0;
            }
            if (!InRange((int2)(in_width_value_b2, in_hb_value), input_dims)) {
                in_b2 = (FLOAT4)0;
            }
            if (!InRange((int2)(in_width_value_b3, in_hb_value), input_dims)) {
                in_b3 = (FLOAT4)0;
            }
#endif

            out0 = mad(in_b0.x, weights_c0_b3, out0);
            out0 = mad(in_b1.x, weights_c0_b1, out0);
            out1 = mad(in_b1.x, weights_c0_b2, out1);
            out1 = mad(in_b2.x, weights_c0_b0, out1);
            out2 = mad(in_b1.x, weights_c0_b3, out2);
            out2 = mad(in_b2.x, weights_c0_b1, out2);
            out3 = mad(in_b2.x, weights_c0_b2, out3);
            out3 = mad(in_b3.x, weights_c0_b0, out3);

            out0 = mad(in_b0.y, weights_c1_b3, out0);
            out0 = mad(in_b1.y, weights_c1_b1, out0);
            out1 = mad(in_b1.y, weights_c1_b2, out1);
            out1 = mad(in_b2.y, weights_c1_b0, out1);
            out2 = mad(in_b1.y, weights_c1_b3, out2);
            out2 = mad(in_b2.y, weights_c1_b1, out2);
            out3 = mad(in_b2.y, weights_c1_b2, out3);
            out3 = mad(in_b3.y, weights_c1_b0, out3);

            out0 = mad(in_b0.z, weights_c2_b3, out0);
            out0 = mad(in_b1.z, weights_c2_b1, out0);
            out1 = mad(in_b1.z, weights_c2_b2, out1);
            out1 = mad(in_b2.z, weights_c2_b0, out1);
            out2 = mad(in_b1.z, weights_c2_b3, out2);
            out2 = mad(in_b2.z, weights_c2_b1, out2);
            out3 = mad(in_b2.z, weights_c2_b2, out3);
            out3 = mad(in_b3.z, weights_c2_b0, out3);

            out0 = mad(in_b0.w, weights_c3_b3, out0);
            out0 = mad(in_b1.w, weights_c3_b1, out0);
            out1 = mad(in_b1.w, weights_c3_b2, out1);
            out1 = mad(in_b2.w, weights_c3_b0, out1);
            out2 = mad(in_b1.w, weights_c3_b3, out2);
            out2 = mad(in_b2.w, weights_c3_b1, out2);
            out3 = mad(in_b2.w, weights_c3_b2, out3);
            out3 = mad(in_b3.w, weights_c3_b0, out3);
        }
    }

    out0 = ActivationProcess(out0, activation_type);
    out1 = ActivationProcess(out1, activation_type);
    out2 = ActivationProcess(out2, activation_type);
    out3 = ActivationProcess(out3, activation_type);

    int out_width_idx   = out_width_blocks_idx << 2;
    const int remain = output_wh.x - out_width_idx;
    int output_cw_idx = output_cw_blocks_idx << 2;

    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_cw_idx,
                               output_bh_idx, remain);
}

__kernel void DepthwiseDeconv2D(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t weights,
                                __read_only image2d_t bias,
                                __write_only image2d_t output,
                                __private const int2 input_wh,
                                __private const int2 output_wh,
                                __private const int2 stride_wh,
                                __private const int2 align_wh,
                                __private const int2 padding_wh,
                                __private const int2 kernel_wh,
                                __private const int kernel_size, __private const int out_channel_blocks,
                                __private const int activation_type) {
    const int output_cw_idx = get_global_id(0);
    const int output_bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

#ifdef CHECK_INPUT_COOR
    int2 input_dims = get_image_dim(input);
#endif

    const int out_channel_blocks_idx    = output_cw_idx / output_wh.x;
    const int out_width_idx             = output_cw_idx % output_wh.x;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_blocks_idx, 0));

    const int out_batch_idx  = output_bh_idx / output_wh.y;
    const int out_height_idx = output_bh_idx % output_wh.y;

    int kernel_start_x = (out_width_idx + align_wh.x) / stride_wh.x;
    int kernel_start_y = max(0, (out_height_idx + align_wh.y) / stride_wh.y);

    int deal_kernel_width  = kernel_wh.x - mad24(kernel_start_x, stride_wh.x, padding_wh.x) + out_width_idx - 1;
    int deal_kernel_height = kernel_wh.y - mad24(kernel_start_y, stride_wh.y, padding_wh.y) + out_height_idx - 1;

    int kernel_image_x;
    FLOAT4 in0;
    FLOAT4 weight;
    int in_width0;
    int in_idx, in_idy;
    for (int k_y = deal_kernel_height, idx_h = kernel_start_y; k_y >= 0; k_y -= stride_wh.y, idx_h++) {
        in_idy          = mad24(out_batch_idx, input_wh.y, idx_h);
        int in_hb_value = select(in_idy, -1, idx_h < 0 || idx_h >= input_wh.y);
        for (int k_x = deal_kernel_width, in_width_idx = kernel_start_x; k_x >= 0; k_x -= stride_wh.x, in_width_idx++) {
            in_width0 = in_width_idx;

            in_idx = mul24(out_channel_blocks_idx, input_wh.x);
            READ_INPUT_IMAGE(0, 0);

            kernel_image_x = mad24(k_y, kernel_wh.x, k_x);
            weight         = RI_F(weights, SAMPLER, (int2)(kernel_image_x, out_channel_blocks_idx));
            out0           = mad(in0, weight, out0);
        }
    }

    out0 = ActivationProcess(out0, activation_type);

    const int output_image_x = mad24(out_channel_blocks_idx, output_wh.x, out_width_idx);
    WI_F(output, (int2)(output_image_x, output_bh_idx), out0);
}
