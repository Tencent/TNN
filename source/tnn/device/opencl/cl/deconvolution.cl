#include "base.inc"
#include "activation.inc"
#include "io.inc"

__kernel void Deconv2D(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __read_only image2d_t weights,
                        __read_only image2d_t bias,
                        __write_only image2d_t output,
                        __private const int2 input_wh,
                        __private const int2 output_wh,
                        __private const int2 stride_wh,
                        __private const int2 align_wh,
                        __private const int2 padding_wh, 
                        __private const int2 kernel_wh,
                        __private const int kernel_size,
                        __private const int in_channel_blocks, __private const int out_channel_blocks) {

    const int out_channel_blocks_idx = get_global_id(0);
    const int out_width_idx          = get_global_id(1);
    const int out_batch_height_idx   = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(out_channel_blocks_idx, out_width_idx, out_batch_height_idx);

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_blocks_idx, 0));

    const int out_batch_idx  = out_batch_height_idx / output_wh.y;
    const int out_height_idx = out_batch_height_idx % output_wh.y;

    int kernel_start_x = (out_width_idx + align_wh.x) / stride_wh.x;
    int kernel_start_y = max(0, (out_height_idx + align_wh.y) / stride_wh.y);

    int deal_kernel_width  = kernel_wh.x - mad24(kernel_start_x, stride_wh.x, padding_wh.x) + out_width_idx - 1;
    int deal_kernel_height = kernel_wh.y - mad24(kernel_start_y, stride_wh.y, padding_wh.y) + out_height_idx - 1;

    int kernel_x_0, kernel_x_1, kernel_x_2, kernel_x_3, kernel_y;
    FLOAT4 in0;
    FLOAT4 weights0, weights1, weights2, weights3;
    for (int ic = 0; ic < in_channel_blocks; ic++) {
        kernel_x_0 = ic << 2;
        kernel_x_1 = kernel_x_0 + 1;
        kernel_x_2 = kernel_x_0 + 2;
        kernel_x_3 = kernel_x_0 + 3;
        for (int k_y = deal_kernel_height, idx_h = kernel_start_y; k_y >= 0; k_y -= stride_wh.y, idx_h++) {
            int in_idy      = mad24(out_batch_idx, input_wh.y, idx_h);
            int in_hb_value = select(in_idy, -1, idx_h < 0 || idx_h >= input_wh.y);
            int in_width0   = kernel_start_x;
            for (int k_x = deal_kernel_width; k_x >= 0; k_x -= stride_wh.x) {
                kernel_y = mad24(k_y, kernel_wh.x, k_x);
                kernel_y = mad24(out_channel_blocks_idx, kernel_size, kernel_y);
                weights0 = RI_F(weights, SAMPLER, (int2)(kernel_x_0, kernel_y));
                weights1 = RI_F(weights, SAMPLER, (int2)(kernel_x_1, kernel_y));
                weights2 = RI_F(weights, SAMPLER, (int2)(kernel_x_2, kernel_y));
                weights3 = RI_F(weights, SAMPLER, (int2)(kernel_x_3, kernel_y));

                int in_idx = mul24(ic, input_wh.x);
                int in_width_value0 = in_width0;
                in_width_value0 =
                    select(in_idx + in_width_value0, -1, (in_width_value0 < 0 || in_width_value0 >= input_wh.x));
                in0 = RI_F(input, SAMPLER, (int2)(in_width_value0, in_hb_value));

                out0 = mad(in0.x, weights0, out0);
                out0 = mad(in0.y, weights1, out0);
                out0 = mad(in0.z, weights2, out0);
                out0 = mad(in0.w, weights3, out0);
                in_width0++;
            }
        }
    }

    out0 = ActivationProcess(out0);

    int out_image_width_idx = mad24(out_channel_blocks_idx, output_wh.x, out_width_idx);
    WI_F(output, (int2)(out_image_width_idx, out_batch_height_idx), out0);
}

__kernel void DepthwiseDeconv2D(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __read_only image2d_t weights,
                                 __read_only image2d_t bias,
                                 __write_only image2d_t output,
                                 __private const int2 input_wh,
                                 __private const int2 output_wh,
                                 __private const int2 stride_wh,
                                 __private const int2 align_wh,
                                 __private const int2 padding_wh,
                                 __private const int2 kernel_wh, 
                                 __private const int kernel_size, __private const int out_channel_blocks) {
    const int out_channel_blocks_idx = get_global_id(0);
    const int out_width_idx          = get_global_id(1);
    const int out_batch_height_idx   = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(out_channel_blocks_idx, out_width_idx, out_batch_height_idx);
    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_blocks_idx, 0));

    const int out_batch_idx  = out_batch_height_idx / output_wh.y;
    const int out_height_idx = out_batch_height_idx % output_wh.y;

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

    out0 = ActivationProcess(out0);
    
    const int output_image_x = mad24(out_channel_blocks_idx, output_wh.x, out_width_idx);
    WI_F(output, (int2)(output_image_x, out_batch_height_idx), out0);
}
