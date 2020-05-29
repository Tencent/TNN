#include "base.inc"

__kernel void Nearest(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                      __write_only image2d_t output,
                      __private const float height_scale,
                      __private const float width_scale,
                      __private const int input_height,
                      __private const int input_width,
                      __private const int out_height,
                      __private const int out_width) {
    const int output_cw_idx = get_global_id(0);
    const int output_bh_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

    const int output_w_idx       = output_cw_idx % out_width;
    const int output_c_block_idx = output_cw_idx / out_width;
    const int output_b_idx       = output_bh_idx / out_height;
    const int output_h_idx       = output_bh_idx % out_height;

    const float scale_height = output_h_idx * height_scale;
    const float scale_width  = output_w_idx * width_scale;
    const int height_lf      = max(0, (int)floor(scale_height));
    const int width_lf       = max(0, (int)floor(scale_width));

    const int input_w_offset = mul24(output_c_block_idx, input_width);
    const int input_h_offset = mul24(output_b_idx, input_height);

    float4 out = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_lf, input_h_offset + height_lf));

    write_imagef(output, (int2)(output_cw_idx, output_bh_idx), out);
}

__kernel void Bilinear(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                       __write_only image2d_t output,
                       __private const float height_scale,
                       __private const float width_scale,
                       __private const int input_height,
                       __private const int input_width,
                       __private const int out_height,
                       __private const int out_width) {
    const int output_cw_idx = get_global_id(0);
    const int output_bh_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

    const int output_w_idx       = output_cw_idx % out_width;
    const int output_c_block_idx = output_cw_idx / out_width;
    const int output_b_idx       = output_bh_idx / out_height;
    const int output_h_idx       = output_bh_idx % out_height;

    float scale_height  = ((float)output_h_idx + 0.5f) * height_scale - 0.5f;
    float scale_width   = ((float)output_w_idx + 0.5f) * width_scale - 0.5f;
    scale_height        = max(0.0f, scale_height);
    scale_width         = max(0.0f, scale_width);
    const int height_lf = max(0, (int)floor(scale_height));
    const int height_uf = min(input_height - 1, height_lf + 1);
    const int width_lf  = max(0, (int)floor(scale_width));
    const int width_uf  = min(input_width - 1, width_lf + 1);

    const float height_gap = scale_height - height_lf;
    const float width_gap  = scale_width - width_lf;

    const int input_w_offset = mul24(output_c_block_idx, input_width);
    const int input_h_offset = mul24(output_b_idx, input_height);

    float4 top_left = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_lf, input_h_offset + height_lf));
    float4 top_right = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_uf, input_h_offset + height_lf));
    float4 bottom_left = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_lf, input_h_offset + height_uf));
    float4 bottom_right = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_uf, input_h_offset + height_uf));

    float4 top    = mad((top_right - top_left), width_gap, top_left);
    float4 bottom = mad((bottom_right - bottom_left), width_gap, bottom_left);
    float4 out    = mad((bottom - top), height_gap, top);

    write_imagef(output, (int2)(output_cw_idx, output_bh_idx), out);
}

__kernel void BilinearAlignCorners(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
    __write_only image2d_t output, __private const float height_scale,
    __private const float width_scale, __private const int input_height,
    __private const int input_width, __private const int out_height,
    __private const int out_width) {
    const int output_cw_idx = get_global_id(0);
    const int output_bh_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

    const int output_w_idx       = output_cw_idx % out_width;
    const int output_c_block_idx = output_cw_idx / out_width;
    const int output_b_idx       = output_bh_idx / out_height;
    const int output_h_idx       = output_bh_idx % out_height;

    float scale_height  = (float)output_h_idx * height_scale;
    float scale_width   = (float)output_w_idx * width_scale;
    const int height_lf = max(0, (int)floor(scale_height));
    const int height_uf = min(input_height - 1, height_lf + 1);
    const int width_lf  = max(0, (int)floor(scale_width));
    const int width_uf  = min(input_width - 1, width_lf + 1);

    const float height_gap = scale_height - height_lf;
    const float width_gap  = scale_width - width_lf;

    const int input_w_offset = mul24(output_c_block_idx, input_width);
    const int input_h_offset = mul24(output_b_idx, input_height);

    float4 top_left = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_lf, input_h_offset + height_lf));
    float4 top_right = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_uf, input_h_offset + height_lf));
    float4 bottom_left = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_lf, input_h_offset + height_uf));
    float4 bottom_right = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_uf, input_h_offset + height_uf));

    float4 top    = mad((top_right - top_left), width_gap, top_left);
    float4 bottom = mad((bottom_right - bottom_left), width_gap, bottom_left);
    float4 out    = mad((bottom - top), height_gap, top);

    write_imagef(output, (int2)(output_cw_idx, output_bh_idx), out);
}
__kernel void NearestGS3D(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                          __write_only image2d_t output,
                          __private const float height_scale,
                          __private const float width_scale,
                          __private const int input_height,
                          __private const int input_width,
                          __private const int out_height) {
    const int output_w_idx       = get_global_id(0);
    const int output_c_block_idx = get_global_id(1);
    const int output_bh_idx      = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_w_idx, output_c_block_idx, output_bh_idx);
    const int output_width = global_size_dim0;

    const int output_b_idx = output_bh_idx / out_height;
    const int output_h_idx = output_bh_idx % out_height;

    const float scale_height = output_h_idx * height_scale;
    const float scale_width  = output_w_idx * width_scale;
    const int height_lf      = max(0, (int)floor(scale_height));
    const int width_lf       = max(0, (int)floor(scale_width));

    const int input_w_offset = mul24(output_c_block_idx, input_width);
    const int input_h_offset = mul24(output_b_idx, input_height);

    float4 out = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_lf, input_h_offset + height_lf));

    const int out_image_w =
        mad24(output_c_block_idx, output_width, output_w_idx);
    const int out_image_h = mad24(output_b_idx, out_height, output_h_idx);

    write_imagef(output, (int2)(out_image_w, out_image_h), out);
}

__kernel void BilinearGS3D(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                           __write_only image2d_t output,
                           __private const float height_scale,
                           __private const float width_scale,
                           __private const int input_height,
                           __private const int input_width,
                           __private const int out_height) {
    const int output_w_idx       = get_global_id(0);
    const int output_c_block_idx = get_global_id(1);
    const int output_bh_idx      = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_w_idx, output_c_block_idx, output_bh_idx);
    const int output_width = global_size_dim0;

    const int output_b_idx = output_bh_idx / out_height;
    const int output_h_idx = output_bh_idx % out_height;

    float scale_height  = ((float)output_h_idx + 0.5f) * height_scale - 0.5f;
    float scale_width   = ((float)output_w_idx + 0.5f) * width_scale - 0.5f;
    scale_height        = max(0.0f, scale_height);
    scale_width         = max(0.0f, scale_width);
    const int height_lf = max(0, (int)floor(scale_height));
    const int height_uf = min(input_height - 1, height_lf + 1);
    const int width_lf  = max(0, (int)floor(scale_width));
    const int width_uf  = min(input_width - 1, width_lf + 1);

    const float height_gap = scale_height - height_lf;
    const float width_gap  = scale_width - width_lf;

    const int input_w_offset = mul24(output_c_block_idx, input_width);
    const int input_h_offset = mul24(output_b_idx, input_height);

    float4 top_left = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_lf, input_h_offset + height_lf));
    float4 top_right = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_uf, input_h_offset + height_lf));
    float4 bottom_left = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_lf, input_h_offset + height_uf));
    float4 bottom_right = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_uf, input_h_offset + height_uf));

    float4 top    = mad((top_right - top_left), width_gap, top_left);
    float4 bottom = mad((bottom_right - bottom_left), width_gap, bottom_left);
    float4 out    = mad((bottom - top), height_gap, top);

    const int out_image_w =
        mad24(output_c_block_idx, output_width, output_w_idx);
    const int out_image_h = mad24(output_b_idx, out_height, output_h_idx);

    write_imagef(output, (int2)(out_image_w, out_image_h), out);
}

__kernel void BilinearAlignCornersGS3D(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
    __write_only image2d_t output, __private const float height_scale,
    __private const float width_scale, __private const int input_height,
    __private const int input_width, __private const int out_height) {
    const int output_w_idx       = get_global_id(0);
    const int output_c_block_idx = get_global_id(1);
    const int output_bh_idx      = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_w_idx, output_c_block_idx, output_bh_idx);
    const int output_width = global_size_dim0;

    const int output_b_idx = output_bh_idx / out_height;
    const int output_h_idx = output_bh_idx % out_height;

    float scale_height  = (float)output_h_idx * height_scale;
    float scale_width   = (float)output_w_idx * width_scale;
    const int height_lf = max(0, (int)floor(scale_height));
    const int height_uf = min(input_height - 1, height_lf + 1);
    const int width_lf  = max(0, (int)floor(scale_width));
    const int width_uf  = min(input_width - 1, width_lf + 1);

    const float height_gap = scale_height - height_lf;
    const float width_gap  = scale_width - width_lf;

    const int input_w_offset = mul24(output_c_block_idx, input_width);
    const int input_h_offset = mul24(output_b_idx, input_height);

    float4 top_left = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_lf, input_h_offset + height_lf));
    float4 top_right = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_uf, input_h_offset + height_lf));
    float4 bottom_left = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_lf, input_h_offset + height_uf));
    float4 bottom_right = read_imagef(
        input, SAMPLER,
        (int2)(input_w_offset + width_uf, input_h_offset + height_uf));

    float4 top    = mad((top_right - top_left), width_gap, top_left);
    float4 bottom = mad((bottom_right - bottom_left), width_gap, bottom_left);
    float4 out    = mad((bottom - top), height_gap, top);

    const int out_image_w =
        mad24(output_c_block_idx, output_width, output_w_idx);
    const int out_image_h = mad24(output_b_idx, out_height, output_h_idx);

    write_imagef(output, (int2)(out_image_w, out_image_h), out);
}
