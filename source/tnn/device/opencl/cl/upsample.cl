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

static inline float4 UpsampleCubicInterpolation(float4 A, float4 B, float4 C, float4 D, float factor) {
    // refer to opencv
    const float w = -0.75f;

    float coeffs[4];
    const float factor_plus_1 = factor + 1;
    const float factor_remain = 1 - factor;
    coeffs[0] = mad(mad(w, factor_plus_1, 3.75f), factor_plus_1, -6.0f) * factor_plus_1 + 3.0f;
    coeffs[1] = mad(1.25f, factor, -2.25f) * factor * factor + 1;
    coeffs[2] = mad(1.25f, factor_remain, - 2.25f) * factor_remain * factor_remain + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];

    return A * coeffs[0] + B * coeffs[1] + C * coeffs[2] + D * coeffs[3];
}

#define CUBIC_READ_INPUT(i, scale_h_pos) \
    float4 A_h##i = read_imagef(input, SAMPLER, (int2)(scale_w_pos.x, scale_h_pos)); \
    float4 B_h##i = read_imagef(input, SAMPLER, (int2)(scale_w_pos.y, scale_h_pos)); \
    float4 C_h##i = read_imagef(input, SAMPLER, (int2)(scale_w_pos.z, scale_h_pos)); \
    float4 D_h##i = read_imagef(input, SAMPLER, (int2)(scale_w_pos.w, scale_h_pos));

__kernel void Cubic(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
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

    const int input_w_offset = mul24(output_c_block_idx, input_width);
    const int input_h_offset = mul24(output_b_idx, input_height);

    float scale_height  = ((float)output_h_idx + 0.5f) * height_scale - 0.5f;
    float scale_width   = ((float)output_w_idx + 0.5f) * width_scale - 0.5f;
    int scale_hh        = floor(scale_height);
    int scale_ww        = floor(scale_width);

    const float h_factor = scale_height - scale_hh;
    const float w_factor = scale_width - scale_ww;

    int4 scale_h_pos    = {scale_hh - 1, scale_hh, scale_hh + 1, scale_hh + 2};
    scale_h_pos = clamp(scale_h_pos, (int4)0, (int4)(input_height - 1));
    scale_h_pos += input_h_offset;

    int4 scale_w_pos    = {scale_ww - 1, scale_ww, scale_ww + 1, scale_ww + 2};
    scale_w_pos = clamp(scale_w_pos, (int4)0, (int4)(input_width - 1));
    scale_w_pos += input_w_offset;

    float4 out_h0, out_h1, out_h2, out_h3, out;

    CUBIC_READ_INPUT(0, scale_h_pos.x);
    CUBIC_READ_INPUT(1, scale_h_pos.y);
    CUBIC_READ_INPUT(2, scale_h_pos.z);
    CUBIC_READ_INPUT(3, scale_h_pos.w);

    out_h0 = UpsampleCubicInterpolation(A_h0, B_h0, C_h0, D_h0, w_factor);
    out_h1 = UpsampleCubicInterpolation(A_h1, B_h1, C_h1, D_h1, w_factor);
    out_h2 = UpsampleCubicInterpolation(A_h2, B_h2, C_h2, D_h2, w_factor);
    out_h3 = UpsampleCubicInterpolation(A_h3, B_h3, C_h3, D_h3, w_factor);

    out = UpsampleCubicInterpolation(out_h0, out_h1, out_h2, out_h3, h_factor);

    write_imagef(output, (int2)(output_cw_idx, output_bh_idx), out);
}

__kernel void CubicAlignCorners(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
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

    const int input_w_offset = mul24(output_c_block_idx, input_width);
    const int input_h_offset = mul24(output_b_idx, input_height);

    float scale_height  = (float)output_h_idx * height_scale;
    float scale_width   = (float)output_w_idx * width_scale;
    int scale_hh        = floor(scale_height);
    int scale_ww        = floor(scale_width);

    const float h_factor = scale_height - scale_hh;
    const float w_factor = scale_width - scale_ww;

    int4 scale_h_pos    = {scale_hh - 1, scale_hh, scale_hh + 1, scale_hh + 2};
    scale_h_pos = clamp(scale_h_pos, (int4)0, (int4)(input_height - 1));
    scale_h_pos += input_h_offset;

    int4 scale_w_pos    = {scale_ww - 1, scale_ww, scale_ww + 1, scale_ww + 2};
    scale_w_pos = clamp(scale_w_pos, (int4)0, (int4)(input_width - 1));
    scale_w_pos += input_w_offset;

    float4 out_h0, out_h1, out_h2, out_h3, out;

    CUBIC_READ_INPUT(0, scale_h_pos.x);
    CUBIC_READ_INPUT(1, scale_h_pos.y);
    CUBIC_READ_INPUT(2, scale_h_pos.z);
    CUBIC_READ_INPUT(3, scale_h_pos.w);

    out_h0 = UpsampleCubicInterpolation(A_h0, B_h0, C_h0, D_h0, w_factor);
    out_h1 = UpsampleCubicInterpolation(A_h1, B_h1, C_h1, D_h1, w_factor);
    out_h2 = UpsampleCubicInterpolation(A_h2, B_h2, C_h2, D_h2, w_factor);
    out_h3 = UpsampleCubicInterpolation(A_h3, B_h3, C_h3, D_h3, w_factor);

    out = UpsampleCubicInterpolation(out_h0, out_h1, out_h2, out_h3, h_factor);

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

__kernel void CubicGS3D(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
    __write_only image2d_t output, __private const float height_scale,
    __private const float width_scale, __private const int input_height,
    __private const int input_width, __private const int out_height) {
    const int output_w_idx       = get_global_id(0);
    const int output_c_block_idx = get_global_id(1);
    const int output_bh_idx      = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_w_idx, output_c_block_idx, output_bh_idx);
    const int output_width = global_size_dim0;

    const int output_b_idx       = output_bh_idx / out_height;
    const int output_h_idx       = output_bh_idx % out_height;

    const int input_w_offset = mul24(output_c_block_idx, input_width);
    const int input_h_offset = mul24(output_b_idx, input_height);

    float scale_height  = ((float)output_h_idx + 0.5f) * height_scale - 0.5f;
    float scale_width   = ((float)output_w_idx + 0.5f) * width_scale - 0.5f;
    int scale_hh        = floor(scale_height);
    int scale_ww        = floor(scale_width);

    const float h_factor = scale_height - scale_hh;
    const float w_factor = scale_width - scale_ww;

    int4 scale_h_pos    = {scale_hh - 1, scale_hh, scale_hh + 1, scale_hh + 2};
    scale_h_pos = clamp(scale_h_pos, (int4)0, (int4)(input_height - 1));
    scale_h_pos += input_h_offset;

    int4 scale_w_pos    = {scale_ww - 1, scale_ww, scale_ww + 1, scale_ww + 2};
    scale_w_pos = clamp(scale_w_pos, (int4)0, (int4)(input_width - 1));
    scale_w_pos += input_w_offset;

    float4 out_h0, out_h1, out_h2, out_h3, out;

    CUBIC_READ_INPUT(0, scale_h_pos.x);
    CUBIC_READ_INPUT(1, scale_h_pos.y);
    CUBIC_READ_INPUT(2, scale_h_pos.z);
    CUBIC_READ_INPUT(3, scale_h_pos.w);

    out_h0 = UpsampleCubicInterpolation(A_h0, B_h0, C_h0, D_h0, w_factor);
    out_h1 = UpsampleCubicInterpolation(A_h1, B_h1, C_h1, D_h1, w_factor);
    out_h2 = UpsampleCubicInterpolation(A_h2, B_h2, C_h2, D_h2, w_factor);
    out_h3 = UpsampleCubicInterpolation(A_h3, B_h3, C_h3, D_h3, w_factor);

    out = UpsampleCubicInterpolation(out_h0, out_h1, out_h2, out_h3, h_factor);

    const int out_image_w =
        mad24(output_c_block_idx, output_width, output_w_idx);
    write_imagef(output, (int2)(out_image_w, output_bh_idx), out);
}

__kernel void CubicAlignCornersGS3D(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
    __write_only image2d_t output, __private const float height_scale,
    __private const float width_scale, __private const int input_height,
    __private const int input_width, __private const int out_height) {
    const int output_w_idx       = get_global_id(0);
    const int output_c_block_idx = get_global_id(1);
    const int output_bh_idx      = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_w_idx, output_c_block_idx, output_bh_idx);
    const int output_width = global_size_dim0;

    const int output_b_idx       = output_bh_idx / out_height;
    const int output_h_idx       = output_bh_idx % out_height;

    const int input_w_offset = mul24(output_c_block_idx, input_width);
    const int input_h_offset = mul24(output_b_idx, input_height);

    float scale_height  = (float)output_h_idx * height_scale;
    float scale_width   = (float)output_w_idx * width_scale;
    int scale_hh        = floor(scale_height);
    int scale_ww        = floor(scale_width);

    const float h_factor = scale_height - scale_hh;
    const float w_factor = scale_width - scale_ww;

    int4 scale_h_pos    = {scale_hh - 1, scale_hh, scale_hh + 1, scale_hh + 2};
    scale_h_pos = clamp(scale_h_pos, (int4)0, (int4)(input_height - 1));
    scale_h_pos += input_h_offset;

    int4 scale_w_pos    = {scale_ww - 1, scale_ww, scale_ww + 1, scale_ww + 2};
    scale_w_pos = clamp(scale_w_pos, (int4)0, (int4)(input_width - 1));
    scale_w_pos += input_w_offset;

    float4 out_h0, out_h1, out_h2, out_h3, out;

    CUBIC_READ_INPUT(0, scale_h_pos.x);
    CUBIC_READ_INPUT(1, scale_h_pos.y);
    CUBIC_READ_INPUT(2, scale_h_pos.z);
    CUBIC_READ_INPUT(3, scale_h_pos.w);

    out_h0 = UpsampleCubicInterpolation(A_h0, B_h0, C_h0, D_h0, w_factor);
    out_h1 = UpsampleCubicInterpolation(A_h1, B_h1, C_h1, D_h1, w_factor);
    out_h2 = UpsampleCubicInterpolation(A_h2, B_h2, C_h2, D_h2, w_factor);
    out_h3 = UpsampleCubicInterpolation(A_h3, B_h3, C_h3, D_h3, w_factor);

    out = UpsampleCubicInterpolation(out_h0, out_h1, out_h2, out_h3, h_factor);

    const int out_image_w =
        mad24(output_c_block_idx, output_width, output_w_idx);
    write_imagef(output, (int2)(out_image_w, output_bh_idx), out);
}