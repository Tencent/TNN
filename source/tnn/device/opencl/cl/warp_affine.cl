#include "base.inc"

#define INTER_REMAP_COEF_BITS  15
#define INTER_REMAP_COEF_SCALE (1<<INTER_REMAP_COEF_BITS)
#define INTER_BITS      5
#define INTER_TAB_SIZE  (1<<INTER_BITS)
#define KSIZE 2
#define AB_BITS 10
#define AB_SCALE (1 << AB_BITS)
#define ROUND_DELTA (1 << (AB_BITS - INTER_BITS - 1))

__constant float coeffs[64] =
{ 
    1.000000f, 0.000000f, 0.968750f, 0.031250f, 0.937500f, 0.062500f, 0.906250f, 0.093750f, 0.875000f, 0.125000f, 0.843750f, 0.156250f,
    0.812500f, 0.187500f, 0.781250f, 0.218750f, 0.750000f, 0.250000f, 0.718750f, 0.281250f, 0.687500f, 0.312500f, 0.656250f, 0.343750f,
    0.625000f, 0.375000f, 0.593750f, 0.406250f, 0.562500f, 0.437500f, 0.531250f, 0.468750f, 0.500000f, 0.500000f, 0.468750f, 0.531250f,
    0.437500f, 0.562500f, 0.406250f, 0.593750f, 0.375000f, 0.625000f, 0.343750f, 0.656250f, 0.312500f, 0.687500f, 0.281250f, 0.718750f,
    0.250000f, 0.750000f, 0.218750f, 0.781250f, 0.187500f, 0.812500f, 0.156250f, 0.843750f, 0.125000f, 0.875000f, 0.093750f, 0.906250f,
    0.062500f, 0.937500f, 0.031250f, 0.968750f
};


__kernel void WarpAffineLinear(GLOBAL_SIZE_2_DIMS
                               __read_only image2d_t input,
                               __write_only image2d_t output,
                               __private const int output_height,
                               __private const int output_width,
                               __private const int channel_blocks,
                               __private const int input_height,
                               __private const int input_width,
                               __constant float* m,
                               __private const float border_val
                               ) {
    int output_cw_idx   = get_global_id(0);
    int output_bh_idx   = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

    const int batch_idx             = output_bh_idx / output_height;
    const int height_idx            = output_bh_idx % output_height;
    const int width_idx             = output_cw_idx / channel_blocks;
    const int channel_blocks_idx    = output_cw_idx % channel_blocks;

    int scale_x     = width_idx << AB_BITS;
    int adelta_x    = rint(m[0] * scale_x);
    int adelta_y    = rint(m[3] * scale_x);
    int bdelta_x    = rint(fma(m[1], height_idx, m[2]) * AB_SCALE);
    int bdelta_y    = rint(fma(m[4], height_idx, m[5]) * AB_SCALE);

    int new_x       = adelta_x + bdelta_x + ROUND_DELTA;
    int new_y       = adelta_y + bdelta_y + ROUND_DELTA;
    int new_x_loc   = new_x >> AB_BITS;
    int new_y_loc   = new_y >> AB_BITS;
    short coeffs_x  = convert_short((new_x >> (AB_BITS - INTER_BITS)) & (INTER_TAB_SIZE - 1));
    short coeffs_y  = convert_short((new_y >> (AB_BITS - INTER_BITS)) & (INTER_TAB_SIZE - 1));

    int2 output_pos         = (int2)(output_cw_idx, output_bh_idx);

    int scale_coeffs_x      = coeffs_x << 1, scale_coeffs_y = coeffs_y << 1;
    float tmp_coeffs0       = coeffs[scale_coeffs_y], tmp_coeffs1 = coeffs[scale_coeffs_y + 1];
    float tmp_coeffs2       = coeffs[scale_coeffs_x], tmp_coeffs3 = coeffs[scale_coeffs_x + 1];
    short bilinearWeight0   = convert_short_sat_rte(tmp_coeffs0 * tmp_coeffs2 * INTER_REMAP_COEF_SCALE);
    short bilinearWeight1   = convert_short_sat_rte(tmp_coeffs0 * tmp_coeffs3 * INTER_REMAP_COEF_SCALE);
    short bilinearWeight2   = convert_short_sat_rte(tmp_coeffs1 * tmp_coeffs2 * INTER_REMAP_COEF_SCALE);
    short bilinearWeight3   = convert_short_sat_rte(tmp_coeffs1 * tmp_coeffs3 * INTER_REMAP_COEF_SCALE);
    if (new_x_loc >= 0 && new_x_loc < (input_width - 1) && new_y_loc >= 0 && new_y_loc < (input_height - 1)) {
        const int2 input_pos0 =
            (int2)(mad24(new_x_loc, channel_blocks, channel_blocks_idx),
                   mad24(batch_idx, input_height, new_y_loc));
        const int2 input_pos1 = (int2)(input_pos0.x + channel_blocks, input_pos0.y);
        const int2 input_pos2 = (int2)(input_pos0.x, input_pos0.y + 1);
        const int2 input_pos3 = (int2)(input_pos0.x + channel_blocks, input_pos0.y + 1);

        float4 val0 = read_imagef(input, SAMPLER, input_pos0);
        float4 val1 = read_imagef(input, SAMPLER, input_pos1);
        float4 val2 = read_imagef(input, SAMPLER, input_pos2);
        float4 val3 = read_imagef(input, SAMPLER, input_pos3);

        int4 val = convert_int4_sat(val0) * bilinearWeight0 +
                   convert_int4_sat(val1) * bilinearWeight1 +
                   convert_int4_sat(val2) * bilinearWeight2 +
                   convert_int4_sat(val3) * bilinearWeight3;

        float4 val_out = convert_float4((val + (1 << (INTER_REMAP_COEF_BITS - 1))) >> INTER_REMAP_COEF_BITS);
        write_imagef(output, output_pos, val_out);
    }
    else if (new_x_loc >= -1 && new_x_loc <= (input_width - 1) &&
                new_y_loc >= -1 && new_y_loc <= (input_height - 1)) {
        const int2 input_pos0 =
            (int2)(mad24(new_x_loc, channel_blocks, channel_blocks_idx),
                   mad24(batch_idx, input_height, new_y_loc));
        const int2 input_pos1 = (int2)(input_pos0.x + channel_blocks, input_pos0.y);
        const int2 input_pos2 = (int2)(input_pos0.x, input_pos0.y + 1);
        const int2 input_pos3 = (int2)(input_pos0.x + channel_blocks, input_pos0.y + 1);

        int mask0 = new_x_loc >= 0 && new_y_loc >= 0;
        int mask1 = new_x_loc <= (input_width - 2) && new_y_loc >= 0;
        int mask2 = new_x_loc >= 0 && new_y_loc <= (input_height - 2);
        int mask3 = new_x_loc <= (input_width - 2) && new_y_loc <= (input_height - 2);

        int4 val = 0;
        float4 val0 = border_val, val1 = border_val, val2 = border_val, val3 = border_val;
        if (mask0) {
            val0 = read_imagef(input, SAMPLER, input_pos0);
        }
        val += convert_int4_sat(val0) * bilinearWeight0;
        if (mask1) {
            val1 = read_imagef(input, SAMPLER, input_pos1);
        }
        val += convert_int4_sat(val1) * bilinearWeight1;
        if (mask2) {
            val2 = read_imagef(input, SAMPLER, input_pos2);
        }
        val += convert_int4_sat(val2) * bilinearWeight2;
        if (mask3) {
            val3 = read_imagef(input, SAMPLER, input_pos3);
        }
        val += convert_int4_sat(val3) * bilinearWeight3;

        float4 val_out = convert_float4((val + (1 << (INTER_REMAP_COEF_BITS - 1))) >> INTER_REMAP_COEF_BITS);
        write_imagef(output, output_pos, val_out);
    }
    else
    {
        float4 val_out = border_val;
        write_imagef(output, output_pos, val_out);
    }
}

__kernel void WarpAffineNearest(GLOBAL_SIZE_2_DIMS
                                __read_only image2d_t input,
                                __write_only image2d_t output,
                                __private const int output_height,
                                __private const int output_width,
                                __private const int channel_blocks,
                                __private const int input_height,
                                __private const int input_width,
                                __constant float* m,
                                __private const float border_val
                                ) {
    int output_cw_idx   = get_global_id(0);
    int output_bh_idx   = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

    const int batch_idx             = output_bh_idx / output_height;
    const int height_idx            = output_bh_idx % output_height;
    const int width_idx             = output_cw_idx / channel_blocks;
    const int channel_blocks_idx    = output_cw_idx % channel_blocks;

    int scale_x     = width_idx << AB_BITS;
    int adelta_x    = rint(m[0] * scale_x);
    int adelta_y    = rint(m[3] * scale_x);
    int bdelta_x    = rint(fma(m[1], height_idx, m[2]) * AB_SCALE);
    int bdelta_y    = rint(fma(m[4], height_idx, m[5]) * AB_SCALE);

    int new_x       = adelta_x + bdelta_x + ROUND_DELTA;
    int new_y       = adelta_y + bdelta_y + ROUND_DELTA;
    int new_x_loc   = new_x >> AB_BITS;
    int new_y_loc   = new_y >> AB_BITS;
    short coeffs_x  = convert_short((new_x >> (AB_BITS - INTER_BITS)) & (INTER_TAB_SIZE - 1));
    short coeffs_y  = convert_short((new_y >> (AB_BITS - INTER_BITS)) & (INTER_TAB_SIZE - 1));
    int is_right    = coeffs_x >= 16;
    int is_bottom   = coeffs_y >= 16;

    int2 output_pos         = (int2)(output_cw_idx, output_bh_idx);
    if (new_x_loc >= 0 && new_x_loc < (input_width - 1) && new_y_loc >= 0 && new_y_loc < (input_height - 1)) {
        const int2 input_pos =
            (int2)(mad24(new_x_loc + is_right, channel_blocks, channel_blocks_idx),
                   mad24(batch_idx, input_height, new_y_loc + is_bottom));
        write_imagef(output, output_pos, read_imagef(input, SAMPLER, input_pos));
    }
    else if (new_x_loc >= -1 && new_x_loc <= (input_width - 1) &&
                new_y_loc >= -1 && new_y_loc <= (input_height - 1)) {
        int mask = select(new_x_loc >= 0, new_x_loc <= (input_width - 2), is_right) &&
                   select(new_y_loc >= 0, new_y_loc <= (input_height - 2), is_bottom);
        const int2 input_pos =
            (int2)(mad24(new_x_loc + is_right, channel_blocks, channel_blocks_idx),
                   mad24(batch_idx, input_height, new_y_loc + is_bottom));
        float4 val_out = border_val;
        if (mask) {
            val_out = read_imagef(input, SAMPLER, input_pos);
        }
        write_imagef(output, output_pos, val_out);
    }
    else
    {
        float4 val_out = border_val;
        write_imagef(output, output_pos, val_out);
    }
}