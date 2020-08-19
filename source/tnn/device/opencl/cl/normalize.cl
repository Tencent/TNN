#include "base.inc"
__kernel void NormalizeCommon0(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                                 __private const int channel_blocks,
                                 __private const int width,
                                 __private const float eps,
                                 __write_only image2d_t output) {
    const int w_idx  = get_global_id(0);
    const int bh_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(w_idx, bh_idx);

    FLOAT4 sum_xyzw = (FLOAT4)(0.0f, 0.0f, 0.0f, 0.0f);
    int cw_idx      = w_idx;
    for (int cb = 0; cb < channel_blocks; ++cb) {
        FLOAT4 val = RI_F(input, SAMPLER, (int2)(cw_idx, bh_idx));
#ifdef NORMALIZE_P2
        sum_xyzw += val * val;
#else
        sum_xyzw += fabs(val);
#endif
        cw_idx += width;
    }
    FLOAT sum = sum_xyzw.x + sum_xyzw.y + sum_xyzw.z + sum_xyzw.w;
#ifdef NORMALIZE_P2
    sum = max(sqrt(sum), (FLOAT)(eps));
#endif

    for (int cb = 0; cb < channel_blocks; ++cb) {
        int2 pos   = (int2)(mad24(cb, width, w_idx), bh_idx);
        FLOAT4 val = RI_F(input, SAMPLER, pos) / sum;
        WI_F(output, pos, val);
    }
}

__kernel void NormalizeCommon(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                               __private const int channel_blocks,
                               __private const int channel_remain,
                               __private const int width,
                               __private const float eps,
                               __write_only image2d_t output) {
    const int w_idx  = get_global_id(0);
    const int bh_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(w_idx, bh_idx);

    FLOAT4 sum_xyzw = (FLOAT4)(0.0f, 0.0f, 0.0f, 0.0f);
    int cw_idx      = w_idx;
    for (int cb = 0; cb < (channel_blocks - 1); ++cb) {
        FLOAT4 val = RI_F(input, SAMPLER, (int2)(cw_idx, bh_idx));
#ifdef NORMALIZE_P2
        sum_xyzw += val * val;
#else
        sum_xyzw += fabs(val);
#endif
        cw_idx += width;
    }
    FLOAT sum = sum_xyzw.x + sum_xyzw.y + sum_xyzw.z + sum_xyzw.w;

    FLOAT4 val_last = RI_F(input, SAMPLER, (int2)(cw_idx, bh_idx));
#ifdef NORMALIZE_P2
    val_last = val_last * val_last;
#else
    val_last = fabs(val_last);
#endif
    if (1 == channel_remain) {
        sum += val_last.x;
    } else if (2 == channel_remain) {
        sum += val_last.x;
        sum += val_last.y;
    } else if (3 == channel_remain) {
        sum += val_last.x;
        sum += val_last.y;
        sum += val_last.z;
    } else {
        sum += val_last.x;
        sum += val_last.y;
        sum += val_last.z;
        sum += val_last.w;
    }

#ifdef NORMALIZE_P2
    sum = max(sqrt(sum), (FLOAT)(eps));
#endif

    for (int cb = 0; cb < channel_blocks; ++cb) {
        int2 pos   = (int2)(mad24(cb, width, w_idx), bh_idx);
        FLOAT4 val = RI_F(input, SAMPLER, pos) / sum;
        WI_F(output, pos, val);
    }
}

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__kernel void image_scaling(GLOBAL_SIZE_2_DIMS __read_only image2d_t sourceImage,
                            __write_only image2d_t destinationImage,
                            float widthNormalizationFactor,
                            float heightNormalizationFactor)
{
    const int w_idx  = get_global_id(0);
    const int bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(w_idx, bh_idx);
    int2 coordinate = (int2)(w_idx, bh_idx);
    float2 normalizedCoordinate = convert_float2(coordinate) *  (float2)(widthNormalizationFactor, heightNormalizationFactor);
    float4 colour = read_imagef(sourceImage, sampler, normalizedCoordinate);
    write_imagef(destinationImage, coordinate, colour);
}

__kernel void image_bilinear(GLOBAL_SIZE_2_DIMS __read_only image2d_t sourceImage,
                            __write_only image2d_t destinationImage,
                            float widthNormalizationFactor,
                            float heightNormalizationFactor,
                            int src_width,
                            int src_height,
                            int dst_width,
                            int dst_height)
{
    const int w_idx  = get_global_id(0);
    const int bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(w_idx, bh_idx);
    int2 coordinate = (int2)(w_idx, bh_idx);
    float scale_w = src_width / dst_width;
    float scale_h = src_height / dst_height;

    // float pos_x = ((float)(w_idx) + 0.5) * scale_w - 0.5;
    // float pos_y = ((float)(bh_idx) + 0.5) * scale_h - 0.5;

    float2 pos_offset = (float2)(0.5f);
    float2 coordinate_pos = convert_float2(coordinate) + pos_offset;
    //float2 normalizedCoordinate_pos = coordinate_pos *  (float2)(widthNormalizationFactor, heightNormalizationFactor);
    float2 normalizedCoordinate_pos = coordinate_pos *  (float2)(scale_w, scale_h);
    float2 normalizedCoordinate = normalizedCoordinate_pos - pos_offset;
    float n,m;
    int x = (int)floor(normalizedCoordinate.x);
    int y = (int)floor(normalizedCoordinate.y);
    m = normalizedCoordinate.x - x;
    n = normalizedCoordinate.y - y;
    printf("x,y: %d, %d", x,y);
    // int x = floor(pos_x);
    // int y = floor(pos_y);
    // m = pos_x - x;
    // n = pos_y - y;
    if (x < 0) {
        x = 0;
        m = 0.f;
    }
    if (x >= src_width - 1) {
        x = src_width - 2;
        m = 1.f;
    }
    if (y < 0) {
        y = 0;
        n = 0.f;
    }
    if (y >= src_height - 1) {
        y = src_height - 2;
        n = 1.f;
    }
    
    float4 colour_upleft = read_imagef(sourceImage, SAMPLER, (int2)(x,y));
    float4 colour_downleft = read_imagef(sourceImage, SAMPLER, (int2)(x,y+1));
    float4 colour_upright = read_imagef(sourceImage, SAMPLER, (int2)(x+1,y));
    float4 colour_downright = read_imagef(sourceImage, SAMPLER, (int2)(x+1,y+1));
    float4 colour = (1 -m)*(1-n)*colour_upleft + 
                    (1-m)*n*colour_downleft + 
                    m*n*colour_downright + 
                    m*(1-n)*colour_upright;

    float x_ef0_ = (1 - m) * 2048;
    float x_ef1_ = m * 2048;
    
    float y_ef0_ = (1 - n) * 2048;
    float y_ef1_ = n * 2048;
    
    float4 col0 = (colour_upleft * x_ef0_ + colour_upright * x_ef1_) / 16;
    float4 col1 = (colour_downleft * x_ef0_ + colour_downright * x_ef1_) / 16;
    
    float4 value = ((col0 * y_ef0_)/65546 + (col1 * y_ef1_)/65536 + 2) / 4;

    write_imagef(destinationImage, coordinate, value);
}