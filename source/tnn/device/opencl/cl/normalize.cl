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

__kernel void ResizeBilinear(GLOBAL_SIZE_2_DIMS __read_only image2d_t sourceImage,
                            __write_only image2d_t destinationImage,
                            float widthNormalizationFactor,
                            float heightNormalizationFactor,
                            int src_width,
                            int src_height,
                            int dst_width,
                            int dst_height){
    int cw_idx  = get_global_id(0);
    int bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw_idx, bh_idx);

    const int batch_idx         = bh_idx / dst_height;
    int2 coordinate = (int2)(cw_idx, bh_idx);

    float pos_x = (cw_idx + 0.5f) * widthNormalizationFactor - 0.5f;
    float pos_y = (bh_idx + 0.5f) * heightNormalizationFactor - 0.5f;

    float rat_x,rat_y;
    int x = floor(pos_x);
    int y = floor(pos_y);

    rat_x = pos_x - x;
    rat_y = pos_y - y;
    if (x < 0) {
        x = 0;
        rat_x = 0.f;
    }
    if (x >= src_width - 1) {
        x = src_width - 2;
        rat_x = 1.f;
    }
    if (y < batch_idx*src_height) {
        y = batch_idx*src_height;
        rat_y = 0.f;
    }
    if (y >= (batch_idx+1)*src_height - 1) {
        y = (batch_idx+1)*src_height - 2;
        rat_y = 1.f;
    }

    float4 colour_upleft = read_imagef(sourceImage, SAMPLER, (int2)(x,y));
    float4 colour_downleft = read_imagef(sourceImage, SAMPLER, (int2)(x,y+1));
    float4 colour_upright = read_imagef(sourceImage, SAMPLER, (int2)(x+1,y));
    float4 colour_downright = read_imagef(sourceImage, SAMPLER, (int2)(x+1,y+1));
    float4 colour = (1 - rat_x)*(1 - rat_y)*colour_upleft + 
                    (1 - rat_x)*rat_y*colour_downleft + 
                    rat_x*rat_y*colour_downright + 
                    rat_x*(1 - rat_y)*colour_upright;

    write_imagef(destinationImage, coordinate, colour);
}
__kernel void ResizeNearest(GLOBAL_SIZE_2_DIMS __read_only image2d_t sourceImage,
                            __write_only image2d_t destinationImage,
                            float widthNormalizationFactor,
                            float heightNormalizationFactor,
                            int src_width,
                            int src_height,
                            int dst_width,
                            int dst_height){
    int cw_idx  = get_global_id(0);
    int bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(cw_idx, bh_idx);

    const int batch_idx         = bh_idx / dst_height;
    int2 coordinate = (int2)(cw_idx, bh_idx);

    float pos_x = (cw_idx + 0.5f) * widthNormalizationFactor - 0.5f;
    float pos_y = (bh_idx + 0.5f) * heightNormalizationFactor - 0.5f;

    float rat_x,rat_y;
    int x = floor(pos_x);
    int y = floor(pos_y);

    rat_x = pos_x - x;
    rat_y = pos_y - y;
    if (x < 0) {
        x = 0;
        rat_x = 0.f;
    }
    if (x >= src_width - 1) {
        x = src_width - 2;
        rat_x = 1.f;
    }
    if (y < batch_idx*src_height) {
        y = batch_idx*src_height;
        rat_y = 0.f;
    }
    if (y >= (batch_idx+1)*src_height - 1) {
        y = (batch_idx+1)*src_height - 2;
        rat_y = 1.f;
    }
    if(rat_x > 0.5f)
        x = x + 1;
    if(rat_y > 0.5f)
        y = y + 1;

    float4 colour = read_imagef(sourceImage, SAMPLER, (int2)(x,y));
    write_imagef(destinationImage, coordinate, colour);
}