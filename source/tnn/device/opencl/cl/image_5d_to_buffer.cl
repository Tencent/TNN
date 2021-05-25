#include "base.inc"

// convert data from image(b d2 d3, ic/4 d4 ic4) to buffer(n d2 d3 d4 c)
__kernel void Image5DToNHWCBuffer(GLOBAL_SIZE_2_DIMS __global float *output, /* nhwc */
                                  __private const int channels, __private const int dim2,
                                  __private const int dim3, __private const int dim4,
                                  __read_only image2d_t input_ptr) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int dim3_idx       = image_height_idx % dim3;
    const int batch_dim2_idx = image_height_idx / dim3;
    const int dim2_idx       = batch_dim2_idx % dim2;
    const int batch_idx      = batch_dim2_idx / dim2;
    const int dim4_idx       = image_width_idx % dim4;
    const int channel_4_idx  = (image_width_idx / dim4) << 2;
    const int buffer_offset  = (((batch_idx * dim2 + dim2_idx) * dim3 + dim3_idx) * dim4 +
                                dim4_idx) * channels + channel_4_idx;

    int2 coord               = (int2)(image_width_idx, image_height_idx);
    float4 values            = read_imagef(input_ptr, SAMPLER, coord);
    const int remain_channel = channels - channel_4_idx;
    if (remain_channel >= 4) {
        vstore4(values, 0, output + buffer_offset);
    } else if (remain_channel == 3) {
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset++;
        output[offset] = values.y;
        offset++;
        output[offset] = values.z;
    } else if (remain_channel == 2) {
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset++;
        output[offset] = values.y;
    } else if (remain_channel == 1) {
        int offset     = buffer_offset;
        output[offset] = values.x;
    }
}

// convert data from image(b d2 d3, ic/4 d4 ic4) to buffer(n c d2 d3 d4)
__kernel void Image5DToNCHWBuffer(GLOBAL_SIZE_2_DIMS __global float *output, /* nhwc */
                                  __private const int channels, __private const int dim2,
                                  __private const int dim3, __private const int dim4,
                                  __read_only image2d_t input_ptr) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int dim3_idx       = image_height_idx % dim3;
    const int batch_dim2_idx = image_height_idx / dim3;
    const int dim2_idx       = batch_dim2_idx % dim2;
    const int batch_idx      = batch_dim2_idx / dim2;
    const int dim4_idx       = image_width_idx % dim4;
    const int channel_4_idx  = (image_width_idx / dim4) << 2;
    const int buffer_offset  = (((batch_idx * channels + channel_4_idx) * dim2 + dim2_idx) *
                                dim3 + dim3_idx) * dim4 + dim4_idx;

    const int stride         = dim2 * dim3 * dim4;
    int2 coord               = (int2)(image_width_idx, image_height_idx);
    #ifdef ENABLE_BUFFER_PRECISION_ADJUST
    __global FLOAT *output_ptr = (__global FLOAT *)output;
    FLOAT4 values    = RI_F(input_ptr, SAMPLER, (int2)(image_width_idx, image_height_idx));
    #else
    __global float *output_ptr = output;
    float4 values    = read_imagef(input_ptr, SAMPLER, (int2)(image_width_idx, image_height_idx));
    #endif
    const int remain_channel = channels - channel_4_idx;
    if (remain_channel >= 4) {
        int offset     = buffer_offset;
        output_ptr[offset] = values.x;
        offset += stride;
        output_ptr[offset] = values.y;
        offset += stride;
        output_ptr[offset] = values.z;
        offset += stride;
        output_ptr[offset] = values.w;
    } else if (remain_channel == 3) {
        int offset     = buffer_offset;
        output_ptr[offset] = values.x;
        offset += stride;
        output_ptr[offset] = values.y;
        offset += stride;
        output_ptr[offset] = values.z;
    } else if (remain_channel == 2) {
        int offset     = buffer_offset;
        output_ptr[offset] = values.x;
        offset += stride;
        output_ptr[offset] = values.y;
    } else if (remain_channel == 1) {
        int offset     = buffer_offset;
        output_ptr[offset] = values.x;
    }
}
