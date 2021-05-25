#include "base.inc"

// convert data from buffer(n d2 d3 d4 c) to image(b d2 d3, ic/4 d4 ic4)
__kernel void NHWCBufferToImage5D(GLOBAL_SIZE_2_DIMS __global const float *input_ptr, /* nhwc */
                                  __private const int channels, __private const int dim2,
                                  __private const int dim3, __private const int dim4,
                                  __write_only image2d_t output) {
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
    __global const float *input_current_ptr = input_ptr + buffer_offset;
    float4 values            = vload4(0, input_current_ptr);
    const int remain_channel = channels - channel_4_idx;
    if (remain_channel == 3) {
        values.w = 0;
    } else if (remain_channel == 2) {
        values.z = 0;
        values.w = 0;
    } else if (remain_channel == 1) {
        values.y = 0;
        values.z = 0;
        values.w = 0;
    }
    write_imagef(output, coord, values);
}

// convert data from buffer(n c d2 d3 d4) to image(b d2 d3, ic/4 d4 ic4)
__kernel void NCHWBufferToImage5D(GLOBAL_SIZE_2_DIMS __global const float *input_ptr, /* nchw */
                                  __private const int channels, __private const int dim2,
                                  __private const int dim3, __private const int dim4,
                                  __write_only image2d_t output) {
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
    float4 output_values    = 0;
    const int remain_channel = channels - channel_4_idx;
    if (remain_channel >= 4) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
        offset += stride;
        output_values.y = *(input_ptr + offset);
        offset += stride;
        output_values.z = *(input_ptr + offset);
        offset += stride;
        output_values.w = *(input_ptr + offset);
    } else if (remain_channel == 3) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
        offset += stride;
        output_values.y = *(input_ptr + offset);
        offset += stride;
        output_values.z = *(input_ptr + offset);
    } else if (remain_channel == 2) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
        offset += stride;
        output_values.y = *(input_ptr + offset);
    } else if (remain_channel == 1) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
    }

    write_imagef(output, (int2)(image_width_idx, image_height_idx), output_values);
}
