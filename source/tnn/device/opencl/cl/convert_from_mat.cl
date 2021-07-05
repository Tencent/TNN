#include "base.inc"

__kernel void ConvertFromNCHW(GLOBAL_SIZE_2_DIMS __write_only image2d_t output,
                              __global const float *input_ptr,
                              __private const int height,
                              __private const int width,
                              __private const int channels,
                              __global const float* scale,
                              __global const float* bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = (image_width_idx / width) << 2;
    const int buffer_offset =
        ((batch_idx * channels + channel_4_idx) * height + height_idx) * width +
        width_idx;

    const int remain_channel    = channels - channel_4_idx;
    const int height_width_size = height * width;
    float4 output_values        = 0;

    if (remain_channel >= 4) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
        offset += height_width_size;
        output_values.y = *(input_ptr + offset);
        offset += height_width_size;
        output_values.z = *(input_ptr + offset);
        offset += height_width_size;
        output_values.w = *(input_ptr + offset);
#ifdef ENABLE_SCALE_BIAS
        float4 scale_data   = vload4(0, scale + channel_4_idx);
        float4 bias_data    = vload4(0, bias + channel_4_idx);
        output_values = output_values * scale_data + bias_data;
#endif
    } else if (remain_channel == 3) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
        offset += height_width_size;
        output_values.y = *(input_ptr + offset);
        offset += height_width_size;
        output_values.z = *(input_ptr + offset);
#ifdef ENABLE_SCALE_BIAS
        float3 scale_data   = vload3(0, scale + channel_4_idx);
        float3 bias_data    = vload3(0, bias + channel_4_idx);
        output_values.xyz = output_values.xyz * scale_data + bias_data;
#endif
    } else if (remain_channel == 2) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
        offset += height_width_size;
        output_values.y = *(input_ptr + offset);
#ifdef ENABLE_SCALE_BIAS
        float2 scale_data   = vload2(0, scale + channel_4_idx);
        float2 bias_data    = vload2(0, bias + channel_4_idx);
        output_values.xy = output_values.xy * scale_data + bias_data;
#endif
    } else if (remain_channel == 1) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
#ifdef ENABLE_SCALE_BIAS
        output_values.x = output_values.x * scale[channel_4_idx] + bias[channel_4_idx];
#endif
    }

    write_imagef(output, (int2)(image_width_idx, image_height_idx),
                 output_values);
}

__kernel void IntBlobConvertFromNCINT32(GLOBAL_SIZE_2_DIMS __write_only image2d_t output,
                                        __global const int *input_ptr,
                                        __private const int height,
                                        __private const int width,
                                        __private const int channels,
                                        __global const float* scale,
                                        __global const float* bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = (image_width_idx / width) << 2;
    const int buffer_offset =
        ((batch_idx * channels + channel_4_idx) * height + height_idx) * width +
        width_idx;

    const int remain_channel    = channels - channel_4_idx;
    const int height_width_size = height * width;
    int4 output_values        = 0;

    if (remain_channel >= 4) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
        offset += height_width_size;
        output_values.y = *(input_ptr + offset);
        offset += height_width_size;
        output_values.z = *(input_ptr + offset);
        offset += height_width_size;
        output_values.w = *(input_ptr + offset);
    } else if (remain_channel == 3) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
        offset += height_width_size;
        output_values.y = *(input_ptr + offset);
        offset += height_width_size;
        output_values.z = *(input_ptr + offset);
    } else if (remain_channel == 2) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
        offset += height_width_size;
        output_values.y = *(input_ptr + offset);
    } else if (remain_channel == 1) {
        int offset      = buffer_offset;
        output_values.x = *(input_ptr + offset);
    }

    write_imagei(output, (int2)(image_width_idx, image_height_idx),
                 output_values);
}

__kernel void CNH4BlobConvertFromNCHW(GLOBAL_SIZE_2_DIMS __write_only image2d_t output,
                                      __global const float *input_ptr,
                                      __private const int height,
                                      __private const int batch,
                                      __private const int channels,
                                      __global const float* scale,
                                      __global const float* bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx % batch;
    const int channel_idx   = image_height_idx / batch;
    const int height_4_idx  = image_width_idx << 2;
    int buffer_offset =
        (batch_idx * channels + channel_idx) * height + height_4_idx;

    const int remain_height = height - height_4_idx;
    float4 output_values        = 0;
    int offset      = buffer_offset;
    if (remain_height >= 4) {
        output_values = vload4(0, input_ptr + offset);
    } else if (remain_height == 3) {
        output_values.xyz = vload3(0, input_ptr + offset);
    } else if (remain_height == 2) {
        output_values.xy = vload2(0, input_ptr + offset);
    } else if (remain_height == 1) {
        output_values.x = input_ptr[offset];
    }

    write_imagef(output, (int2)(image_width_idx, image_height_idx),
                 output_values);
}

__kernel void ConvertFromN8UC4(GLOBAL_SIZE_2_DIMS __write_only image2d_t output,
                               __global const uchar *input_ptr,
                               __private const int height,
                               __private const int width,
                               __private const int channels,
                               __private const float4 scale,
                               __private const float4 bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx  = image_height_idx / height;
    const int height_idx = image_height_idx % height;

    int buffer_offset =
        ((batch_idx * height + height_idx) * width + image_width_idx) * 4;

    float4 values = convert_float4(vload4(0, input_ptr + buffer_offset));

    if (channels == 3) {
        values.w = 0;
    }

#ifdef SWAP_RB
    float temp = values.x;
    values.x   = values.z;
    values.z   = temp;
#endif

#ifdef ENABLE_SCALE_BIAS
    values = values * scale + bias;
#endif

    int2 coord = (int2)(image_width_idx, image_height_idx);
    write_imagef(output, coord, values);
}

__kernel void ConvertFromN8UC3(GLOBAL_SIZE_2_DIMS __write_only image2d_t output,
                               __global const uchar *input_ptr,
                               __private const int height,
                               __private const int width,
                               __private const float4 scale,
                               __private const float4 bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx  = image_height_idx / height;
    const int height_idx = image_height_idx % height;

    int buffer_offset =
        ((batch_idx * height + height_idx) * width + image_width_idx) * 3;

    float4 values = (float4)(0.0f);

#ifdef SWAP_RB
    values.z = convert_float(input_ptr[buffer_offset]);
    values.y = convert_float(input_ptr[buffer_offset + 1]);
    values.x = convert_float(input_ptr[buffer_offset + 2]);
#else
    values.x = convert_float(input_ptr[buffer_offset]);
    values.y = convert_float(input_ptr[buffer_offset + 1]);
    values.z = convert_float(input_ptr[buffer_offset + 2]);
#endif

#ifdef ENABLE_SCALE_BIAS
    values = values * scale + bias;
#endif

    int2 coord = (int2)(image_width_idx, image_height_idx);
    write_imagef(output, coord, values);
}

__kernel void ConvertFromNGray(GLOBAL_SIZE_2_DIMS __write_only image2d_t output,
                               __global const uchar *input_ptr,
                               __private const int height,
                               __private const int width,
                               __private const float4 scale,
                               __private const float4 bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx  = image_height_idx / height;
    const int height_idx = image_height_idx % height;

    int buffer_offset =
        (batch_idx * height + height_idx) * width + image_width_idx;

    float4 values = 0;
    values.x      = convert_float(input_ptr[buffer_offset]);

#ifdef ENABLE_SCALE_BIAS
    values = values * scale + bias;
#endif

    int2 coord = (int2)(image_width_idx, image_height_idx);
    write_imagef(output, coord, values);
}

__kernel void ConvertFromNNV21(GLOBAL_SIZE_2_DIMS __write_only image2d_t output,
                               __global const uchar *input_ptr,
                               __private const int height,
                               __private const int width,
                               __private const float4 scale,
                               __private const float4 bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    int y_offset = image_height_idx * width + image_width_idx;
    int v_offset = width * height + (image_height_idx >> 1) * width +
                   (image_width_idx & (~(0x01)));
    int u_offset = v_offset + 1;

    int y = (int)(input_ptr[y_offset]);
    int u = (int)(input_ptr[u_offset]);
    int v = (int)(input_ptr[v_offset]);

    u -= 128;
    v -= 128;

    int r = y + v + ((v * 103) >> 8);
    int g = y - ((u * 88) >> 8) - ((v * 183) >> 8);
    int b = y + u + ((u * 198) >> 8);

    r = clamp(r, 0, 255);
    g = clamp(g, 0, 255);
    b = clamp(b, 0, 255);

    float4 values = (float4)((float)r, (float)g, (float)b, (float)0.0f);

#ifdef ENABLE_SCALE_BIAS
    values = values * scale + bias;
#endif

#ifdef SWAP_RB
    float temp = values.x;
    values.x   = values.z;
    values.z   = temp;
#endif

    int2 coord = (int2)(image_width_idx, image_height_idx);
    write_imagef(output, coord, values);
}

__kernel void ConvertFromN32FC4Image(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
    __write_only image2d_t output, __private const int channels, __private const float4 scale,
    __private const float4 bias) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    int2 coord    = (int2)(image_width_idx, image_height_idx);
    float4 values = read_imagef(input, SAMPLER, coord);

    if (channels == 3) {
        values.w = 0;
    }

#ifdef SWAP_RB
    float temp = values.x;
    values.x   = values.z;
    values.z   = temp;
#endif

#ifdef ENABLE_SCALE_BIAS
    values = values * scale + bias;
#endif

    write_imagef(output, coord, values);
}

__kernel void CopyFromN8UC3(GLOBAL_SIZE_2_DIMS __write_only image2d_t output,
                               __global const uchar *input_ptr,
                               __private const int height,
                               __private const int width) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx  = image_height_idx / height;
    const int height_idx = image_height_idx % height;

    int buffer_offset =
        ((batch_idx * height + height_idx) * width + image_width_idx) * 3;

    float4 values = (float4)(0.0f);

    values.x = convert_float(input_ptr[buffer_offset]);
    values.y = convert_float(input_ptr[buffer_offset + 1]);
    values.z = convert_float(input_ptr[buffer_offset + 2]);


    int2 coord = (int2)(image_width_idx, image_height_idx);
    write_imagef(output, coord, values);
}

__kernel void CopyFromN8UC4(GLOBAL_SIZE_2_DIMS __write_only image2d_t output,
                               __global const uchar *input_ptr,
                               __private const int height,
                               __private const int width) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx  = image_height_idx / height;
    const int height_idx = image_height_idx % height;

    int buffer_offset =
        ((batch_idx * height + height_idx) * width + image_width_idx) * 4;

    float4 values = convert_float4(vload4(0, input_ptr + buffer_offset));

    int2 coord = (int2)(image_width_idx, image_height_idx);
    write_imagef(output, coord, values);
}
