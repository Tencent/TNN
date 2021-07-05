#include "base.inc"

__kernel void NC4HW4BufferToImage(GLOBAL_SIZE_2_DIMS __global const float *input_ptr, __private const int2 output_wh,
                                     __private const int channel_up_4, __write_only image2d_t output) {

    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx         = image_height_idx / output_wh.x;
    const int height_idx        = image_height_idx % output_wh.x;
    const int width_idx         = image_width_idx % output_wh.y;
    const int channel_block_idx = image_width_idx / output_wh.y;
    int buffer_offset =
        (((batch_idx * channel_up_4 + channel_block_idx) * output_wh.x + height_idx) * output_wh.y + width_idx) * 4;

    float4 values = vload4(0, input_ptr + buffer_offset);

    int2 coord = (int2)(image_width_idx, image_height_idx);
    write_imagef(output, coord, values);
}

// convert kernel : from buffer(oihw) to image [w,h]=(ic oc4, oc/4 h w)
__kernel void Conv2DFilterBufferToImage(GLOBAL_SIZE_2_DIMS __global const float *input_ptr,
                                            __private const int output_channel, __private const int2 kernel_wh, __private const int ic_h_w_size,
                                            __private const int height_width_size, __write_only image2d_t output) {
    int image_width_idx  = get_global_id(0); // ic
    int image_height_idx = get_global_id(1); // oc/4 h w

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int input_channel_4_idx  = image_width_idx;
    const int output_channel_4_idx = (image_height_idx / height_width_size) * 4;
    const int height_width_idx     = image_height_idx % height_width_size;
    const int buffer_height_idx    = height_width_idx / kernel_wh.y;
    const int buffer_width_idx     = height_width_idx % kernel_wh.y;

    const int buffer_offset = output_channel_4_idx * ic_h_w_size + input_channel_4_idx * height_width_size +
                              buffer_height_idx * kernel_wh.y + buffer_width_idx;

    float4 output_values = 0;
    if (output_channel_4_idx < output_channel) {
        const int remain_channel = output_channel - output_channel_4_idx;
        if (remain_channel >= 4) {
            int offset      = buffer_offset;
            output_values.x = *(input_ptr + offset);
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = *(input_ptr + offset);
            offset += ic_h_w_size;
            output_values.z = *(input_ptr + offset);
            offset += ic_h_w_size;
            output_values.w = *(input_ptr + offset);
        } else if (remain_channel == 3) {
            int offset      = buffer_offset;
            output_values.x = *(input_ptr + offset);
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = *(input_ptr + offset);
            offset += ic_h_w_size;
            output_values.z = *(input_ptr + offset);

        } else if (remain_channel == 2) {
            int offset      = buffer_offset;
            output_values.x = *(input_ptr + offset);
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = *(input_ptr + offset);
        } else if (remain_channel == 1) {
            int offset      = buffer_offset;
            output_values.x = *(input_ptr + offset);
        }
    }

    write_imagef(output, (int2)(image_width_idx, image_height_idx), output_values);
}


// convert kernel from buffer(mihw) to image [w,h]=(h w m ic4, ic/4)
// but now dw only support m == 1
__kernel void DWFilterBufferToImage(GLOBAL_SIZE_2_DIMS __global const float *input_ptr,
                                        __private const int4 kernel_shape,
                                        __private const int height_width_size, __write_only image2d_t output) {
    const int image_width_idx  = get_global_id(0);
    const int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    float4 output_values = 0;
    if (kernel_shape.x == 1) {
        const int input_channel_4_idx = image_height_idx << 2;
        const int buffer_height_idx   = image_width_idx / kernel_shape.w;
        const int buffer_width_idx    = image_width_idx % kernel_shape.w;

        const int buffer_offset =
            mad24(mad24(input_channel_4_idx, kernel_shape.z, buffer_height_idx), kernel_shape.w, buffer_width_idx);

        const int remain_channel = kernel_shape.y - input_channel_4_idx;
        if (input_channel_4_idx < kernel_shape.y) {
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
        }
    }

    write_imagef(output, (int2)(image_width_idx, image_height_idx), output_values);
}

// convert data from buffer(nhwc) to image(b h, ic/4 w ic4)
__kernel void NHWCBufferToImage(GLOBAL_SIZE_2_DIMS __global const float *input_ptr, __private const int height,
                                   __private const int width, __private const int channels,
                                   __write_only image2d_t output) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = (image_width_idx / width) << 2;
    const int buffer_offset = ((batch_idx * height + height_idx) * width + width_idx) * channels + channel_4_idx;

    const int remain_channel                    = channels - channel_4_idx;
    __global const float *input_current_ptr = input_ptr + buffer_offset;
    float4 values                           = 0;
    values                                      = vload4(0, input_current_ptr);

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
    write_imagef(output, (int2)(image_width_idx, image_height_idx), values);
}

__kernel void NHWCBufferToImageFLOAT(GLOBAL_SIZE_2_DIMS __global const FLOAT *input_ptr, __private const int height,
                                   __private const int width, __private const int channels,
                                   __write_only image2d_t output) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = (image_width_idx / width) << 2;
    const int buffer_offset = ((batch_idx * height + height_idx) * width + width_idx) * channels + channel_4_idx;

    const int remain_channel                = channels - channel_4_idx;
    __global const FLOAT *input_current_ptr = input_ptr + buffer_offset;
    FLOAT4 values                           = 0;
    values                                  = vload4(0, input_current_ptr);

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
    WI_F(output, (int2)(image_width_idx, image_height_idx), values);
}

// convert data from buffer(nchw) to image(b h, ic/4 w ic4)
__kernel void NCHWBufferToImage(GLOBAL_SIZE_2_DIMS __global const float *input_ptr, /* nchw */
                                   __private const int height, __private const int width, __private const int channels,
                                   __write_only image2d_t output) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = (image_width_idx / width) << 2;
    const int buffer_offset = ((batch_idx * channels + channel_4_idx) * height + height_idx) * width + width_idx;

    const int remain_channel    = channels - channel_4_idx;
    const int height_width_size = height * width;
    float4 output_values    = 0;

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

    write_imagef(output, (int2)(image_width_idx, image_height_idx), output_values);
}

__kernel void NCHWBufferToImageFLOAT(GLOBAL_SIZE_2_DIMS __global const FLOAT *input_ptr, /* nchw */
                                   __private const int height, __private const int width, __private const int channels,
                                   __write_only image2d_t output) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = image_width_idx / width << 2;
    const int buffer_offset = ((batch_idx * channels + channel_4_idx) * height + height_idx) * width + width_idx;

    const int remain_channel    = channels - channel_4_idx;
    const int height_width_size = height * width;
    FLOAT4 output_values        = 0;

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

    WI_F(output, (int2)(image_width_idx, image_height_idx), output_values);
}

// convert data from buffer(nchw) to image(b h, ic/4 w ic4)
__kernel void ArgBufferToImage(GLOBAL_SIZE_2_DIMS __global const float *input_ptr, __private const int count,
                                  __write_only image2d_t output) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int buffer_4_offset = image_width_idx << 2;
    const int remain          = count - buffer_4_offset;

    int offset = buffer_4_offset;
    float4 values = 0;
    if (remain >= 4) {
        values = vload4(0, input_ptr + offset);
    } else if (remain == 3) {
        values.x = *(input_ptr + offset);
        offset++;
        values.y = *(input_ptr + offset);
        offset++;
        values.z = *(input_ptr + offset);
    } else if (remain == 2) {
        values.x = *(input_ptr + offset);
        offset++;
        values.y = *(input_ptr + offset);
    } else if (remain == 1) {
        values.x = *(input_ptr + offset);
    }
    write_imagef(output, (int2)(image_width_idx, image_height_idx), values);
}

// convert data from buffer(num_directions, 4 * hidden_size, weights_width)
// to image(weights_width, num_directions * 4 * hidden_size/4 hidden_size4)
__kernel void LstmFilterBufferToImage(GLOBAL_SIZE_2_DIMS __global const float *input_ptr,
                                      __private const int num_directions,
                                      __private const int hidden_size,
                                      __private const int weights_width,
                                      __private const int hidden_updiv_4_size,
                                      __private const int hidden_mul_4_size,
                                      __write_only image2d_t output) {
    int image_width_idx   = get_global_id(0);
    int weights_width_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, weights_width_idx);

    const int hid_4_idx     = image_width_idx % hidden_updiv_4_size;
    const int dir_gate_idx  = image_width_idx / hidden_updiv_4_size;
    const int dir_idx       = dir_gate_idx >> 2;
    const int gate_idx      = dir_gate_idx % 4;
    const int hid_idx       = hid_4_idx << 2;
    const int remain        = hidden_size - hid_idx;

    int buffer_offset       = (dir_idx * hidden_mul_4_size + gate_idx * hidden_size + hid_idx) *
                              weights_width + weights_width_idx;

    float4 out = 0;
    if (remain >= 4) {
        out.x = *(input_ptr + buffer_offset);
        buffer_offset += weights_width;
        out.y = *(input_ptr + buffer_offset);
        buffer_offset += weights_width;
        out.z = *(input_ptr + buffer_offset);
        buffer_offset += weights_width;
        out.w = *(input_ptr + buffer_offset);
    } else if (remain == 3) {
        out.x = *(input_ptr + buffer_offset);
        buffer_offset += weights_width;
        out.y = *(input_ptr + buffer_offset);
        buffer_offset += weights_width;
        out.z = *(input_ptr + buffer_offset);
    } else if (remain == 2) {
        out.x = *(input_ptr + buffer_offset);
        buffer_offset += weights_width;
        out.y = *(input_ptr + buffer_offset);
    } else if (remain == 1) {
        out.x = *(input_ptr + buffer_offset);
    }
    write_imagef(output, (int2)(image_width_idx, weights_width_idx), out);
}

// convert data from buffer(num_directions, 8 * hidden_size)
// to image(num_directions, 8 * hidden_size/4 hidden_size4)
__kernel void LstmBiasBufferToImage(GLOBAL_SIZE_2_DIMS __global const float *input_ptr,
                                      __private const int num_directions,
                                      __private const int hidden_size,
                                      __private const int hidden_updiv_4_size,
                                      __private const int hidden_mul_8_size,
                                      __write_only image2d_t output) {
    int image_width_idx  = get_global_id(0);
    int dir_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, dir_idx);

    const int hid_4_idx     = image_width_idx % hidden_updiv_4_size;
    const int gate_idx      = image_width_idx / hidden_updiv_4_size;
    const int hid_idx       = hid_4_idx << 2;
    const int remain        = hidden_size - hid_idx;

    int buffer_offset       = dir_idx * hidden_mul_8_size + gate_idx * hidden_size + hid_idx;

    float4 out = 0;
    if (remain >= 4) {
        out = vload4(0, input_ptr + buffer_offset);
    } else if (remain == 3) {
        out.xyz = vload3(0, input_ptr + buffer_offset);
    } else if (remain == 2) {
        out.xy = vload2(0, input_ptr + buffer_offset);
    } else if (remain == 1) {
        out.x = *(input_ptr + buffer_offset);
    }
    write_imagef(output, (int2)(image_width_idx, dir_idx), out);
}

__kernel void RGBABufferToImage(GLOBAL_SIZE_2_DIMS __global const uchar *input_ptr, __write_only image2d_t output, 
								   __private const int height, __private const int width,
					      		   __private const float4 scale, __private const float4 bias) {

    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

	const int batch_idx  = image_height_idx / height;
	const int height_idx = image_height_idx % height;

	int buffer_offset = ((batch_idx * height + height_idx) * width + image_width_idx) * 4;

    float4 values = convert_float4(vload4(0, input_ptr + buffer_offset));
#ifdef ENABLE_SCALE_BIAS
    values = values * scale + bias;
#endif

#ifdef SWAP_RB
    float temp = values.x;
    values.x = values.z;
    values.z = temp;
#endif

    int2 coord = (int2)(image_width_idx, image_height_idx);
    write_imagef(output, coord, values);
}

__kernel void NV21ToImage(GLOBAL_SIZE_2_DIMS __global const uchar *input_ptr, __write_only image2d_t output, 
						  __private const int height, __private const int width,
					      __private const float4 scale, __private const float4 bias) {

    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

	int y_offset = image_height_idx * width + image_width_idx;
	int v_offset = width * height + (image_height_idx >> 1) * width + (image_width_idx & (~(0x01)));
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
    values.x = values.z;
    values.z = temp;
#endif

    int2 coord = (int2)(image_width_idx, image_height_idx);
    write_imagef(output, coord, values);
}
