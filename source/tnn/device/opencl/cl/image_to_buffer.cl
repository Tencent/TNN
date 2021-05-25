#include "base.inc"

__kernel void ImageToNC4HW4Buffer(GLOBAL_SIZE_2_DIMS __global float *output, /* nchw */
                                     __private const int2 output_wh,
                                     __private const int channel_up_4,
                                     __read_only image2d_t input_ptr) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx   = image_height_idx / output_wh.x;
    const int height_idx  = image_height_idx % output_wh.x;
    const int width_idx   = image_width_idx % output_wh.y;
    int channel_block_idx = image_width_idx / output_wh.y;

    int buffer_offset =
        (((batch_idx * channel_up_4 + channel_block_idx) * output_wh.x + height_idx) * output_wh.y + width_idx) * 4;

    int2 coord        = (int2)(image_width_idx, image_height_idx);
    float4 values = read_imagef(input_ptr, SAMPLER, coord);

    vstore4(values, 0, output + buffer_offset);
}

// only for debug
// convert kernel : from image(oc/4 h w , ic oc4) to buffer(oihw)
__kernel void Conv2DFilterImageToBuffer(GLOBAL_SIZE_2_DIMS __global float *output_ptr,
                                            __private const int output_channel, __private const int2 kernel_wh,
                                            __private const int ic_h_w_size,
                                            __private const int height_width_size, __read_only image2d_t input_ptr) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int input_channel_4_idx  = image_width_idx;
    const int output_channel_4_idx = image_height_idx / height_width_size * 4;
    const int height_width_idx     = image_height_idx % height_width_size;
    const int buffer_height_idx    = height_width_idx / kernel_wh.y;
    const int buffer_width_idx     = height_width_idx % kernel_wh.y;

    const int buffer_offset = output_channel_4_idx * ic_h_w_size + input_channel_4_idx * height_width_size +
                              buffer_height_idx * kernel_wh.y + buffer_width_idx;

    if (output_channel_4_idx < output_channel) {
        int2 coord               = (int2)(image_width_idx, image_height_idx);
        float4 values        = read_imagef(input_ptr, SAMPLER, coord);
        const int remain_channel = (output_channel - output_channel_4_idx);

        if (remain_channel >= 4) {
            int offset         = buffer_offset;
            output_ptr[offset] = values.x;
            offset             = mad24(1, ic_h_w_size, offset);
            output_ptr[offset] = values.y;
            offset += ic_h_w_size;
            output_ptr[offset] = values.z;
            offset += ic_h_w_size;
            output_ptr[offset] = values.w;
        } else if (remain_channel == 3) {
            int offset         = buffer_offset;
            output_ptr[offset] = values.x;
            offset             = mad24(1, ic_h_w_size, offset);
            output_ptr[offset] = values.y;
            offset += ic_h_w_size;
            output_ptr[offset] = values.z;

        } else if (remain_channel == 2) {
            int offset         = buffer_offset;
            output_ptr[offset] = values.x;
            offset             = mad24(1, ic_h_w_size, offset);
            output_ptr[offset] = values.y;
        } else if (remain_channel == 1) {
            int offset         = buffer_offset;
            output_ptr[offset] = values.x;
        }
    }
}

// convert data from image(b h, ic/4 w ic4) to buffer(nhwc)
__kernel void ImageToNHWCBuffer(GLOBAL_SIZE_2_DIMS __global float *output, /* nhwc */
                                   __private const int height, __private const int width, __private const int channels,
                                   __read_only image2d_t input_ptr) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = (image_width_idx / width) << 2;
    const int buffer_offset = ((batch_idx * height + height_idx) * width + width_idx) * channels + channel_4_idx;

    int2 coord               = (int2)(image_width_idx, image_height_idx);
    float4 values        = read_imagef(input_ptr, SAMPLER, coord);
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

__kernel void ImageToNHWCBufferFLOAT(GLOBAL_SIZE_2_DIMS __global FLOAT *output, /* nhwc */
                                   __private const int height, __private const int width, __private const int channels,
                                   __read_only image2d_t input_ptr) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx     = image_height_idx / height;
    const int height_idx    = image_height_idx % height;
    const int width_idx     = image_width_idx % width;
    const int channel_4_idx = (image_width_idx / width) << 2;
    const int buffer_offset = ((batch_idx * height + height_idx) * width + width_idx) * channels + channel_4_idx;

    int2 coord               = (int2)(image_width_idx, image_height_idx);
    FLOAT4 values            = RI_F(input_ptr, SAMPLER, coord);
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

// convert data from image(b h, ic/4 w ic4) to buffer(nchw)
__kernel void ImageToNCHWBuffer(GLOBAL_SIZE_2_DIMS __global float *output, /* nchw */
                                   __private const int height, __private const int width, __private const int channels,
                                   __read_only image2d_t input_ptr) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx  = image_height_idx / height;
    const int height_idx = image_height_idx % height;
    const int width_idx  = image_width_idx % width;
    int channel_4_idx    = (image_width_idx / width) * 4;
    int buffer_offset    = ((batch_idx * channels + channel_4_idx) * height + height_idx) * width + width_idx;
    #ifdef ENABLE_BUFFER_PRECISION_ADJUST
    __global FLOAT *output_ptr = (__global FLOAT *)output;
    FLOAT4 values    = RI_F(input_ptr, SAMPLER, (int2)(image_width_idx, image_height_idx));
    #else
    __global float *output_ptr = output;
    float4 values    = read_imagef(input_ptr, SAMPLER, (int2)(image_width_idx, image_height_idx));
    #endif

    const int height_width_size = height * width;

    const int remain_channel = channels - channel_4_idx;

    if (remain_channel >= 4) {
        int offset     = buffer_offset;
        output_ptr[offset] = values.x;
        offset += height_width_size;
        output_ptr[offset] = values.y;
        offset += height_width_size;
        output_ptr[offset] = values.z;
        offset += height_width_size;
        output_ptr[offset] = values.w;
    } else if (remain_channel == 3) {
        int offset     = buffer_offset;
        output_ptr[offset] = values.x;
        offset += height_width_size;
        output_ptr[offset] = values.y;
        offset += height_width_size;
        output_ptr[offset] = values.z;
    } else if (remain_channel == 2) {
        int offset     = buffer_offset;
        output_ptr[offset] = values.x;
        offset += height_width_size;
        output_ptr[offset] = values.y;
    } else if (remain_channel == 1) {
        int offset     = buffer_offset;
        output_ptr[offset] = values.x;
    }
}

__kernel void ImageToNCHWBufferFLOAT(GLOBAL_SIZE_2_DIMS __global FLOAT *output, /* nchw */
                                   __private const int height, __private const int width, __private const int channels,
                                   __read_only image2d_t input_ptr) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int batch_idx  = image_height_idx / height;
    const int height_idx = image_height_idx % height;
    const int width_idx  = image_width_idx % width;
    int channel_4_idx    = (image_width_idx / width) * 4;
    int buffer_offset    = ((batch_idx * channels + channel_4_idx) * height + height_idx) * width + width_idx;
    FLOAT4 values        = RI_F(input_ptr, SAMPLER, (int2)(image_width_idx, image_height_idx));

    const int height_width_size = height * width;

    const int remain_channel = channels - channel_4_idx;

    if (remain_channel >= 4) {
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset += height_width_size;
        output[offset] = values.y;
        offset += height_width_size;
        output[offset] = values.z;
        offset += height_width_size;
        output[offset] = values.w;
    } else if (remain_channel == 3) {
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset += height_width_size;
        output[offset] = values.y;
        offset += height_width_size;
        output[offset] = values.z;
    } else if (remain_channel == 2) {
        int offset     = buffer_offset;
        output[offset] = values.x;
        offset += height_width_size;
        output[offset] = values.y;
    } else if (remain_channel == 1) {
        int offset     = buffer_offset;
        output[offset] = values.x;
    }
}

__kernel void ArgImageToBuffer(GLOBAL_SIZE_2_DIMS __global float *output, __private const int count,
                                  __read_only image2d_t input_ptr) {
    int image_width_idx  = get_global_id(0);
    int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int buffer_4_offset = image_width_idx << 2;

    int2 coord        = (int2)(image_width_idx, image_height_idx);
    float4 values = read_imagef(input_ptr, SAMPLER, coord);
    const int remain  = count - buffer_4_offset;
    if (remain < 4) {
        switch (remain) {
            case 3:
                output[buffer_4_offset + 2] = values.s2;
            case 2:
                output[buffer_4_offset + 1] = values.s1;
            case 1:
                output[buffer_4_offset] = values.s0;
        }
    } else {
        vstore4(values, 0, output + buffer_4_offset);
    }

    if (remain >= 4) {
        vstore4(values, 0, output + buffer_4_offset);
    } else if (remain == 3) {
        int offset     = buffer_4_offset;
        output[offset] = values.x;
        offset++;
        output[offset] = values.y;
        offset++;
        output[offset] = values.z;
    } else if (remain == 2) {
        int offset     = buffer_4_offset;
        output[offset] = values.x;
        offset++;
        output[offset] = values.y;
    } else if (remain == 1) {
        int offset     = buffer_4_offset;
        output[offset] = values.x;
    }
}

__kernel void ImageToRGBABuffer(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_ptr, __global uchar *output,
							  __private const int height, __private const int width, __private const int channel_up_4,
        					  __private const float4 scale, __private const float4 bias) {
	int image_width_idx  = get_global_id(0);
	int image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

	const int batch_idx 	    = image_height_idx / height;
	const int height_idx	    = image_height_idx % height;
	const int width_idx 	    = image_width_idx % width;
	const int channel_block_idx = image_width_idx / width;

	int buffer_offset = (((batch_idx * height + height_idx) * width + width_idx) * channel_up_4 + channel_block_idx) * 4;

	int2 coord        = (int2)(image_width_idx, image_height_idx);
	float4 values_f = read_imagef(input_ptr, SAMPLER, coord);
#ifdef ENABLE_SCALE_BIAS
	values_f = values_f * scale + bias;
#endif
	uchar4 values = convert_uchar4_sat(values_f);

#ifdef SWAP_RB
    uchar temp = values.x;
    values.x = values.z;
    values.z = temp;
#endif

	vstore4(values, 0, output + buffer_offset);
}
