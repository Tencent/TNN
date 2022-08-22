#include "base.inc"
#include "activation.inc"
#include "io.inc"

__kernel void DepthwiseConv2DS1(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                                __read_only image2d_t filter,
                                __read_only image2d_t bias,
                                __write_only image2d_t output,
                                __private const int2 input_wh,
                                __private const int2 output_wh,
                                __private const int2 kernel_wh,
                                __private const int2 padding_wh,
                                __private const int activation_type) {
    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightBlockIdx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightBlockIdx);

#ifdef CHECK_INPUT_COOR
    int2 input_dims = get_image_dim(input);
#endif

    int ow4                      = (output_wh.x + 3) / 4;
    const int outChannelBlockIdx = outChannelWidthIdx / ow4;
    const int outWidthBlockidx   = outChannelWidthIdx % ow4;

    const int inChannelBlockIdx = outChannelBlockIdx;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    const int outWidthBlockidx4 = outWidthBlockidx << 2;
    const int in_width0         = outWidthBlockidx4 - padding_wh.x;
    const int in_width1         = in_width0 + 1;
    const int in_width2         = in_width0 + 2;
    const int in_width3         = in_width0 + 3;

    int heightIdx = outHeightBlockIdx % output_wh.y - padding_wh.y;
    const int outBatchIdx =
        mul24((outHeightBlockIdx / output_wh.y), input_wh.y);
    const int in_idx = mul24(inChannelBlockIdx, input_wh.x);

    const int inWidthIdx0 = select(in_idx + in_width0, -1, (in_width0 < 0 || in_width0 >= input_wh.x));
    const int inWidthIdx1 = select(in_idx + in_width1, -1, (in_width1 < 0 || in_width1 >= input_wh.x));
    const int inWidthIdx2 = select(in_idx + in_width2, -1, (in_width2 < 0 || in_width2 >= input_wh.x));

    FLOAT4 in0, in1, in2, in3;
    for (int kh = 0; kh < kernel_wh.y; kh++) {
        int in_hb_value = select(heightIdx + outBatchIdx, -1, (heightIdx < 0 || heightIdx >= input_wh.y));
        heightIdx++;
        in1 = RI_F(input, SAMPLER, (int2)(inWidthIdx0, in_hb_value));
        in2 = RI_F(input, SAMPLER, (int2)(inWidthIdx1, in_hb_value));
        in3 = RI_F(input, SAMPLER, (int2)(inWidthIdx2, in_hb_value));

#ifdef CHECK_INPUT_COOR
        if (!InRange((int2)(inWidthIdx0, in_hb_value), input_dims)) {
            in1 = (FLOAT4)0;
        }
        if (!InRange((int2)(inWidthIdx1, in_hb_value), input_dims)) {
            in2 = (FLOAT4)0;
        }
        if (!InRange((int2)(inWidthIdx2, in_hb_value), input_dims)) {
            in3 = (FLOAT4)0;
        }
#endif

        for (int kw = 0; kw < kernel_wh.x; kw++) {
            int filterIdx = mad24(kh, kernel_wh.x, kw);

            in0 = in1;
            in1 = in2;
            in2 = in3;

            int inWidthIdx = in_width3 + kw;
            inWidthIdx     = select(in_idx + inWidthIdx, -1, (inWidthIdx < 0 || inWidthIdx >= input_wh.x));
            in3 = RI_F(input, SAMPLER, (int2)(inWidthIdx, in_hb_value));

#ifdef CHECK_INPUT_COOR
            if (!InRange((int2)(inWidthIdx, in_hb_value), input_dims)) {
                in3 = (FLOAT4)0;
            }
#endif

            FLOAT4 weights = RI_F(filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));

            out0 = mad(in0, weights, out0);
            out1 = mad(in1, weights, out1);
            out2 = mad(in2, weights, out2);
            out3 = mad(in3, weights, out3);
        }
    }

    out0 = ActivationProcess(out0, activation_type);
    out1 = ActivationProcess(out1, activation_type);
    out2 = ActivationProcess(out2, activation_type);
    out3 = ActivationProcess(out3, activation_type);

    const int remain = output_wh.x - outWidthBlockidx4;
    int outWidthIdx = mul24(outChannelBlockIdx, output_wh.x) + outWidthBlockidx4;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, outWidthIdx,
                               outHeightBlockIdx, remain);
}

__kernel void DepthwiseConv2D(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
    __read_only image2d_t filter, __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int2 output_wh, __private const int2 kernel_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int2 stride_wh,
    __private const int activation_type) {
    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightIdx       = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightIdx);

#ifdef CHECK_INPUT_COOR
    int2 input_dims = get_image_dim(input);
#endif

    int ow4                      = (output_wh.x + 3) / 4;
    const int outChannelBlockIdx = outChannelWidthIdx / ow4;
    const int outWidthBlockidx   = outChannelWidthIdx % ow4;

    const int inChannelBlockIdx = outChannelBlockIdx;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    const int in_width0 = mad24(outWidthBlockidx, stride_wh.x << 2, -padding_wh.x);
    const int in_width1 = in_width0 + stride_wh.x;
    const int in_width2 = in_width1 + stride_wh.x;
    const int in_width3 = in_width2 + stride_wh.x;
    int heightIdx = mad24(outHeightIdx % output_wh.y, stride_wh.y, -padding_wh.y);

    const int outBatchIdx = mul24((outHeightIdx / output_wh.y), input_wh.y);

    const int in_idx = mul24(inChannelBlockIdx, input_wh.x);
    for (int kh = 0; kh < kernel_wh.y; kh++) {
        int in_hb_value = select(heightIdx + outBatchIdx, -1, (heightIdx < 0 || heightIdx >= input_wh.y));
        heightIdx += dilation_wh.y;
        for (int kw = 0; kw < kernel_wh.x; kw++) {
            int filterIdx = mad24(kh, kernel_wh.x, kw);
            FLOAT4 in0, in1, in2, in3;
            int inWidthIdx = mul24(kw, dilation_wh.x);

            READ_INPUT_IMAGE(0, inWidthIdx);
            READ_INPUT_IMAGE(1, inWidthIdx);
            READ_INPUT_IMAGE(2, inWidthIdx);
            READ_INPUT_IMAGE(3, inWidthIdx);

            FLOAT4 weights = RI_F(filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));

            out0 = mad(in0, weights, out0);
            out1 = mad(in1, weights, out1);
            out2 = mad(in2, weights, out2);
            out3 = mad(in3, weights, out3);
        }
    }

    out0 = ActivationProcess(out0, activation_type);
    out1 = ActivationProcess(out1, activation_type);
    out2 = ActivationProcess(out2, activation_type);
    out3 = ActivationProcess(out3, activation_type);

    const int outWidthBlockidx4 = outWidthBlockidx << 2;
    const int remain            = output_wh.x - outWidthBlockidx4;
    int outWidthIdx = mul24(outChannelBlockIdx, output_wh.x) + outWidthBlockidx4;

    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, outWidthIdx,
                               outHeightIdx, remain);
}
