#include "base.inc"

__kernel void Tile(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __write_only image2d_t output,
                  __private const int input_batch,
                  __private const int input_channel,
                   __private const int input_height,
                   __private const int input_width,
                   __private const int output_batch,
                   __private const int output_channel,
                   __private const int output_height,
                   __private const int output_width
                   ) {
    const int cw = get_global_id(0);
    const int hb = get_global_id(1);
    const int count = output_width * output_height *  output_channel * output_batch;
    DEAL_NON_UNIFORM_DIM2(cw, hb);
    int out_batch =  hb / output_height;
    int out_channel = cw / output_width;
    int out_height = hb % output_height;
    int out_width = cw % output_width;


    // compute four position of output
    // 0 index
    int index =  out_batch * output_channel * output_height * output_width + out_channel * 4 * output_height  * output_width +
                 out_height * output_width + out_width;
    if (out_channel * 4 >= output_channel) {
       FLOAT4 out_zero = (FLOAT4)((FLOAT)0, (FLOAT)0, (FLOAT)0, (FLOAT)0);
       WI_F(output, (int2)(cw, hb), out_zero);
       return;
    }

    int prod = count;
    prod = prod / output_batch;
    int batch = index / prod % input_batch;

    prod = prod /  output_channel;
    int channel = index / prod % input_channel;

    prod = prod / output_height;
    int h =  index / prod % input_height;

    prod = prod / output_width;
    int w  = index / prod % input_width;

    FLOAT4 in = RI_F(input, SAMPLER, (int2)( channel / 4 * input_width + w , batch * input_height + h));
    int idx = channel % 4;
    FLOAT out0 = idx == 0 ? in.x : (idx == 1 ? in.y : (idx == 2 ? in.z : in.w));

    //1 index
    index = index + output_height  * output_width;
    prod = count;
    prod = prod / output_batch;
    batch = index / prod % input_batch;

    prod = prod / output_channel;
    channel = index / prod % input_channel;

    prod = prod / output_height;
    h =  index / prod % input_height;

    prod = prod / output_width;
    w  = index / prod % input_width;

    in = RI_F(input, SAMPLER, (int2)(channel / 4 * input_width + w , batch * input_height + h));
    idx = channel % 4;
    FLOAT out1 = idx == 0 ? in.x : (idx == 1 ? in.y : (idx == 2 ? in.z : in.w));
    out1 = (out_channel * 4 + 1) >= output_channel ? (FLOAT)0 : out1;

    //2 index
    index = index + output_height  * output_width;
    prod = count;
    prod = prod / output_batch;
    batch = index / prod % input_batch;

    prod = prod / output_channel;
    channel = index / prod % input_channel;

    prod = prod / output_height;
    h =  index / prod % input_height;

    prod = prod / output_width;
    w  = index / prod % input_width;

    in = RI_F(input, SAMPLER, (int2)(channel / 4 * input_width + w , batch * input_height + h));
    idx = channel % 4;
    FLOAT out2 = idx == 0 ? in.x : (idx == 1 ? in.y : (idx == 2 ? in.z : in.w));
    out2 = (out_channel * 4 + 2) >= output_channel ? (FLOAT)0 : out2;


    //3 index
    index = index + output_height  * output_width;
    prod = count;
    prod = prod / output_batch;
    batch = index / prod % input_batch;

    prod = prod / output_channel;
    channel = index / prod % input_channel;

    prod = prod / output_height;
    h =  index / prod % input_height;

    prod = prod / output_width;
    w  = index / prod % input_width;

    in = RI_F(input, SAMPLER, (int2)(channel / 4 * input_width + w , batch * input_height + h));
    idx = channel % 4;
    FLOAT out3 = idx == 0 ? in.x : (idx == 1 ? in.y : (idx == 2 ? in.z : in.w));
    out3 = (out_channel * 4 + 3) >= output_channel ? (FLOAT)0 : out3;


    FLOAT4 out = (FLOAT4)(out0, out1, out2, out3);
    WI_F(output, (int2)(cw, hb), out);
}