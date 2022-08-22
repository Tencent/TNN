#include "base.inc"

// matrix_a format: image:(K/4, batch_a * M)
// matrix_b format: image:(N/4, batch_b * K)
// matrix_c format: image:(N/4, batch_c * M)
__kernel void MatMul(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t matrix_a,
    __read_only image2d_t matrix_b,
    __private const int m,
    __private const int k_blocks, __private const int k, __private const int k_remain,
    __private const int batch_a, __private const int batch_b,
    __write_only image2d_t matrix_c) {
    const int image_row = get_global_id(0);
    const int image_col = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_row, image_col);

    int batch_c_idx = image_col / m;
    int m_idx       = image_col % m;
    int batch_a_idx = select(0, batch_c_idx, batch_c_idx < batch_a);
    int batch_b_idx = select(0, batch_c_idx, batch_c_idx < batch_b);

    FLOAT4 matrix_a_data;
    FLOAT4 matrix_b_data_0;
    FLOAT4 matrix_b_data_1;
    FLOAT4 matrix_b_data_2;
    FLOAT4 matrix_b_data_3;
    FLOAT4 sum = (FLOAT4)0;

    int k_size = k_blocks;
    if (k_remain > 0) {
        k_size--;
    }

    int matrix_a_y_idx = batch_a_idx * m + m_idx;
    int matrix_b_y_offset = batch_b_idx * k;
    int y = matrix_b_y_offset, i = 0;
    for (; i < k_size; i++) {
        matrix_a_data = RI_F(matrix_a, SAMPLER, (int2)(i, matrix_a_y_idx));
        matrix_b_data_0 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y));
        matrix_b_data_1 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y + 1));
        matrix_b_data_2 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y + 2));
        matrix_b_data_3 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y + 3));

        sum = mad(matrix_a_data.x, matrix_b_data_0, sum);
        sum = mad(matrix_a_data.y, matrix_b_data_1, sum);
        sum = mad(matrix_a_data.z, matrix_b_data_2, sum);
        sum = mad(matrix_a_data.w, matrix_b_data_3, sum);
        y += 4;
    }

    if (k_remain > 0) {
        matrix_a_data = RI_F(matrix_a, SAMPLER, (int2)(i, matrix_a_y_idx));
    }

    switch (k_remain) {
        case 3:
            matrix_b_data_0 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y));
            matrix_b_data_1 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y + 1));
            matrix_b_data_2 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y + 2));
            sum      = mad(matrix_a_data.x, matrix_b_data_0, sum);
            sum      = mad(matrix_a_data.y, matrix_b_data_1, sum);
            sum      = mad(matrix_a_data.z, matrix_b_data_2, sum);
            break;
        case 2:
            matrix_b_data_0 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y));
            matrix_b_data_1 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y + 1));
            sum      = mad(matrix_a_data.x, matrix_b_data_0, sum);
            sum      = mad(matrix_a_data.y, matrix_b_data_1, sum);
            break;
        case 1:
            matrix_b_data_0 = RI_F(matrix_b, SAMPLER, (int2)(image_row, y));
            sum      = mad(matrix_a_data.x, matrix_b_data_0, sum);
            break;
    }
    WI_F(matrix_c, (int2)(image_row, image_col), sum);
}

__kernel void MatMul6D(GLOBAL_SIZE_2_DIMS __read_only image2d_t matrix_a, __read_only image2d_t matrix_b,
                       shape_6d matrix_a_shape, shape_6d matrix_b_shape, shape_6d matrix_c_shape,
                       __private const int matrix_a_c_4_blocks, __private const int matrix_b_c_4_blocks,
                       __write_only image2d_t matrix_c) {
    const int image_row = get_global_id(0);
    const int image_col = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(image_row, image_col);

    const int output_d2xd3 = matrix_c_shape.data[2] * matrix_c_shape.data[3];
    const int output_d2_d3 = image_col % output_d2xd3;
    const int output_d4xd5 = matrix_c_shape.data[4] * matrix_c_shape.data[5];
    const int output_d4_d5 = image_row % output_d4xd5;

    const int output_b_idx   = image_col / output_d2xd3;
    const int output_c_4_idx = image_row / output_d4xd5;
    const int output_d2_idx  = output_d2_d3 / matrix_c_shape.data[3];
    const int output_d3_idx  = output_d2_d3 % matrix_c_shape.data[3];
    const int output_d4_idx  = output_d4_d5 / matrix_c_shape.data[5];
    const int output_d5_idx  = output_d4_d5 % matrix_c_shape.data[5];

    const int matrix_a_b_idx   = select(output_b_idx, 0, matrix_a_shape.data[0] == 1);
    const int matrix_a_c_4_idx = select(matrix_a_c_4_blocks - 1, output_c_4_idx, output_c_4_idx < matrix_a_c_4_blocks);
    const int matrix_a_d2_idx  = select(output_d2_idx, 0, matrix_a_shape.data[2] == 1);
    const int matrix_a_d3_idx  = select(output_d3_idx, 0, matrix_a_shape.data[3] == 1);
    const int matrix_a_d4_idx  = select(output_d4_idx, 0, matrix_a_shape.data[4] == 1);
    const int matrix_a_d5_idx  = 0;

    const int matrix_b_b_idx   = select(output_b_idx, 0, matrix_b_shape.data[0] == 1);
    const int matrix_b_c_4_idx = select(matrix_b_c_4_blocks - 1, output_c_4_idx, output_c_4_idx < matrix_b_c_4_blocks);
    const int matrix_b_d2_idx  = select(output_d2_idx, 0, matrix_b_shape.data[2] == 1);
    const int matrix_b_d3_idx  = select(output_d3_idx, 0, matrix_b_shape.data[3] == 1);
    const int matrix_b_d4_idx  = 0;
    const int matrix_b_d5_idx  = select(output_d5_idx, 0, matrix_b_shape.data[5] == 1);

    int matrix_a_ix = matrix_a_c_4_idx * matrix_a_shape.data[4] * matrix_a_shape.data[5] +
                      matrix_a_d4_idx * matrix_a_shape.data[5] + matrix_a_d5_idx;
    int matrix_a_iy = matrix_a_b_idx * matrix_a_shape.data[2] * matrix_a_shape.data[3] +
                      matrix_a_d2_idx * matrix_a_shape.data[3] + matrix_a_d3_idx;

    int matrix_b_ix = matrix_b_c_4_idx * matrix_b_shape.data[4] * matrix_b_shape.data[5] +
                      matrix_b_d4_idx * matrix_b_shape.data[5] + matrix_b_d5_idx;
    int matrix_b_iy = matrix_b_b_idx * matrix_b_shape.data[2] * matrix_b_shape.data[3] +
                      matrix_b_d2_idx * matrix_b_shape.data[3] + matrix_b_d3_idx;

    // use float for calculations to prevent overflow.
    float4 sum = (float4)0;
    float4 in0, in1;
    const int K = matrix_a_shape.data[5];
    for (int k = 1; k <= K; k++) {
        in0 = convert_float4(RI_F(matrix_a, SAMPLER, (int2)(matrix_a_ix, matrix_a_iy)));
        in1 = convert_float4(RI_F(matrix_b, SAMPLER, (int2)(matrix_b_ix, matrix_b_iy)));

        if (matrix_a_shape.data[1] == 1) {
            in0.y = in0.x;
            in0.z = in0.x;
            in0.w = in0.x;
        }

        if (matrix_b_shape.data[1] == 1) {
            in1.y = in1.x;
            in1.z = in1.x;
            in1.w = in1.x;
        }

        sum = mad(in0, in1, sum);

        matrix_a_ix += k;
        matrix_b_ix += matrix_b_shape.data[5];
    }

    WI_F(matrix_c, (int2)(image_row, image_col), CONVERT_FLOAT4(sum));
}
