#include "kannarotate.h"
#include <string.h>
#include "rotate_x.h"

typedef void (*rotate_func)(const unsigned char*, int, int, unsigned char*);

static const rotate_func kannarotate_func_table[4][8] =
{
    {
        NULL,        rotate_2_c1, rotate_3_c1, rotate_4_c1,
        rotate_5_c1, rotate_6_c1, rotate_7_c1, rotate_8_c1
    },
    {
        NULL,        rotate_2_c2, rotate_3_c2, rotate_4_c2,
        rotate_5_c2, rotate_6_c2, rotate_7_c2, rotate_8_c2
    },
    {
        NULL,        rotate_2_c3, rotate_3_c3, rotate_4_c3,
        rotate_5_c3, rotate_6_c3, rotate_7_c3, rotate_8_c3
    },
    {
        NULL,        rotate_2_c4, rotate_3_c4, rotate_4_c4,
        rotate_5_c4, rotate_6_c4, rotate_7_c4, rotate_8_c4
    }
};

int kannarotate(const unsigned char* src, int srcw, int srch, unsigned char* dst, int channels, int type)
{
    if (!src || srcw <= 0 || srch <= 0 || !dst)
        return -1;

    if (channels < 1 || channels > 4)
        return -2;

    if (type < 1 || type > 8)
        return -3;

    rotate_func f = kannarotate_func_table[channels-1][type-1];
    if (!f)
    {
        // type 1, copy
        memcpy(dst, src, srcw*srch*channels);

        return 0;
    }

    f(src, srcw, srch, dst);

    return 0;
}

int kannarotate_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int type)
{
    if (!src || srcw <= 0 || srch <= 0 || !dst)
        return -1;

    if (type < 1 || type > 8)
        return -3;

    if (srcw % 2 != 0 || srch % 2 != 0)
        return -4;

    rotate_func f1 = kannarotate_func_table[0][type-1];
    rotate_func f2 = kannarotate_func_table[1][type-1];

    if (!f1 || !f2)
    {
        // type 1, copy
        memcpy(dst, src, srcw*srch*3/2);

        return 0;
    }

    // Y
    f1(src, srcw, srch, dst);

    // UV
    f2(src + srcw*srch, srcw/2, srch/2, dst + srcw*srch);

    return 0;
}
