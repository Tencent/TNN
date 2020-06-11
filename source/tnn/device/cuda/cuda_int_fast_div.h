#ifndef _INT_FASTDIV_KJGIUHFG
#define _INT_FASTDIV_KJGIUHFG

namespace TNN_NS {

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__ __forceinline__
#else
#define CUDA_HOSTDEV inline
#endif

class int_fastdiv {
public:
    // divisor != 0
    CUDA_HOSTDEV int_fastdiv() {}

    void init(int divisor) {
        this->d = divisor;
        update_magic_numbers();
    }

    CUDA_HOSTDEV operator int() const {
        return d;
    }

private:
    int d;
    int M;
    int s;
    int n_add_sign;

    // Hacker's Delight, Second Edition, Chapter 10, Integer Division By
    // Constants
    CUDA_HOSTDEV void update_magic_numbers() {
        if (d == 1) {
            M          = 0;
            s          = -1;
            n_add_sign = 1;
            return;
        }

        int p;
        unsigned int ad, anc, delta, q1, r1, q2, r2, t;
        const unsigned two31 = 0x80000000;
        ad                   = d;
        t                    = two31 + ((unsigned int)d >> 31);
        anc                  = t - 1 - t % ad;
        p                    = 31;
        q1                   = two31 / anc;
        r1                   = two31 - q1 * anc;
        q2                   = two31 / ad;
        r2                   = two31 - q2 * ad;
        do {
            ++p;
            q1 = 2 * q1;
            r1 = 2 * r1;
            if (r1 >= anc) {
                ++q1;
                r1 -= anc;
            }
            q2 = 2 * q2;
            r2 = 2 * r2;
            if (r2 >= ad) {
                ++q2;
                r2 -= ad;
            }
            delta = ad - r2;
        } while (q1 < delta || (q1 == delta && r1 == 0));
        this->M = q2 + 1;
        this->s = p - 32;

        if ((M < 0))
            n_add_sign = 1;
        else
            n_add_sign = 0;
    }

    CUDA_HOSTDEV
    friend int operator/(const int divident, const int_fastdiv& divisor);
};

CUDA_HOSTDEV
int operator/(const int n, const int_fastdiv& divisor) {
    int q;
#ifdef __CUDA_ARCH__
    asm("mul.hi.s32 %0, %1, %2;" : "=r"(q) : "r"(divisor.M), "r"(n));
#else
    q = (((unsigned long long)((long long)divisor.M * (long long)n)) >> 32);
#endif
    q += n * divisor.n_add_sign;
    if (divisor.s >= 0) {
        q >>=
            divisor.s;  // we rely on this to be implemented as arithmetic shift
        q += (((unsigned int)q) >> 31);
    }
    return q;
}

CUDA_HOSTDEV
int operator%(const int n, const int_fastdiv& divisor) {
    int quotient  = n / divisor;
    int remainder = n - quotient * divisor;
    return remainder;
}

CUDA_HOSTDEV
int operator/(const unsigned int n, const int_fastdiv& divisor) {
    return ((int)n) / divisor;
}

CUDA_HOSTDEV
int operator%(const unsigned int n, const int_fastdiv& divisor) {
    return ((int)n) % divisor;
}

CUDA_HOSTDEV
int operator/(const short n, const int_fastdiv& divisor) {
    return ((int)n) / divisor;
}

CUDA_HOSTDEV
int operator%(const short n, const int_fastdiv& divisor) {
    return ((int)n) % divisor;
}

CUDA_HOSTDEV
int operator/(const unsigned short n, const int_fastdiv& divisor) {
    return ((int)n) / divisor;
}

CUDA_HOSTDEV
int operator%(const unsigned short n, const int_fastdiv& divisor) {
    return ((int)n) % divisor;
}

CUDA_HOSTDEV
int operator/(const char n, const int_fastdiv& divisor) {
    return ((int)n) / divisor;
}

CUDA_HOSTDEV
int operator%(const char n, const int_fastdiv& divisor) {
    return ((int)n) % divisor;
}

CUDA_HOSTDEV
int operator/(const unsigned char n, const int_fastdiv& divisor) {
    return ((int)n) / divisor;
}

CUDA_HOSTDEV
int operator%(const unsigned char n, const int_fastdiv& divisor) {
    return ((int)n) % divisor;
}

#undef CUDA_HOSTDEV

}  // namespace TNN_NS

#endif
