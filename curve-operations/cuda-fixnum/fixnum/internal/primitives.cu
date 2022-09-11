#pragma once

#include <cstdint>
#include <cassert>
#include <type_traits>

namespace cuFIXNUM
{

namespace internal
{
typedef std::uint32_t u32;
typedef std::uint64_t u64;

__device__ __forceinline__ void
addc(u32 &s, u32 a, u32 b)
{
    asm("addc.u32 %0, %1, %2;"
        : "=r"(s)
        : "r"(a), "r"(b));
}

__device__ __forceinline__ void
add_cc(u32 &s, u32 a, u32 b)
{
    asm("add.cc.u32 %0, %1, %2;"
        : "=r"(s)
        : "r"(a), "r"(b));
}

__device__ __forceinline__ void
addc_cc(u32 &s, u32 a, u32 b)
{
    asm("addc.cc.u32 %0, %1, %2;"
        : "=r"(s)
        : "r"(a), "r"(b));
}

__device__ __forceinline__ void
addc(u64 &s, u64 a, u64 b)
{
    asm("addc.u64 %0, %1, %2;"
        : "=l"(s)
        : "l"(a), "l"(b));
}

__device__ __forceinline__ void
add_cc(u64 &s, u64 a, u64 b)
{
    asm("add.cc.u64 %0, %1, %2;"
        : "=l"(s)
        : "l"(a), "l"(b));
}

__device__ __forceinline__ void
addc_cc(u64 &s, u64 a, u64 b)
{
    asm("addc.cc.u64 %0, %1, %2;"
        : "=l"(s)
        : "l"(a), "l"(b));
}

/*
     * hi * 2^n + lo = a * b
     */
__device__ __forceinline__ void
mul_hi(u32 &hi, u32 a, u32 b)
{
    asm("mul.hi.u32 %0, %1, %2;"
        : "=r"(hi)
        : "r"(a), "r"(b));
}

__device__ __forceinline__ void
mul_hi(u64 &hi, u64 a, u64 b)
{
    asm("mul.hi.u64 %0, %1, %2;"
        : "=l"(hi)
        : "l"(a), "l"(b));
}

/*
     * hi * 2^n + lo = a * b
     */
__device__ __forceinline__ void
mul_wide(u32 &hi, u32 &lo, u32 a, u32 b)
{
    // TODO: Measure performance difference between this and the
    // equivalent:
    //   mul.hi.u32 %0, %2, %3
    //   mul.lo.u32 %1, %2, %3
    asm("{\n\t"
        " .reg .u64 tmp;\n\t"
        " mul.wide.u32 tmp, %2, %3;\n\t"
        " mov.b64 { %1, %0 }, tmp;\n\t"
        "}"
        : "=r"(hi), "=r"(lo)
        : "r"(a), "r"(b));
}

__device__ __forceinline__ void
mul_wide(u64 &hi, u64 &lo, u64 a, u64 b)
{
    asm("mul.hi.u64 %0, %2, %3;\n\t"
        "mul.lo.u64 %1, %2, %3;"
        : "=l"(hi), "=l"(lo)
        : "l"(a), "l"(b));
}

/*
     * (hi, lo) = a * b + c
     */
__device__ __forceinline__ void
mad_wide(u32 &hi, u32 &lo, u32 a, u32 b, u32 c)
{
    asm("{\n\t"
        " .reg .u64 tmp;\n\t"
        " mad.wide.u32 tmp, %2, %3, %4;\n\t"
        " mov.b64 { %1, %0 }, tmp;\n\t"
        "}"
        : "=r"(hi), "=r"(lo)
        : "r"(a), "r"(b), "r"(c));
}

__device__ __forceinline__ void
mad_wide(u64 &hi, u64 &lo, u64 a, u64 b, u64 c)
{
    asm("mad.lo.cc.u64 %1, %2, %3, %4;\n\t"
        "madc.hi.u64 %0, %2, %3, 0;"
        : "=l"(hi), "=l"(lo)
        : "l"(a), "l"(b), "l"(c));
}

// lo = a * b + c (mod 2^n)
__device__ __forceinline__ void
mad_lo(u32 &lo, u32 a, u32 b, u32 c)
{
    asm("mad.lo.u32 %0, %1, %2, %3;"
        : "=r"(lo)
        : "r"(a), "r"(b), "r"(c));
}

__device__ __forceinline__ void
mad_lo(u64 &lo, u64 a, u64 b, u64 c)
{
    asm("mad.lo.u64 %0, %1, %2, %3;"
        : "=l"(lo)
        : "l"(a), "l"(b), "l"(c));
}

// as above but with carry in cy
__device__ __forceinline__ void
mad_lo_cc(u32 &lo, u32 a, u32 b, u32 c)
{
    asm("mad.lo.cc.u32 %0, %1, %2, %3;"
        : "=r"(lo)
        : "r"(a), "r"(b), "r"(c));
}

__device__ __forceinline__ void
mad_lo_cc(u64 &lo, u64 a, u64 b, u64 c)
{
    asm("mad.lo.cc.u64 %0, %1, %2, %3;"
        : "=l"(lo)
        : "l"(a), "l"(b), "l"(c));
}

__device__ __forceinline__ void
madc_lo_cc(u32 &lo, u32 a, u32 b, u32 c)
{
    asm("madc.lo.cc.u32 %0, %1, %2, %3;"
        : "=r"(lo)
        : "r"(a), "r"(b), "r"(c));
}

__device__ __forceinline__ void
madc_lo_cc(u64 &lo, u64 a, u64 b, u64 c)
{
    asm("madc.lo.cc.u64 %0, %1, %2, %3;"
        : "=l"(lo)
        : "l"(a), "l"(b), "l"(c));
}

__device__ __forceinline__ void
mad_hi(u32 &hi, u32 a, u32 b, u32 c)
{
    asm("mad.hi.u32 %0, %1, %2, %3;"
        : "=r"(hi)
        : "r"(a), "r"(b), "r"(c));
}

__device__ __forceinline__ void
mad_hi(u64 &hi, u64 a, u64 b, u64 c)
{
    asm("mad.hi.u64 %0, %1, %2, %3;"
        : "=l"(hi)
        : "l"(a), "l"(b), "l"(c));
}

__device__ __forceinline__ void
mad_hi_cc(u32 &hi, u32 a, u32 b, u32 c)
{
    asm("mad.hi.cc.u32 %0, %1, %2, %3;"
        : "=r"(hi)
        : "r"(a), "r"(b), "r"(c));
}

__device__ __forceinline__ void
mad_hi_cc(u64 &hi, u64 a, u64 b, u64 c)
{
    asm("mad.hi.cc.u64 %0, %1, %2, %3;"
        : "=l"(hi)
        : "l"(a), "l"(b), "l"(c));
}

__device__ __forceinline__ void
madc_hi_cc(u32 &hi, u32 a, u32 b, u32 c)
{
    asm("madc.hi.cc.u32 %0, %1, %2, %3;"
        : "=r"(hi)
        : "r"(a), "r"(b), "r"(c));
}

__device__ __forceinline__ void
madc_hi_cc(u64 &hi, u64 a, u64 b, u64 c)
{
    asm("madc.hi.cc.u64 %0, %1, %2, %3;\n\t"
        : "=l"(hi)
        : "l"(a), "l"(b), "l"(c));
}

// Source: https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-shf
__device__ __forceinline__ void
lshift(u32 &out_hi, u32 &out_lo, u32 in_hi, u32 in_lo, unsigned b)
{
    asm("shf.l.clamp.b32 %1, %2, %3, %4;\n\t"
        "shl.b32 %0, %2, %4;"
        : "=r"(out_lo), "=r"(out_hi)
        : "r"(in_lo), "r"(in_hi), "r"(b));
}

/*
     * Left shift by b bits; b <= 32.
     * Source: https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-shf
     */
__device__ __forceinline__ void
lshift_b32(u64 &out_hi, u64 &out_lo, u64 in_hi, u64 in_lo, unsigned b)
{
    assert(b <= 32);
    asm("{\n\t"
        " .reg .u32 t1;\n\t"
        " .reg .u32 t2;\n\t"
        " .reg .u32 t3;\n\t"
        " .reg .u32 t4;\n\t"
        " .reg .u32 t5;\n\t"
        " .reg .u32 t6;\n\t"
        " .reg .u32 t7;\n\t"
        " .reg .u32 t8;\n\t"
        // (t4, t3, t2, t1) = (in_hi, in_lo)
        " mov.b64 { t3, t4 }, %3;\n\t"
        " mov.b64 { t1, t2 }, %2;\n\t"
        " shf.l.clamp.b32 t8, t3, t4, %4;\n\t"
        " shf.l.clamp.b32 t7, t2, t3, %4;\n\t"
        " shf.l.clamp.b32 t6, t1, t2, %4;\n\t"
        " shl.b32 t5, t1, %4;\n\t"
        " mov.b64 %1, { t7, t8 };\n\t"
        " mov.b64 %0, { t5, t6 };\n\t"
        "}"
        : "=l"(out_lo), "=l"(out_hi)
        : "l"(in_lo), "l"(in_hi), "r"(b));
}

__device__ __forceinline__ void
lshift(u64 &out_hi, u64 &out_lo, u64 in_hi, u64 in_lo, unsigned b)
{
    assert(b <= 64);
    unsigned c = min(b, 32);
    lshift_b32(out_hi, out_lo, in_hi, in_lo, c);
    lshift_b32(out_hi, out_lo, out_hi, out_lo, b - c);
}

// Source: https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-shf
__device__ __forceinline__ void
rshift(u32 &out_hi, u32 &out_lo, u32 in_hi, u32 in_lo, unsigned b)
{
    asm("shf.r.clamp.b32 %0, %2, %3, %4;\n\t"
        "shr.b32 %1, %2, %4;"
        : "=r"(out_lo), "=r"(out_hi)
        : "r"(in_lo), "r"(in_hi), "r"(b));
}

/*
     * Right shift by b bits; b <= 32.
     * Source: https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-shf
     */
__device__ __forceinline__ void
rshift_b32(u64 &out_hi, u64 &out_lo, u64 in_hi, u64 in_lo, unsigned b)
{
    assert(b <= 32);
    asm("{\n\t"
        " .reg .u32 t1;\n\t"
        " .reg .u32 t2;\n\t"
        " .reg .u32 t3;\n\t"
        " .reg .u32 t4;\n\t"
        " .reg .u32 t5;\n\t"
        " .reg .u32 t6;\n\t"
        " .reg .u32 t7;\n\t"
        " .reg .u32 t8;\n\t"
        // (t4, t3, t2, t1) = (in_hi, in_lo)
        " mov.b64 { t1, t2 }, %2;\n\t"
        " mov.b64 { t3, t4 }, %3;\n\t"
        " shf.r.clamp.b32 t5, t1, t2, %4;\n\t"
        " shf.r.clamp.b32 t6, t2, t3, %4;\n\t"
        " shf.r.clamp.b32 t7, t3, t4, %4;\n\t"
        " shr.b32 t8, t4, %4;\n\t"
        " mov.b64 %0, { t5, t6 };\n\t"
        " mov.b64 %1, { t7, t8 };\n\t"
        "}"
        : "=l"(out_lo), "=l"(out_hi)
        : "l"(in_lo), "l"(in_hi), "r"(b));
}

__device__ __forceinline__ void
rshift(u64 &out_hi, u64 &out_lo, u64 in_hi, u64 in_lo, unsigned b)
{
    assert(b <= 64);
    unsigned c = min(b, 32);
    rshift_b32(out_hi, out_lo, in_hi, in_lo, c);
    rshift_b32(out_hi, out_lo, out_hi, out_lo, b - c);
}

/*
     * Count Leading Zeroes in x.
     */
__device__ __forceinline__ int
clz(u32 x)
{
    int n;
    asm("clz.b32 %0, %1;"
        : "=r"(n)
        : "r"(x));
    return n;
}

__device__ __forceinline__ int
clz(u64 x)
{
    int n;
    asm("clz.b64 %0, %1;"
        : "=r"(n)
        : "l"(x));
    return n;
}

/*
     * Count Trailing Zeroes in x.
     */
__device__ __forceinline__ int
ctz(u32 x)
{
    int n;
    asm("{\n\t"
        " .reg .u32 tmp;\n\t"
        " brev.b32 tmp, %1;\n\t"
        " clz.b32 %0, tmp;\n\t"
        "}"
        : "=r"(n)
        : "r"(x));
    return n;
}

__device__ __forceinline__ int
ctz(u64 x)
{
    int n;
    asm("{\n\t"
        " .reg .u64 tmp;\n\t"
        " brev.b64 tmp, %1;\n\t"
        " clz.b64 %0, tmp;\n\t"
        "}"
        : "=r"(n)
        : "l"(x));
    return n;
}

__device__ __forceinline__ void
min(u32 &m, u32 a, u32 b)
{
    asm("min.u32 %0, %1, %2;"
        : "=r"(m)
        : "r"(a), "r"(b));
}

__device__ __forceinline__ void
min(u64 &m, u64 a, u64 b)
{
    asm("min.u64 %0, %1, %2;"
        : "=l"(m)
        : "l"(a), "l"(b));
}

__device__ __forceinline__ void
max(u32 &m, u32 a, u32 b)
{
    asm("max.u32 %0, %1, %2;"
        : "=r"(m)
        : "r"(a), "r"(b));
}

__device__ __forceinline__ void
max(u64 &m, u64 a, u64 b)
{
    asm("max.u64 %0, %1, %2;"
        : "=l"(m)
        : "l"(a), "l"(b));
}

__device__ __forceinline__ void
modinv_2exp(u32 &x, u32 b)
{
    assert(b & 1);

    x = (2U - b * b) * b;
    x *= 2U - b * x;
    x *= 2U - b * x;
    x *= 2U - b * x;
}

__device__ __forceinline__ void
modinv_2exp(u64 &x, u64 b)
{
    assert(b & 1);

    x = (2UL - b * b) * b;
    x *= 2UL - b * x;
    x *= 2UL - b * x;
    x *= 2UL - b * x;
    x *= 2UL - b * x;
}

/*
     * For 512 <= d < 1024,
     *
     *   RECIPROCAL_TABLE_32[d - 512] = floor((2^24 - 2^14 + 2^9)/d)
     *
     * Total space at the moment is 512*2 = 1024 bytes.
     *
     * TODO: Investigate whether alternative storage layouts are better; examples:
     *
     *  - redundantly store each element in a uint32_t
     *  - pack two uint16_t values into each uint32_t
     *  - is __constant__ the right storage specifier? Maybe load into shared memory?
     *    Shared memory seems like an excellent choice (48k available per SM), though
     *    I'll need to be mindful of bank conflicts (perhaps circumvent by having
     *    many copies of the data in SM?).
     *  - perhaps reading an element from memory is slower than simply calculating
     *    floor((2^24 - 2^14 + 2^9)/d) in assembly?
     */
extern __device__ __constant__ uint16_t RECIPROCAL_TABLE_32[0x200];

__device__ __forceinline__
    uint32_t
    lookup_reciprocal(uint32_t d10)
{
    assert((d10 >> 9) == 1);
    return RECIPROCAL_TABLE_32[d10 - 0x200];
}

/*
     * Source: Niels Möller and Torbjörn Granlund, “Improved division by
     * invariant integers”, IEEE Transactions on Computers, 11 June
     * 2010. https://gmplib.org/~tege/division-paper.pdf
     */
__device__ __forceinline__
    uint32_t
    quorem_reciprocal(uint32_t d)
{
    // Top bit must be set, i.e. d must be already normalised.
    assert((d >> 31) == 1);

    uint32_t d0_mask, d10, d21, d31, v0, v1, v2, v3, e, t0, t1;

    d0_mask = -(uint32_t)(d & 1); // 0 if d&1=0, 0xFF..FF if d&1=1.
    d10 = d >> 22;
    d21 = (d >> 11) + 1;
    d31 = d - (d >> 1); // ceil(d/2) = d - floor(d/2)

    v0 = lookup_reciprocal(d10); // 15 bits
    mul_hi(t0, v0 * v0, d21);
    v1 = (v0 << 4) - t0 - 1; // 18 bits
    e = -(v1 * d31) + ((v1 >> 1) & d0_mask);
    mul_hi(t0, v1, e);
    v2 = (v1 << 15) + (t0 >> 1); // 33 bits (hi bit is implicit)
    mul_wide(t1, t0, v2, d);
    t1 += d + ((t0 + d) < d);
    v3 = v2 - t1; // 33 bits (hi bit is implicit)
    return v3;
}

/*
     * For 256 <= d < 512,
     *
     *   RECIPROCAL_TABLE_64[d - 256] = floor((2^19 - 3*2^9)/d)
     *
     * Total space ATM is 256*2 = 512 bytes. Entries range from 10 to 11
     * bits, so with some clever handling of hi bits, we could get three
     * entries per 32 bit word, reducing the size to about 256*11/8 = 352
     * bytes.
     *
     * TODO: Investigate whether alternative storage layouts are better;
     * see RECIPROCAL_TABLE_32 above for ideas.
     */
extern __device__ __constant__
    uint16_t
        RECIPROCAL_TABLE_64[0x100];

__device__ __forceinline__
    uint64_t
    lookup_reciprocal(uint64_t d9)
{
    assert((d9 >> 8) == 1);
    return RECIPROCAL_TABLE_64[d9 - 0x100];
}

/*
     * Source: Niels Möller and Torbjörn Granlund, “Improved division by
     * invariant integers”, IEEE Transactions on Computers, 11 June
     * 2010. https://gmplib.org/~tege/division-paper.pdf
     */
__device__ __forceinline__
    uint64_t
    quorem_reciprocal(uint64_t d)
{
    // Top bit must be set, i.e. d must be already normalised.
    assert((d >> 63) == 1);

    uint64_t d0_mask, d9, d40, d63, v0, v1, v2, v3, v4, e, t0, t1;

    d0_mask = -(uint64_t)(d & 1); // 0 if d&1=0, 0xFF..FF if d&1=1.
    d9 = d >> 55;
    d40 = (d >> 24) + 1;
    d63 = d - (d >> 1); // ceil(d/2) = d - floor(d/2)

    v0 = lookup_reciprocal(d9); // 11 bits
    t0 = v0 * v0 * d40;
    v1 = (v0 << 11) - (t0 >> 40) - 1; // 21 bits
    t0 = v1 * ((1UL << 60) - (v1 * d40));
    v2 = (v1 << 13) + (t0 >> 47); // 34 bits

    e = -(v2 * d63) + ((v1 >> 1) & d0_mask);
    mul_hi(t0, v2, e);
    v3 = (v2 << 31) + (t0 >> 1); // 65 bits (hi bit is implicit)
    mul_wide(t1, t0, v3, d);
    t1 += d + ((t0 + d) < d);
    v4 = v3 - t1; // 65 bits (hi bit is implicit)
    return v4;
}

template <typename uint_tp>
__device__ __forceinline__ int
quorem_normalise_divisor(uint_tp &d)
{
    int cnt = clz(d);
    d <<= cnt;
    return cnt;
}

template <typename uint_tp>
__device__ __forceinline__
    uint_tp
    quorem_normalise_dividend(uint_tp &u_hi, uint_tp &u_lo, int cnt)
{
    // TODO: For 32 bit operands we can just do the following
    // asm ("shf.l.clamp.b32 %0, %1, %0, %2;\n\t"
    //      "shl.b32 %1, %1, %2;"
    //     : "+r"(u_hi), "+r"(u_lo) : "r"(cnt));
    //
    // For 64 bits it's a bit more long-winded
    // Inspired by https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shf
    // asm ("{\n\t"
    //     " .reg .u32 t1;\n\t"
    //     " .reg .u32 t2;\n\t"
    //     " .reg .u32 t3;\n\t"
    //     " .reg .u32 t4;\n\t"
    //     " mov.b64 { t1, t2 }, %0;\n\t"
    //     " mov.b64 { t3, t4 }, %1;\n\t"
    //     " shf.l.clamp.b32 t4, t3, t4, %2;\n\t"
    //     " shf.l.clamp.b32 t3, t2, t3, %2;\n\t"
    //     " shf.l.clamp.b32 t2, t1, t2, %2;\n\t"
    //     " shl.b32 t1, t1, %2;\n\t"
    //     " mov.b64 %0, { t1, t2 };\n\t"
    //     " mov.b64 %1, { t3, t4 };\n\t"
    //     "}"
    //     : "+l"(u_lo), "+l"(u_hi) : "r"(cnt));

    static constexpr int WORD_BITS = sizeof(uint_tp) * 8;
    uint_tp overflow = u_hi >> (WORD_BITS - cnt);
    uint_tp u_hi_lsb = u_lo >> (WORD_BITS - cnt);
#ifndef __CUDA_ARCH__
    // Compensate for the fact that, unlike CUDA, shifts by WORD_BITS
    // are undefined in C.
    // u_hi_lsb = 0 if cnt=0 or u_hi_lsb if cnt!=0.
    u_hi_lsb &= -(uint_tp) !!cnt;
    overflow &= -(uint_tp) !!cnt;
#endif
    u_hi = (u_hi << cnt) | u_hi_lsb;
    u_lo <<= cnt;
    return overflow;
}

/*
     * Suppose Q and r satisfy U = Qd + r, where Q = (q_hi, q_lo) and U =
     * (u_hi, u_lo) are two-word numbers. This function returns q = min(Q,
     * 2^WORD_BITS - 1) and r = U - Qd if q = Q or r = q in the latter
     * case.  v should be set to quorem_reciprocal(d).
     *
     * CAVEAT EMPTOR: d and {u_hi, u_lo} need to be normalised (using the
     * functions provided) PRIOR to being passed to this
     * function. Similarly, the resulting remainder r (but NOT the
     * quotient q) needs to be denormalised (i.e. right shift by the
     * normalisation factor) after receipt.
     *
     * Source: Niels Möller and Torbjörn Granlund, “Improved division by
     * invariant integers”, IEEE Transactions on Computers, 11 June
     * 2010. https://gmplib.org/~tege/division-paper.pdf
     */
template <typename uint_tp>
__device__ void
quorem_wide_normalised(
    uint_tp &q, uint_tp &r,
    uint_tp u_hi, uint_tp u_lo, uint_tp d, uint_tp v)
{
    static_assert(std::is_unsigned<uint_tp>::value == true,
                  "template type must be unsigned");
    if (u_hi > d)
    {
        q = r = (uint_tp)-1;
        return;
    }

    uint_tp q_hi, q_lo, mask;

    mul_wide(q_hi, q_lo, u_hi, v);
    q_lo += u_lo;
    q_hi += u_hi + (q_lo < u_lo) + 1;
    r = u_lo - q_hi * d;

    // Branch is unpredicable
    //if (r > q_lo) { --q_hi; r += d; }
    mask = -(uint_tp)(r > q_lo);
    q_hi += mask;
    r += mask & d;

    // Branch is very unlikely to be taken
    if (r >= d)
    {
        r -= d;
        ++q_hi;
    }
    //mask = -(uint_tp)(r >= d);
    //q_hi -= mask;
    //r -= mask & d;

    q = q_hi;
}

/*
     * As above, but calculate, then return, the precomputed inverse for d.
     * Normalisation of the divisor and dividend is performed then thrown away.
     */
template <typename uint_tp>
__device__ __forceinline__
    uint_tp
    quorem_wide(
        uint_tp &q, uint_tp &r,
        uint_tp u_hi, uint_tp u_lo, uint_tp d)
{
    static_assert(std::is_unsigned<uint_tp>::value == true,
                  "template type must be unsigned");
    int lz = quorem_normalise_divisor(d);
    uint_tp overflow = quorem_normalise_dividend(u_hi, u_lo, lz);
    uint_tp v = quorem_reciprocal(d);
    if (overflow)
    {
        q = r = (uint_tp)-1;
        return v;
    }
    quorem_wide_normalised(q, r, u_hi, u_lo, d, v);
    assert((r & (((uint_tp)1 << lz) - 1U)) == 0);
    r >>= lz;
    return v;
}

/*
     * As above, but uses a given precomputed inverse. If the precomputed
     * inverse comes from quorem_reciprocal() rather than from quorem_wide()
     * above, then make sure the divisor given to quorem_reciprocal() was
     * normalised with quorem_normalise_divisor() first.
     */
template <typename uint_tp>
__device__ __forceinline__ void
quorem_wide(
    uint_tp &q, uint_tp &r,
    uint_tp u_hi, uint_tp u_lo, uint_tp d, uint_tp v)
{
    static_assert(std::is_unsigned<uint_tp>::value == true,
                  "template type must be unsigned");
    int lz = quorem_normalise_divisor(d);
    uint_tp overflow = quorem_normalise_dividend(u_hi, u_lo, lz);
    if (overflow)
    {
        q = r = -(uint_tp)1;
    }
    quorem_wide_normalised(q, r, u_hi, u_lo, d, v);
    assert((r & (((uint_tp)1 << lz) - 1U)) == 0);
    r >>= lz;
}

/*
     * ceiling(n / d)
     */
template <typename T>
__device__ __forceinline__ void
ceilquo(T &q, T n, T d)
{
    q = (n + d - 1) / d;
}

} // End namespace internal

} // End namespace cuFIXNUM
