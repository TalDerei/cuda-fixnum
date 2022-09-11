#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>

#include "io.hpp"
#include "constants.hpp"

using namespace libff;
using namespace std;

void read_mnt_fq(uint8_t *dest, FILE *inputs)
{
  fread((void *)(dest), io_bytes_per_elem * sizeof(uint8_t), 1, inputs);
}

template <typename curve>
void read_mnt_fq_montgomery(uint8_t *dest, FILE *inputs)
{
  // cout << "entered read_mnt_fq_montgomery " << endl;
  Fq<curve> x;
  fread((void *)(x.mont_repr.data), io_bytes_per_elem * sizeof(uint8_t), 1, inputs);
  // cout << "x is: " << x << endl;
  memcpy(dest, (uint8_t *)x.as_bigint().data, io_bytes_per_elem);
}

void read_mnt4_fq_montgomery(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq_montgomery<mnt4753_pp>(dest, inputs);
}

void read_mnt6_fq_montgomery(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq_montgomery<mnt6753_pp>(dest, inputs);
}

template <typename curve>
void read_mnt_fq2_montgomery(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq_montgomery<curve>(dest, inputs);
  read_mnt_fq_montgomery<curve>(dest + bytes_per_elem, inputs);
}

template <typename curve>
void read_mnt_fq3_montgomery(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq_montgomery<curve>(dest, inputs);
  read_mnt_fq_montgomery<curve>(dest + bytes_per_elem, inputs);
  read_mnt_fq_montgomery<curve>(dest + 2 * bytes_per_elem, inputs);
}

void read_mnt_fq2(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq(dest, inputs);
  read_mnt_fq(dest + bytes_per_elem, inputs);
}

void read_mnt_fq3(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq(dest, inputs);
  read_mnt_fq(dest + bytes_per_elem, inputs);
  read_mnt_fq(dest + 2 * bytes_per_elem, inputs);
}

template <typename curve>
void read_mnt_g1_montgomery(uint8_t *dest, FILE *inputs)
{
  // cout << "entered read_mnt_g1_montgomery " << endl;
  // Need to call read_mnt_fq_montgomery twice to read in two elements from the Fq field
  read_mnt_fq_montgomery<curve>(dest, inputs);
  read_mnt_fq_montgomery<curve>(dest + bytes_per_elem, inputs);
  // Then we're copying 1 for our result array
  memcpy(dest + 2 * bytes_per_elem, (void *)Fq<mnt4753_pp>::one().as_bigint().data, io_bytes_per_elem);
  // cout << "Fq<mnt4753_pp>::one() is " << Fq<mnt4753_pp>::one().as_bigint() << endl;
}

void read_mnt4_g1_montgomery(uint8_t *dest, FILE *inputs)
{
  read_mnt_g1_montgomery<mnt4753_pp>(dest, inputs);
}

void read_mnt6_g1_montgomery(uint8_t *dest, FILE *inputs)
{
  read_mnt_g1_montgomery<mnt6753_pp>(dest, inputs);
}

void read_mnt4_g2_montgomery(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq2_montgomery<mnt4753_pp>(dest, inputs);
  read_mnt_fq2_montgomery<mnt4753_pp>(dest + 2 * bytes_per_elem, inputs);

  memcpy(dest + 4 * bytes_per_elem, (void *)Fq<mnt4753_pp>::one().as_bigint().data, io_bytes_per_elem);
  memcpy(dest + 5 * bytes_per_elem, (void *)Fq<mnt4753_pp>::zero().as_bigint().data, io_bytes_per_elem);
}

void read_mnt6_g2_montgomery(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq3_montgomery<mnt6753_pp>(dest, inputs);
  read_mnt_fq3_montgomery<mnt6753_pp>(dest + 3 * bytes_per_elem, inputs);

  memcpy(dest + 6 * bytes_per_elem, (void *)Fq<mnt6753_pp>::one().as_bigint().data, io_bytes_per_elem);
  memcpy(dest + 7 * bytes_per_elem, (void *)Fq<mnt6753_pp>::zero().as_bigint().data, io_bytes_per_elem);
  memcpy(dest + 8 * bytes_per_elem, (void *)Fq<mnt6753_pp>::zero().as_bigint().data, io_bytes_per_elem);
}

void read_mnt4_g2(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq2(dest, inputs);
  read_mnt_fq2(dest + 2 * bytes_per_elem, inputs);
}

void read_mnt6_g2(uint8_t *dest, FILE *inputs)
{
  read_mnt_fq3(dest, inputs);
  read_mnt_fq3(dest + 3 * bytes_per_elem, inputs);
}

void write_mnt_fq(uint8_t *fq, FILE *outputs)
{
  fwrite((void *)fq, io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
}

template <typename curve>
void write_mnt_fq_montgomery(uint8_t *fq, FILE *outputs)
{
  Fq<curve> x;
  memcpy((void *)x.mont_repr.data, fq, io_bytes_per_elem);

  Fq<curve> result = Fq<curve>(x.mont_repr);
  fwrite((void *)result.mont_repr.data, io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
}

void write_mnt4_fq_montgomery(uint8_t *fq, FILE *outputs)
{
  write_mnt_fq_montgomery<mnt4753_pp>(fq, outputs);
}

void write_mnt6_fq_montgomery(uint8_t *fq, FILE *outputs)
{
  write_mnt_fq_montgomery<mnt6753_pp>(fq, outputs);
}

template <typename curve>
void write_mnt_fq2_montgomery(uint8_t *src, FILE *outputs)
{
  write_mnt_fq_montgomery<curve>(src, outputs);
  write_mnt_fq_montgomery<curve>(src + bytes_per_elem, outputs);
}

template <typename curve>
void write_mnt_fq3_montgomery(uint8_t *src, FILE *outputs)
{
  write_mnt_fq_montgomery<curve>(src, outputs);
  write_mnt_fq_montgomery<curve>(src + bytes_per_elem, outputs);
  write_mnt_fq_montgomery<curve>(src + 2 * bytes_per_elem, outputs);
}

void write_mnt4_fq2_montgomery(uint8_t *src, FILE *outputs)
{
  write_mnt_fq2_montgomery<mnt4753_pp>(src, outputs);
}

void write_mnt6_fq3_montgomery(uint8_t *src, FILE *outputs)
{
  write_mnt_fq3_montgomery<mnt6753_pp>(src, outputs);
}

void write_mnt_fq2(uint8_t *src, FILE *outputs)
{
  write_mnt_fq(src, outputs);
  write_mnt_fq(src + bytes_per_elem, outputs);
}

void write_mnt_fq3(uint8_t *src, FILE *outputs)
{
  write_mnt_fq(src, outputs);
  write_mnt_fq(src + bytes_per_elem, outputs);
  write_mnt_fq(src + 2 * bytes_per_elem, outputs);
}

template <typename curve>
void write_mnt_g1_montgomery(uint8_t *src, FILE *outputs)
{
  write_mnt_fq_montgomery<curve>(src, outputs);
  write_mnt_fq_montgomery<curve>(src + bytes_per_elem, outputs);
}

void write_mnt4_g1_montgomery(uint8_t *src, FILE *outputs)
{
  write_mnt_g1_montgomery<mnt4753_pp>(src, outputs);
}

void write_mnt6_g1_montgomery(uint8_t *src, FILE *outputs)
{
  write_mnt_g1_montgomery<mnt6753_pp>(src, outputs);
}

void write_mnt4_g2_montgomery(uint8_t *src, FILE *outputs)
{
  write_mnt4_fq2_montgomery(src, outputs);
  write_mnt4_fq2_montgomery(src + 2 * bytes_per_elem, outputs);
}

void write_mnt6_g2_montgomery(uint8_t *src, FILE *outputs)
{
  write_mnt6_fq3_montgomery(src, outputs);
  write_mnt6_fq3_montgomery(src + 3 * bytes_per_elem, outputs);
}

void write_mnt4_g2(uint8_t *src, FILE *outputs)
{
  write_mnt_fq2(src, outputs);
  write_mnt_fq2(src + 2 * bytes_per_elem, outputs);
}

void write_mnt6_g2(uint8_t *src, FILE *outputs)
{
  write_mnt_fq3(src, outputs);
  write_mnt_fq3(src + 3 * bytes_per_elem, outputs);
}

// LIBFF reading montgomery and numeral representations from array
template <typename curve>
Fq<curve> libff_read_mnt_fq_numeral(uint8_t *src)
{
  // bigint<mnt4753_q_limbs> n;
  Fq<curve> x;
  memcpy((void *)x.mont_repr.data, src, 12 * sizeof(mp_size_t));
  return Fq<curve>(x.mont_repr);
}

Fqe<mnt4753_pp> libff_read_mnt4_fq2_numeral(uint8_t *src)
{
  Fq<mnt4753_pp> c0 = libff_read_mnt_fq_numeral<mnt4753_pp>(src);
  Fq<mnt4753_pp> c1 = libff_read_mnt_fq_numeral<mnt4753_pp>(src + bytes_per_elem);
  return Fqe<mnt4753_pp>(c0, c1);
}

Fqe<mnt6753_pp> libff_read_mnt6_fq3_numeral(uint8_t *src)
{
  Fq<mnt6753_pp> c0 = libff_read_mnt_fq_numeral<mnt6753_pp>(src);
  Fq<mnt6753_pp> c1 = libff_read_mnt_fq_numeral<mnt6753_pp>(src + bytes_per_elem);
  Fq<mnt6753_pp> c2 = libff_read_mnt_fq_numeral<mnt6753_pp>(src + 2 * bytes_per_elem);
  return Fqe<mnt6753_pp>(c0, c1, c2);
}

template <typename curve>
G1<curve> libff_read_mnt_g1_numeral(uint8_t *src)
{
  Fq<curve> x = libff_read_mnt_fq_numeral<curve>(src);
  Fq<curve> y = libff_read_mnt_fq_numeral<curve>(src + bytes_per_elem);
  Fq<curve> z = libff_read_mnt_fq_numeral<curve>(src + 2 * bytes_per_elem);
  return G1<curve>(x, y, z);
}

G2<mnt4753_pp> libff_read_mnt4_g2_numeral(uint8_t *src)
{
  Fqe<mnt4753_pp> x = libff_read_mnt4_fq2_numeral(src);
  Fqe<mnt4753_pp> y = libff_read_mnt4_fq2_numeral(src + 2 * bytes_per_elem);
  Fqe<mnt4753_pp> z = libff_read_mnt4_fq2_numeral(src + 4 * bytes_per_elem);
  return G2<mnt4753_pp>(x, y, z);
}

G2<mnt6753_pp> libff_read_mnt6_g2_numeral(uint8_t *src)
{
  Fqe<mnt6753_pp> x = libff_read_mnt6_fq3_numeral(src);
  Fqe<mnt6753_pp> y = libff_read_mnt6_fq3_numeral(src + 3 * bytes_per_elem);
  Fqe<mnt6753_pp> z = libff_read_mnt6_fq3_numeral(src + 6 * bytes_per_elem);
  return G2<mnt6753_pp>(x, y, z);
}

void init_libff()
{
  mnt4753_pp::init_public_params();
  mnt6753_pp::init_public_params();
}

template <typename curve>
uint8_t *mnt_g1_to_affine(uint8_t *src)
{
  G1<curve> g = libff_read_mnt_g1_numeral<curve>(src);
  g.to_affine_coordinates();

  uint8_t *dst = new uint8_t[3 * bytes_per_elem];
  memset(dst, 0, 3 * bytes_per_elem);

  memcpy(dst, (uint8_t *)g.X().as_bigint().data, io_bytes_per_elem);
  memcpy(dst + bytes_per_elem, (uint8_t *)g.Y().as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 2 * bytes_per_elem, (uint8_t *)g.Z().as_bigint().data, io_bytes_per_elem);

  return dst;
}

uint8_t *mnt4_g1_to_affine(uint8_t *src)
{
  return mnt_g1_to_affine<mnt4753_pp>(src);
}

uint8_t *mnt6_g1_to_affine(uint8_t *src)
{
  return mnt_g1_to_affine<mnt6753_pp>(src);
}

uint8_t *mnt4_g2_to_affine(uint8_t *src)
{
  G2<mnt4753_pp> g = libff_read_mnt4_g2_numeral(src);
  g.to_affine_coordinates();

  uint8_t *dst = new uint8_t[6 * bytes_per_elem];
  memset(dst, 0, 6 * bytes_per_elem);

  memcpy(dst, (uint8_t *)g.X().c0.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + bytes_per_elem, (uint8_t *)g.X().c1.as_bigint().data, io_bytes_per_elem);

  memcpy(dst + 2 * bytes_per_elem, (uint8_t *)g.Y().c0.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 3 * bytes_per_elem, (uint8_t *)g.Y().c1.as_bigint().data, io_bytes_per_elem);

  memcpy(dst + 4 * bytes_per_elem, (uint8_t *)g.Z().c0.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 5 * bytes_per_elem, (uint8_t *)g.Z().c1.as_bigint().data, io_bytes_per_elem);

  return dst;
}

uint8_t *mnt6_g2_to_affine(uint8_t *src)
{
  G2<mnt6753_pp> g = libff_read_mnt6_g2_numeral(src);
  g.to_affine_coordinates();

  uint8_t *dst = new uint8_t[9 * bytes_per_elem];
  memset(dst, 0, 9 * bytes_per_elem);

  memcpy(dst, (uint8_t *)g.X().c0.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + bytes_per_elem, (uint8_t *)g.X().c1.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 2 * bytes_per_elem, (uint8_t *)g.X().c2.as_bigint().data, io_bytes_per_elem);

  memcpy(dst + 3 * bytes_per_elem, (uint8_t *)g.Y().c0.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 4 * bytes_per_elem, (uint8_t *)g.Y().c1.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 5 * bytes_per_elem, (uint8_t *)g.Y().c2.as_bigint().data, io_bytes_per_elem);

  memcpy(dst + 6 * bytes_per_elem, (uint8_t *)g.Z().c0.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 7 * bytes_per_elem, (uint8_t *)g.Z().c1.as_bigint().data, io_bytes_per_elem);
  memcpy(dst + 8 * bytes_per_elem, (uint8_t *)g.Z().c2.as_bigint().data, io_bytes_per_elem);

  return dst;
}
