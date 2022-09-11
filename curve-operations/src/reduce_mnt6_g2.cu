#include <cstring>

#include "reduce.hpp"
#include "constants.hpp"

#include "retrieve_utils.cuh"
#include "gpu_constants.cuh"
#include "gpu_fq.cuh"
#include "gpu_fq3.cuh"
#include "gpu_g2.cuh"
#include "gpu_params.cuh"

template <typename fixnum>
struct reduce_mnt6_g2_gpu
{
  __device__ void operator()(fixnum &r0c0, fixnum &r0c1, fixnum &r0c2, fixnum &r1c0, fixnum &r1c1, fixnum &r1c2, fixnum &r2c0, fixnum &r2c1, fixnum &r2c2, fixnum a0c0, fixnum a0c1, fixnum a0c2, fixnum a1c0, fixnum a1c1, fixnum a1c2, fixnum a2c0, fixnum a2c1, fixnum a2c2, fixnum b0c0, fixnum b0c1, fixnum b0c2, fixnum b1c0, fixnum b1c1, fixnum b1c2, fixnum b2c0, fixnum b2c1, fixnum b2c2) const
  {
    GpuParams &params = get_gpu_params();
    modnum mod = params.get_mnt_mod();
    GpuFq3 coeff_a3 = params.get_mnt_coeff_a3();
    GpuFq non_residue = params.get_mnt_non_residue();

    GpuG2<GpuFq3> gA = GpuG2<GpuFq3>(GpuFq3(GpuFq::load(a0c0, mod), GpuFq::load(a0c1, mod), GpuFq::load(a0c2, mod), non_residue), GpuFq3(GpuFq::load(a1c0, mod), GpuFq::load(a1c1, mod), GpuFq::load(a1c2, mod), non_residue), GpuFq3(GpuFq::load(a2c0, mod), GpuFq::load(a2c1, mod), GpuFq::load(a2c2, mod), non_residue), coeff_a3);
    GpuG2<GpuFq3> gB = GpuG2<GpuFq3>(GpuFq3(GpuFq::load(b0c0, mod), GpuFq::load(b0c1, mod), GpuFq::load(b0c2, mod), non_residue), GpuFq3(GpuFq::load(b1c0, mod), GpuFq::load(b1c1, mod), GpuFq::load(b1c2, mod), non_residue), GpuFq3(GpuFq::load(b2c0, mod), GpuFq::load(b2c1, mod), GpuFq::load(b2c2, mod), non_residue), coeff_a3);
    GpuG2<GpuFq3> fqS = gA + gB;

    fqS.X.save(r0c0, r0c1, r0c2);
    fqS.Y.save(r1c0, r1c1, r1c2);
    fqS.Z.save(r2c0, r2c1, r2c2);
  }
};

uint8_t *reduce_mnt6_g2_internal(my_fixnum_array *r0c0, my_fixnum_array *r0c1, my_fixnum_array *r0c2,
                                 my_fixnum_array *r1c0, my_fixnum_array *r1c1, my_fixnum_array *r1c2,
                                 my_fixnum_array *r2c0, my_fixnum_array *r2c1, my_fixnum_array *r2c2,
                                 my_fixnum_array *a0c0, my_fixnum_array *a0c1, my_fixnum_array *a0c2,
                                 my_fixnum_array *a1c0, my_fixnum_array *a1c1, my_fixnum_array *a1c2,
                                 my_fixnum_array *a2c0, my_fixnum_array *a2c1, my_fixnum_array *a2c2,
                                 my_fixnum_array *b0c0, my_fixnum_array *b0c1, my_fixnum_array *b0c2,
                                 my_fixnum_array *b1c0, my_fixnum_array *b1c1, my_fixnum_array *b1c2,
                                 my_fixnum_array *b2c0, my_fixnum_array *b2c1, my_fixnum_array *b2c2,
                                 int nelts)
{
  my_fixnum_array::template map<reduce_mnt6_g2_gpu>(r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2, a0c0, a0c1, a0c2, a1c0, a1c1, a1c2, a2c0, a2c1, a2c2, b0c0, b0c1, b0c2, b1c0, b1c1, b1c2, b2c0, b2c1, b2c2);

  if (nelts == 1)
  {
    uint8_t *result = new uint8_t[3 * 3 * bytes_per_elem];
    r0c0->retrieve_into(result, bytes_per_elem, 0);
    r0c1->retrieve_into(result + bytes_per_elem, bytes_per_elem, 0);
    r0c2->retrieve_into(result + 2 * bytes_per_elem, bytes_per_elem, 0);
    r1c0->retrieve_into(result + 3 * bytes_per_elem, bytes_per_elem, 0);
    r1c1->retrieve_into(result + 4 * bytes_per_elem, bytes_per_elem, 0);
    r1c2->retrieve_into(result + 5 * bytes_per_elem, bytes_per_elem, 0);
    r2c0->retrieve_into(result + 6 * bytes_per_elem, bytes_per_elem, 0);
    r2c1->retrieve_into(result + 7 * bytes_per_elem, bytes_per_elem, 0);
    r2c2->retrieve_into(result + 8 * bytes_per_elem, bytes_per_elem, 0);
    return result;
  }
  else
  {
    // next round
    // input is previous result divided into two halves
    my_fixnum_array *new_a0c0 = my_fixnum_array::wrap(r0c0->get_ptr(), nelts / 2);
    my_fixnum_array *new_a0c1 = my_fixnum_array::wrap(r0c1->get_ptr(), nelts / 2);
    my_fixnum_array *new_a0c2 = my_fixnum_array::wrap(r0c2->get_ptr(), nelts / 2);
    my_fixnum_array *new_b0c0 = my_fixnum_array::wrap(r0c0->get_ptr() + bytes_per_elem * nelts / 2, nelts / 2);
    my_fixnum_array *new_b0c1 = my_fixnum_array::wrap(r0c1->get_ptr() + bytes_per_elem * nelts / 2, nelts / 2);
    my_fixnum_array *new_b0c2 = my_fixnum_array::wrap(r0c2->get_ptr() + bytes_per_elem * nelts / 2, nelts / 2);
    my_fixnum_array *new_a1c0 = my_fixnum_array::wrap(r1c0->get_ptr(), nelts / 2);
    my_fixnum_array *new_a1c1 = my_fixnum_array::wrap(r1c1->get_ptr(), nelts / 2);
    my_fixnum_array *new_a1c2 = my_fixnum_array::wrap(r1c2->get_ptr(), nelts / 2);
    my_fixnum_array *new_b1c0 = my_fixnum_array::wrap(r1c0->get_ptr() + bytes_per_elem * nelts / 2, nelts / 2);
    my_fixnum_array *new_b1c1 = my_fixnum_array::wrap(r1c1->get_ptr() + bytes_per_elem * nelts / 2, nelts / 2);
    my_fixnum_array *new_b1c2 = my_fixnum_array::wrap(r1c2->get_ptr() + bytes_per_elem * nelts / 2, nelts / 2);
    my_fixnum_array *new_a2c0 = my_fixnum_array::wrap(r2c0->get_ptr(), nelts / 2);
    my_fixnum_array *new_a2c1 = my_fixnum_array::wrap(r2c1->get_ptr(), nelts / 2);
    my_fixnum_array *new_a2c2 = my_fixnum_array::wrap(r2c2->get_ptr(), nelts / 2);
    my_fixnum_array *new_b2c0 = my_fixnum_array::wrap(r2c0->get_ptr() + bytes_per_elem * nelts / 2, nelts / 2);
    my_fixnum_array *new_b2c1 = my_fixnum_array::wrap(r2c1->get_ptr() + bytes_per_elem * nelts / 2, nelts / 2);
    my_fixnum_array *new_b2c2 = my_fixnum_array::wrap(r2c2->get_ptr() + bytes_per_elem * nelts / 2, nelts / 2);

    // output is reused previous input
    my_fixnum_array *new_r0c0 = my_fixnum_array::wrap(b0c0->get_ptr(), nelts / 2);
    my_fixnum_array *new_r0c1 = my_fixnum_array::wrap(b0c1->get_ptr(), nelts / 2);
    my_fixnum_array *new_r0c2 = my_fixnum_array::wrap(b0c2->get_ptr(), nelts / 2);
    my_fixnum_array *new_r1c0 = my_fixnum_array::wrap(b1c0->get_ptr(), nelts / 2);
    my_fixnum_array *new_r1c1 = my_fixnum_array::wrap(b1c1->get_ptr(), nelts / 2);
    my_fixnum_array *new_r1c2 = my_fixnum_array::wrap(b1c2->get_ptr(), nelts / 2);
    my_fixnum_array *new_r2c0 = my_fixnum_array::wrap(b2c0->get_ptr(), nelts / 2);
    my_fixnum_array *new_r2c1 = my_fixnum_array::wrap(b2c1->get_ptr(), nelts / 2);
    my_fixnum_array *new_r2c2 = my_fixnum_array::wrap(b2c2->get_ptr(), nelts / 2);

    return reduce_mnt6_g2_internal(new_r0c0, new_r0c1, new_r0c2,
                                   new_r1c0, new_r1c1, new_r1c2,
                                   new_r2c0, new_r2c1, new_r2c2,
                                   new_a0c0, new_a0c1, new_a0c2,
                                   new_a1c0, new_a1c1, new_a1c2,
                                   new_a2c0, new_a2c1, new_a2c2,
                                   new_b0c0, new_b0c1, new_b0c2,
                                   new_b1c0, new_b1c1, new_b1c2,
                                   new_b2c0, new_b2c1, new_b2c2,
                                   nelts / 2);
  }
}

uint8_t *reduce_mnt6_g2(uint8_t *a, int nelts)
{
  uint8_t *input_a0c0 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_a0c1 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_a0c2 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_a1c0 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_a1c1 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_a1c2 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_a2c0 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_a2c1 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_a2c2 = new uint8_t[bytes_per_elem * nelts / 2];

  for (int i = 0; i < nelts / 2; i++)
  {
    memcpy(input_a0c0 + i * bytes_per_elem, a + 9 * i * bytes_per_elem, bytes_per_elem);
    memcpy(input_a0c1 + i * bytes_per_elem, a + 9 * i * bytes_per_elem + bytes_per_elem, bytes_per_elem);
    memcpy(input_a0c2 + i * bytes_per_elem, a + 9 * i * bytes_per_elem + 2 * bytes_per_elem, bytes_per_elem);
    memcpy(input_a1c0 + i * bytes_per_elem, a + 9 * i * bytes_per_elem + 3 * bytes_per_elem, bytes_per_elem);
    memcpy(input_a1c1 + i * bytes_per_elem, a + 9 * i * bytes_per_elem + 4 * bytes_per_elem, bytes_per_elem);
    memcpy(input_a1c2 + i * bytes_per_elem, a + 9 * i * bytes_per_elem + 5 * bytes_per_elem, bytes_per_elem);
    memcpy(input_a2c0 + i * bytes_per_elem, a + 9 * i * bytes_per_elem + 6 * bytes_per_elem, bytes_per_elem);
    memcpy(input_a2c1 + i * bytes_per_elem, a + 9 * i * bytes_per_elem + 7 * bytes_per_elem, bytes_per_elem);
    memcpy(input_a2c2 + i * bytes_per_elem, a + 9 * i * bytes_per_elem + 8 * bytes_per_elem, bytes_per_elem);
  }

  uint8_t *input_b0c0 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_b0c1 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_b0c2 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_b1c0 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_b1c1 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_b1c2 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_b2c0 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_b2c1 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_b2c2 = new uint8_t[bytes_per_elem * nelts / 2];

  for (int i = 0; i < nelts / 2; i++)
  {
    memcpy(input_b0c0 + i * bytes_per_elem, a + 9 * (i + nelts / 2) * bytes_per_elem, bytes_per_elem);
    memcpy(input_b0c1 + i * bytes_per_elem, a + 9 * (i + nelts / 2) * bytes_per_elem + bytes_per_elem, bytes_per_elem);
    memcpy(input_b0c2 + i * bytes_per_elem, a + 9 * (i + nelts / 2) * bytes_per_elem + 2 * bytes_per_elem, bytes_per_elem);
    memcpy(input_b1c0 + i * bytes_per_elem, a + 9 * (i + nelts / 2) * bytes_per_elem + 3 * bytes_per_elem, bytes_per_elem);
    memcpy(input_b1c1 + i * bytes_per_elem, a + 9 * (i + nelts / 2) * bytes_per_elem + 4 * bytes_per_elem, bytes_per_elem);
    memcpy(input_b1c2 + i * bytes_per_elem, a + 9 * (i + nelts / 2) * bytes_per_elem + 5 * bytes_per_elem, bytes_per_elem);
    memcpy(input_b2c0 + i * bytes_per_elem, a + 9 * (i + nelts / 2) * bytes_per_elem + 6 * bytes_per_elem, bytes_per_elem);
    memcpy(input_b2c1 + i * bytes_per_elem, a + 9 * (i + nelts / 2) * bytes_per_elem + 7 * bytes_per_elem, bytes_per_elem);
    memcpy(input_b2c2 + i * bytes_per_elem, a + 9 * (i + nelts / 2) * bytes_per_elem + 8 * bytes_per_elem, bytes_per_elem);
  }

  my_fixnum_array *a0c0 = my_fixnum_array::create(input_a0c0, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *a0c1 = my_fixnum_array::create(input_a0c1, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *a0c2 = my_fixnum_array::create(input_a0c2, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *a1c0 = my_fixnum_array::create(input_a1c0, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *a1c1 = my_fixnum_array::create(input_a1c1, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *a1c2 = my_fixnum_array::create(input_a1c2, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *a2c0 = my_fixnum_array::create(input_a2c0, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *a2c1 = my_fixnum_array::create(input_a2c1, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *a2c2 = my_fixnum_array::create(input_a2c2, bytes_per_elem * nelts / 2, bytes_per_elem);

  my_fixnum_array *b0c0 = my_fixnum_array::create(input_b0c0, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *b0c1 = my_fixnum_array::create(input_b0c1, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *b0c2 = my_fixnum_array::create(input_b0c2, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *b1c0 = my_fixnum_array::create(input_b1c0, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *b1c1 = my_fixnum_array::create(input_b1c1, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *b1c2 = my_fixnum_array::create(input_b1c2, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *b2c0 = my_fixnum_array::create(input_b2c0, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *b2c1 = my_fixnum_array::create(input_b2c1, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *b2c2 = my_fixnum_array::create(input_b2c2, bytes_per_elem * nelts / 2, bytes_per_elem);

  my_fixnum_array *r0c0 = my_fixnum_array::create(nelts / 2);
  my_fixnum_array *r0c1 = my_fixnum_array::create(nelts / 2);
  my_fixnum_array *r0c2 = my_fixnum_array::create(nelts / 2);
  my_fixnum_array *r1c0 = my_fixnum_array::create(nelts / 2);
  my_fixnum_array *r1c1 = my_fixnum_array::create(nelts / 2);
  my_fixnum_array *r1c2 = my_fixnum_array::create(nelts / 2);
  my_fixnum_array *r2c0 = my_fixnum_array::create(nelts / 2);
  my_fixnum_array *r2c1 = my_fixnum_array::create(nelts / 2);
  my_fixnum_array *r2c2 = my_fixnum_array::create(nelts / 2);

  uint8_t *v_res = reduce_mnt6_g2_internal(r0c0, r0c1, r0c2,
                                           r1c0, r1c1, r1c2,
                                           r2c0, r2c1, r2c2,
                                           a0c0, a0c1, a0c2,
                                           a1c0, a1c1, a1c2,
                                           a2c0, a2c1, a2c2,
                                           b0c0, b0c1, b0c2,
                                           b1c0, b1c1, b1c2,
                                           b2c0, b2c1, b2c2,
                                           nelts / 2);

  delete a0c0;
  delete a0c1;
  delete a0c2;
  delete a1c0;
  delete a1c1;
  delete a1c2;
  delete a2c0;
  delete a2c1;
  delete a2c2;
  delete b0c0;
  delete b0c1;
  delete b0c2;
  delete b1c0;
  delete b1c1;
  delete b1c2;
  delete b2c0;
  delete b2c1;
  delete b2c2;
  delete r0c0;
  delete r0c1;
  delete r0c2;
  delete r1c0;
  delete r1c1;
  delete r1c2;
  delete r2c0;
  delete r2c1;
  delete r2c2;
  delete[] input_a0c0;
  delete[] input_a0c1;
  delete[] input_a0c2;
  delete[] input_a1c0;
  delete[] input_a1c1;
  delete[] input_a1c2;
  delete[] input_a2c0;
  delete[] input_a2c1;
  delete[] input_a2c2;
  delete[] input_b0c0;
  delete[] input_b0c1;
  delete[] input_b0c2;
  delete[] input_b1c0;
  delete[] input_b1c1;
  delete[] input_b1c2;
  delete[] input_b2c0;
  delete[] input_b2c1;
  delete[] input_b2c2;
  return v_res;
}
