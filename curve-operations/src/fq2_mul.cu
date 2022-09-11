#include "fq_mul.hpp"
#include "constants.hpp"

#include "retrieve_utils.cuh"
#include "gpu_constants.cuh"
#include "gpu_fq.cuh"
#include "gpu_fq2.cuh"
#include "gpu_params.cuh"

template <typename fixnum>
struct fq2_mul_gpu
{
  __device__ void operator()(fixnum &r0, fixnum &r1, fixnum a0, fixnum a1, fixnum b0, fixnum b1)
  {
    GpuParams &params = get_gpu_params();
    modnum mod = params.get_mnt_mod();
    GpuFq non_residue = params.get_mnt_non_residue();

    GpuFq2 fqA = GpuFq2(GpuFq::load(a0, mod), GpuFq::load(a1, mod), non_residue);
    GpuFq2 fqB = GpuFq2(GpuFq::load(b0, mod), GpuFq::load(b1, mod), non_residue);
    GpuFq2 fqS = fqA * fqB;
    fqS.save(r0, r1);
  }
};

uint8_t *fq2_mul(uint8_t *a, uint8_t *b, int nelts)
{
  uint8_t *input_a0 = new uint8_t[bytes_per_elem * nelts];
  uint8_t *input_a1 = new uint8_t[bytes_per_elem * nelts];

  for (int i = 0; i < nelts; i++)
  {
    mempcpy(input_a0 + i * bytes_per_elem, a + 2 * i * bytes_per_elem, bytes_per_elem);
    mempcpy(input_a1 + i * bytes_per_elem, a + 2 * i * bytes_per_elem + bytes_per_elem, bytes_per_elem);
  }

  uint8_t *input_b0 = new uint8_t[bytes_per_elem * nelts];
  uint8_t *input_b1 = new uint8_t[bytes_per_elem * nelts];

  for (int i = 0; i < nelts; i++)
  {
    mempcpy(input_b0 + i * bytes_per_elem, b + 2 * i * bytes_per_elem, bytes_per_elem);
    mempcpy(input_b1 + i * bytes_per_elem, b + 2 * i * bytes_per_elem + bytes_per_elem, bytes_per_elem);
  }

  my_fixnum_array *in_a0 = my_fixnum_array::create(input_a0, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_a1 = my_fixnum_array::create(input_a1, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_b0 = my_fixnum_array::create(input_b0, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_b1 = my_fixnum_array::create(input_b1, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *res0 = my_fixnum_array::create(nelts);
  my_fixnum_array *res1 = my_fixnum_array::create(nelts);

  my_fixnum_array::template map<fq2_mul_gpu>(res0, res1, in_a0, in_a1, in_b0, in_b1);

  uint8_t *v_res = get_2D_fixnum_array(res0, res1, nelts);

  delete in_a0;
  delete in_a1;
  delete in_b0;
  delete in_b1;
  delete res0;
  delete res1;
  delete[] input_a0;
  delete[] input_a1;
  delete[] input_b0;
  delete[] input_b1;

  return v_res;
}
