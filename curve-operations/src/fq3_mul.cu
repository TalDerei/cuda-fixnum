#include "fq_mul.hpp"
#include "constants.hpp"

#include "retrieve_utils.cuh"
#include "gpu_constants.cuh"
#include "gpu_fq.cuh"
#include "gpu_fq3.cuh"
#include "gpu_params.cuh"

template <typename fixnum>
struct fq3_mul_gpu
{
  __device__ void operator()(fixnum &r0, fixnum &r1, fixnum &r2, fixnum a0, fixnum a1, fixnum a2, fixnum b0, fixnum b1, fixnum b2)
  {
    GpuParams &params = get_gpu_params();
    modnum mod = params.get_mnt_mod();
    GpuFq non_residue = params.get_mnt_non_residue();

    GpuFq3 fqA = GpuFq3(GpuFq::load(a0, mod), GpuFq::load(a1, mod), GpuFq::load(a2, mod), non_residue);
    GpuFq3 fqB = GpuFq3(GpuFq::load(b0, mod), GpuFq::load(b1, mod), GpuFq::load(b2, mod), non_residue);
    GpuFq3 fqS = fqA * fqB;
    fqS.save(r0, r1, r2);
  }
};

uint8_t *fq3_mul(uint8_t *a, uint8_t *b, int nelts)
{
  uint8_t *input_a0 = new uint8_t[bytes_per_elem * nelts];
  uint8_t *input_a1 = new uint8_t[bytes_per_elem * nelts];
  uint8_t *input_a2 = new uint8_t[bytes_per_elem * nelts];

  for (int i = 0; i < nelts; i++)
  {
    mempcpy(input_a0 + i * bytes_per_elem, a + 3 * i * bytes_per_elem, bytes_per_elem);
    mempcpy(input_a1 + i * bytes_per_elem, a + 3 * i * bytes_per_elem + bytes_per_elem, bytes_per_elem);
    mempcpy(input_a2 + i * bytes_per_elem, a + 3 * i * bytes_per_elem + 2 * bytes_per_elem, bytes_per_elem);
  }

  uint8_t *input_b0 = new uint8_t[bytes_per_elem * nelts];
  uint8_t *input_b1 = new uint8_t[bytes_per_elem * nelts];
  uint8_t *input_b2 = new uint8_t[bytes_per_elem * nelts];

  for (int i = 0; i < nelts; i++)
  {
    mempcpy(input_b0 + i * bytes_per_elem, b + 3 * i * bytes_per_elem, bytes_per_elem);
    mempcpy(input_b1 + i * bytes_per_elem, b + 3 * i * bytes_per_elem + bytes_per_elem, bytes_per_elem);
    mempcpy(input_b2 + i * bytes_per_elem, b + 3 * i * bytes_per_elem + 2 * bytes_per_elem, bytes_per_elem);
  }

  my_fixnum_array *in_a0 = my_fixnum_array::create(input_a0, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_a1 = my_fixnum_array::create(input_a1, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_a2 = my_fixnum_array::create(input_a2, bytes_per_elem * nelts, bytes_per_elem);

  my_fixnum_array *in_b0 = my_fixnum_array::create(input_b0, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_b1 = my_fixnum_array::create(input_b1, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_b2 = my_fixnum_array::create(input_b2, bytes_per_elem * nelts, bytes_per_elem);

  my_fixnum_array *res0 = my_fixnum_array::create(nelts);
  my_fixnum_array *res1 = my_fixnum_array::create(nelts);
  my_fixnum_array *res2 = my_fixnum_array::create(nelts);

  my_fixnum_array::template map<fq3_mul_gpu>(res0, res1, res2, in_a0, in_a1, in_a2, in_b0, in_b1, in_b2);

  uint8_t *v_res = get_3D_fixnum_array(res0, res1, res2, nelts);

  delete in_a0;
  delete in_a1;
  delete in_a2;
  delete in_b0;
  delete in_b1;
  delete in_b2;
  delete res0;
  delete res1;
  delete res2;
  delete[] input_a0;
  delete[] input_a1;
  delete[] input_a2;
  delete[] input_b0;
  delete[] input_b1;
  delete[] input_b2;

  return v_res;
}
