#include "fq_mul.hpp"
#include "constants.hpp"

#include "retrieve_utils.cuh"
#include "gpu_constants.cuh"
#include "gpu_fq.cuh"
#include "gpu_params.cuh"

template <typename fixnum>
struct fq_mul_gpu
{
  __device__ void operator()(fixnum &r, fixnum a, fixnum b)
  {
    GpuParams &params = get_gpu_params();
    modnum mod = params.get_mnt_mod();
    GpuFq fqA = GpuFq::load(a, mod);
    GpuFq fqB = GpuFq::load(b, mod);
    GpuFq fqS = fqA * fqB;

    fqS.save(r);
  }
};

uint8_t *fq_mul(uint8_t *a, uint8_t *b, int nelts)
{
  my_fixnum_array *in_a = my_fixnum_array::create(a, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *in_b = my_fixnum_array::create(b, bytes_per_elem * nelts, bytes_per_elem);
  my_fixnum_array *res = my_fixnum_array::create(nelts);

  my_fixnum_array::template map<fq_mul_gpu>(res, in_a, in_b);
  uint8_t *v_res = get_1D_fixnum_array(res, nelts);

  delete in_a;
  delete in_b;
  delete res;

  return v_res;
}
