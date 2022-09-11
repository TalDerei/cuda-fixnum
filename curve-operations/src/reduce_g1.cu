#include <cstring>

#include "reduce.hpp"
#include "constants.hpp"

#include "retrieve_utils.cuh"
#include "gpu_constants.cuh"
#include "gpu_fq.cuh"
#include "gpu_g1.cuh"
#include "gpu_params.cuh"
#include "utils.hpp"

using namespace std;

template <typename fixnum>
struct reduce_g1_gpu
{
  __device__ void operator()(fixnum &r0, fixnum &r1, fixnum &r2, fixnum a0, fixnum a1, fixnum a2, fixnum b0, fixnum b1, fixnum b2) const
  {
    GpuParams &params = get_gpu_params();
    modnum mod = params.get_mnt_mod();
    GpuFq coeff_a = params.get_mnt_coeff_a();

    // Loading in values and converting them to monty arithmetic
    GpuG1 gA = GpuG1(GpuFq::load(a0, mod), GpuFq::load(a1, mod), GpuFq::load(a2, mod), coeff_a);
    GpuG1 gB = GpuG1(GpuFq::load(b0, mod), GpuFq::load(b1, mod), GpuFq::load(b2, mod), coeff_a);
    // Performing the addition of the elements in parallel
    GpuG1 fqS = gA + gB;

    // Saving the results
    fqS.save(r0, r1, r2);
  }
};

uint8_t *reduce_g1_internal(my_fixnum_array *res0, my_fixnum_array *res1, my_fixnum_array *res2, my_fixnum_array *in_a0, my_fixnum_array *in_a1, my_fixnum_array *in_a2, my_fixnum_array *in_b0, my_fixnum_array *in_b1, my_fixnum_array *in_b2, int nelts)
{
  // cout << "Entered reduce_g1_internal" << endl;
  // Calling 'reduce_g1_gpu' struct that contains GPU functions to be performed on these arrays
  // This calls functions in fixnum_array.cu which allocate memory and synchronize memory between host and device
  my_fixnum_array::template map<reduce_g1_gpu>(res0, res1, res2, in_a0, in_a1, in_a2, in_b0, in_b1, in_b2);

  if (nelts == 1)
  {
    // Store result in single pointer to uint8_t array of size 3 * bytes_per_elem
    uint8_t *result = new uint8_t[3 * bytes_per_elem];
    res0->retrieve_into(result, bytes_per_elem, 0);
    res1->retrieve_into(result + bytes_per_elem, bytes_per_elem, 0);
    res2->retrieve_into(result + 2 * bytes_per_elem, bytes_per_elem, 0);
    return result;
  }
  else
  {
    // Next round input is previous result divided into two halves
    my_fixnum_array *new_in_a0 = my_fixnum_array::wrap(res0->get_ptr(), nelts / 2);
    my_fixnum_array *new_in_b0 = my_fixnum_array::wrap(res0->get_ptr() + bytes_per_elem * nelts / 2, nelts / 2);
    my_fixnum_array *new_in_a1 = my_fixnum_array::wrap(res1->get_ptr(), nelts / 2);
    my_fixnum_array *new_in_b1 = my_fixnum_array::wrap(res1->get_ptr() + bytes_per_elem * nelts / 2, nelts / 2);
    my_fixnum_array *new_in_a2 = my_fixnum_array::wrap(res2->get_ptr(), nelts / 2);
    my_fixnum_array *new_in_b2 = my_fixnum_array::wrap(res2->get_ptr() + bytes_per_elem * nelts / 2, nelts / 2);

    // Output is reused previous input
    my_fixnum_array *new_res0 = my_fixnum_array::wrap(in_b0->get_ptr(), nelts / 2);
    my_fixnum_array *new_res1 = my_fixnum_array::wrap(in_b1->get_ptr(), nelts / 2);
    my_fixnum_array *new_res2 = my_fixnum_array::wrap(in_b2->get_ptr(), nelts / 2);

    // Keeps repeating until we have a single X, Y, and Z value
    return reduce_g1_internal(new_res0, new_res1, new_res2, new_in_a0, new_in_a1, new_in_a2, new_in_b0, new_in_b1, new_in_b2, nelts / 2);
  }
}

uint8_t *reduce_g1(uint8_t *a, int nelts)
{
  // Allocate memory to uint8_t arrays of size 128 x 1024 / 2 bytes each
  uint8_t *input_a0 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_a1 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_a2 = new uint8_t[bytes_per_elem * nelts / 2];

  // Splitting half of the elements into the arrays defined below
  for (int i = 0; i < nelts / 2; i++)
  {
    memcpy(input_a0 + i * bytes_per_elem, a + 3 * i * bytes_per_elem, bytes_per_elem);
    // print_array(input_a0);
    memcpy(input_a1 + i * bytes_per_elem, a + 3 * i * bytes_per_elem + bytes_per_elem, bytes_per_elem);
    // print_array(input_a1);
    memcpy(input_a2 + i * bytes_per_elem, a + 3 * i * bytes_per_elem + 2 * bytes_per_elem, bytes_per_elem);
    // print_array(input_a2);
  }

  uint8_t *input_b0 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_b1 = new uint8_t[bytes_per_elem * nelts / 2];
  uint8_t *input_b2 = new uint8_t[bytes_per_elem * nelts / 2];

  for (int i = 0; i < nelts / 2; i++)
  {
    mempcpy(input_b0 + i * bytes_per_elem, a + 3 * (i + nelts / 2) * bytes_per_elem, bytes_per_elem);
    mempcpy(input_b1 + i * bytes_per_elem, a + 3 * (i + nelts / 2) * bytes_per_elem + bytes_per_elem, bytes_per_elem);
    mempcpy(input_b2 + i * bytes_per_elem, a + 3 * (i + nelts / 2) * bytes_per_elem + 2 * bytes_per_elem, bytes_per_elem);
  }

  // Converting uint8_t * array into fixnum_array
  // 'Create' takes a pointer to the data, total bytes (bytes_per_elem * nelts / 2), and bytes_per_elem 
  my_fixnum_array *in_a0 = my_fixnum_array::create(input_a0, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *in_a1 = my_fixnum_array::create(input_a1, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *in_a2 = my_fixnum_array::create(input_a2, bytes_per_elem * nelts / 2, bytes_per_elem);

  my_fixnum_array *in_b0 = my_fixnum_array::create(input_b0, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *in_b1 = my_fixnum_array::create(input_b1, bytes_per_elem * nelts / 2, bytes_per_elem);
  my_fixnum_array *in_b2 = my_fixnum_array::create(input_b2, bytes_per_elem * nelts / 2, bytes_per_elem);

  my_fixnum_array *res0 = my_fixnum_array::create(nelts / 2);
  my_fixnum_array *res1 = my_fixnum_array::create(nelts / 2);
  my_fixnum_array *res2 = my_fixnum_array::create(nelts / 2);

  // Call reduce_g1_internal
  uint8_t *v_res = reduce_g1_internal(res0, res1, res2, in_a0, in_a1, in_a2, in_b0, in_b1, in_b2, nelts / 2);

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
