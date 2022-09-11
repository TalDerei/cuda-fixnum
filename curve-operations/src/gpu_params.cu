#include "gpu_constants.cuh"
#include "gpu_fq.cuh"
#include "gpu_params.cuh"

using namespace std;

__constant__ HostParams host_params;
__device__ GpuParams gpu_params;

template <typename fixnum>
struct init_params_gpu
{
  __device__ void operator()(fixnum dummy)
  {
    // Converting all array's to fixnums and setting the parameters on the gpu
    gpu_params.set_mnt_mod(modnum(array_to_fixnum(host_params.get_mnt_mod())));
    gpu_params.set_mnt_non_residue(array_to_fixnum(host_params.get_mnt_non_residue()));
    gpu_params.set_mnt_coeff_a(array_to_fixnum(host_params.get_mnt_coeff_a()));
    gpu_params.set_mnt_coeff_a2(array_to_fixnum(host_params.get_mnt_coeff_a2_c0()), array_to_fixnum(host_params.get_mnt_coeff_a2_c1()));
    gpu_params.set_mnt_coeff_a3(array_to_fixnum(host_params.get_mnt_coeff_a3_c0()), array_to_fixnum(host_params.get_mnt_coeff_a3_c1()), array_to_fixnum(host_params.get_mnt_coeff_a3_c2()));
  }

  __device__ fixnum array_to_fixnum(fixnum *arr)
  {
    return arr[fixnum::layout::laneIdx()];
  }
};

void init_params(HostParams &params)
{
  cout << "Entered init_params" << endl;
  my_fixnum_array *dummy = my_fixnum_array::create(1);
  // Copy params (on host) to host_params (on device)
  cudaMemcpyToSymbol(host_params, &params, sizeof(HostParams));
  // Creates the cuda memory streams and synchronizations -- takes in dummy input
  my_fixnum_array::template map<init_params_gpu>(dummy);
  delete dummy;
}

__device__
    GpuParams &
    get_gpu_params()
{
  return gpu_params;
}