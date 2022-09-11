#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>

#include "params.hpp"
#include "utils.hpp"

#include "gpu_constants.cuh"
#include "gpu_params.cuh"

using namespace libff;
using namespace std;

// MNT 4 includes MNT4 G1: (Fq, Fq) and MNT4 G2: (Fq2, Fq2)
void init_params_mnt4()
{
  // Instantiate HostParams class -- the parameters below are stored interally as fixnum arrays of size '16 U' which is # of libms_per_elem
  HostParams params;
  // set_mnt_mod sets the finite field modulus for mnt4, and casts it to a pointer to fixnum
  params.set_mnt_mod((fixnum *)mnt4_modulus);
  // set_mnt_non_residue sets parameter 'non_residue' for the quadratic extension field
  params.set_mnt_non_residue((fixnum *)mnt4_non_residue);
  // set_mnt_coeff_a sets unique paramater 'coeff_a' for MNT4_G1
  params.set_mnt_coeff_a((fixnum *)mnt4_g1_coeff_a);

  // Set coeff_a which is additonal parameter unique to MNT4 curve
  // mnt4753_G2::coeff_a = mnt4753_twist_coeff_a, so it's equal to mnt4753_Fq2(mnt4753_G1::coeff_a * mnt4753_Fq2::non_residue,mnt4753_Fq::zero()) 
  // where mnt4753_G1::coeff_a = mnt4753_Fq("2") and mnt4753_Fq2::non_residue = mnt4753_Fq("13") == 26
  uint8_t *mnt4_g2_coeff_a2_c0 = new uint8_t[bytes_per_elem];
  memset(mnt4_g2_coeff_a2_c0, 0, bytes_per_elem);
  memcpy(mnt4_g2_coeff_a2_c0, (void *)mnt4753_G2::coeff_a.c0.as_bigint().data, io_bytes_per_elem);
  // cout << "coeff_a.c0 is: " << mnt4753_G2::coeff_a.c0 << endl; 

  uint8_t *mnt4_g2_coeff_a2_c1 = new uint8_t[bytes_per_elem];
  memset(mnt4_g2_coeff_a2_c1, 0, bytes_per_elem);
  memcpy(mnt4_g2_coeff_a2_c1, (void *)mnt4753_G2::coeff_a.c1.as_bigint().data, io_bytes_per_elem);
  // cout << "coeff_a.c1 is: " << mnt4753_G2::coeff_a.c0 << endl; 

  params.set_mnt_coeff_a2((fixnum *)mnt4_g2_coeff_a2_c0, (fixnum *)mnt4_g2_coeff_a2_c1);

  // set the parameters on the gpu (converting host parameters to device parameters)
  init_params(params);

  delete[] mnt4_g2_coeff_a2_c0;
  delete[] mnt4_g2_coeff_a2_c1;
}

void init_params_mnt6()
{
  HostParams params;
  params.set_mnt_mod((fixnum *)mnt6_non_residue);
  params.set_mnt_non_residue((fixnum *)mnt6_non_residue);
  params.set_mnt_coeff_a((fixnum *)mnt6_g1_coeff_a);

  uint8_t *mnt6_g2_coeff_a3_c0 = new uint8_t[bytes_per_elem];
  memset(mnt6_g2_coeff_a3_c0, 0, bytes_per_elem);
  memcpy(mnt6_g2_coeff_a3_c0, (void *)mnt6753_G2::coeff_a.c0.as_bigint().data, io_bytes_per_elem);

  uint8_t *mnt6_g2_coeff_a3_c1 = new uint8_t[bytes_per_elem];
  memset(mnt6_g2_coeff_a3_c1, 0, bytes_per_elem);
  memcpy(mnt6_g2_coeff_a3_c1, (void *)mnt6753_G2::coeff_a.c1.as_bigint().data, io_bytes_per_elem);

  uint8_t *mnt6_g2_coeff_a3_c2 = new uint8_t[bytes_per_elem];
  memset(mnt6_g2_coeff_a3_c2, 0, bytes_per_elem);
  memcpy(mnt6_g2_coeff_a3_c2, (void *)mnt6753_G2::coeff_a.c2.as_bigint().data, io_bytes_per_elem);

  params.set_mnt_coeff_a3((fixnum *)mnt6_g2_coeff_a3_c0, (fixnum *)mnt6_g2_coeff_a3_c1, (fixnum *)mnt6_g2_coeff_a3_c2);

  init_params(params);

  delete[] mnt6_g2_coeff_a3_c0;
  delete[] mnt6_g2_coeff_a3_c1;
  delete[] mnt6_g2_coeff_a3_c2;
}
