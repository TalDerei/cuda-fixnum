#pragma once

#include "gpu_constants.cuh"
#include "gpu_fq.cuh"
#include "gpu_fq2.cuh"
#include "gpu_fq3.cuh"

class GpuParams
{
  uint8_t mnt_mod[libms_per_elem * sizeof(modnum)];
  fixnum mnt_non_residue[libms_per_elem];
  fixnum mnt_coeff_a[libms_per_elem];

  fixnum mnt_coeff_a2_c0[libms_per_elem];
  fixnum mnt_coeff_a2_c1[libms_per_elem];

  fixnum mnt_coeff_a3_c0[libms_per_elem];
  fixnum mnt_coeff_a3_c1[libms_per_elem];
  fixnum mnt_coeff_a3_c2[libms_per_elem];

public:
  __device__
      modnum
      get_mnt_mod()
  {
    return ((modnum *)this->mnt_mod)[fixnum::layout::laneIdx()];
  }

  __device__ void set_mnt_mod(modnum mod)
  {
    ((modnum *)this->mnt_mod)[fixnum::layout::laneIdx()] = mod;
  }

  __device__
      GpuFq
      get_mnt_non_residue()
  {
    fixnum non_residue = this->mnt_non_residue[fixnum::layout::laneIdx()];
    modnum mod = this->get_mnt_mod();
    return GpuFq(non_residue, mod); // saved as MM, no need to load
  }

  __device__ void set_mnt_non_residue(fixnum non_residue)
  {
    modnum mod = this->get_mnt_mod();
    GpuFq fq = GpuFq::load(non_residue, mod);
    this->mnt_non_residue[fixnum::layout::laneIdx()] = fq.data; // save as MM
  }

  __device__
      GpuFq
      get_mnt_coeff_a()
  {
    fixnum coeff_a = this->mnt_coeff_a[fixnum::layout::laneIdx()];
    modnum mod = this->get_mnt_mod();
    return GpuFq(coeff_a, mod); // saved as MM, no need to load
  }

  __device__ void set_mnt_coeff_a(fixnum coeff_a)
  {
    modnum mod = this->get_mnt_mod();
    GpuFq fq = GpuFq::load(coeff_a, mod);
    this->mnt_coeff_a[fixnum::layout::laneIdx()] = fq.data; // save as MM
  }

  __device__
      GpuFq2
      get_mnt_coeff_a2()
  {
    fixnum coeff_a2_c0 = this->mnt_coeff_a2_c0[fixnum::layout::laneIdx()];
    fixnum coeff_a2_c1 = this->mnt_coeff_a2_c1[fixnum::layout::laneIdx()];

    modnum mod = this->get_mnt_mod();
    GpuFq non_residue = this->get_mnt_non_residue();
    return GpuFq2(GpuFq(coeff_a2_c0, mod), GpuFq(coeff_a2_c1, mod), non_residue); // saved as MM, no need to load
  }

  __device__
      GpuFq3
      get_mnt_coeff_a3()
  {
    fixnum coeff_a3_c0 = this->mnt_coeff_a3_c0[fixnum::layout::laneIdx()];
    fixnum coeff_a3_c1 = this->mnt_coeff_a3_c1[fixnum::layout::laneIdx()];
    fixnum coeff_a3_c2 = this->mnt_coeff_a3_c2[fixnum::layout::laneIdx()];

    modnum mod = this->get_mnt_mod();
    GpuFq non_residue = this->get_mnt_non_residue();
    return GpuFq3(GpuFq(coeff_a3_c0, mod), GpuFq(coeff_a3_c1, mod), GpuFq(coeff_a3_c2, mod), non_residue); // saved as MM, no need to load
  }

  __device__ void set_mnt_coeff_a2(fixnum coeff_a2_c0, fixnum coeff_a2_c1)
  {
    modnum mod = this->get_mnt_mod();
    GpuFq fq_c0 = GpuFq::load(coeff_a2_c0, mod);
    this->mnt_coeff_a2_c0[fixnum::layout::laneIdx()] = fq_c0.data; // save as MM

    GpuFq fq_c1 = GpuFq::load(coeff_a2_c1, mod);
    this->mnt_coeff_a2_c1[fixnum::layout::laneIdx()] = fq_c1.data; // save as MM
  }

  __device__ void set_mnt_coeff_a3(fixnum coeff_a3_c0, fixnum coeff_a3_c1, fixnum coeff_a3_c2)
  {
    modnum mod = this->get_mnt_mod();
    GpuFq fq_c0 = GpuFq::load(coeff_a3_c0, mod);
    this->mnt_coeff_a3_c0[fixnum::layout::laneIdx()] = fq_c0.data; // save as MM

    GpuFq fq_c1 = GpuFq::load(coeff_a3_c1, mod);
    this->mnt_coeff_a3_c1[fixnum::layout::laneIdx()] = fq_c1.data; // save as MM

    GpuFq fq_c2 = GpuFq::load(coeff_a3_c2, mod);
    this->mnt_coeff_a3_c2[fixnum::layout::laneIdx()] = fq_c2.data; // save as MM
  }
};

class HostParams
{
  fixnum mnt_mod[libms_per_elem];
  fixnum mnt_non_residue[libms_per_elem];
  fixnum mnt_coeff_a[libms_per_elem];

  fixnum mnt_coeff_a2_c0[libms_per_elem];
  fixnum mnt_coeff_a2_c1[libms_per_elem];

  fixnum mnt_coeff_a3_c0[libms_per_elem];
  fixnum mnt_coeff_a3_c1[libms_per_elem];
  fixnum mnt_coeff_a3_c2[libms_per_elem];

public:
  __host__ void set_mnt_mod(fixnum *mod)
  {
    memcpy(this->mnt_mod, mod, bytes_per_elem);
  }

  __device__
      fixnum *
      get_mnt_mod()
  {
    return this->mnt_mod;
  }

  __host__ void set_mnt_non_residue(fixnum *non_residue)
  {
    memcpy(this->mnt_non_residue, non_residue, bytes_per_elem);
  }

  __device__
      fixnum *
      get_mnt_non_residue()
  {
    return this->mnt_non_residue;
  }

  __host__ void set_mnt_coeff_a(fixnum *coeff_a)
  {
    memcpy(this->mnt_coeff_a, coeff_a, bytes_per_elem);
  }

  __device__
      fixnum *
      get_mnt_coeff_a()
  {
    return this->mnt_coeff_a;
  }

  __host__ void set_mnt_coeff_a2(fixnum *coeff_a2_c0, fixnum *coeff_a2_c1)
  {
    memcpy(this->mnt_coeff_a2_c0, coeff_a2_c0, bytes_per_elem);
    memcpy(this->mnt_coeff_a2_c1, coeff_a2_c1, bytes_per_elem);
  }

  __host__ void set_mnt_coeff_a3(fixnum *coeff_a3_c0, fixnum *coeff_a3_c1, fixnum *coeff_a3_c2)
  {
    memcpy(this->mnt_coeff_a3_c0, coeff_a3_c0, bytes_per_elem);
    memcpy(this->mnt_coeff_a3_c1, coeff_a3_c1, bytes_per_elem);
    memcpy(this->mnt_coeff_a3_c2, coeff_a3_c2, bytes_per_elem);
  }

  __device__
      fixnum *
      get_mnt_coeff_a2_c0()
  {
    return this->mnt_coeff_a2_c0;
  }

  __device__
      fixnum *
      get_mnt_coeff_a2_c1()
  {
    return this->mnt_coeff_a2_c1;
  }

  __device__
      fixnum *
      get_mnt_coeff_a3_c0()
  {
    return this->mnt_coeff_a3_c0;
  }

  __device__
      fixnum *
      get_mnt_coeff_a3_c1()
  {
    return this->mnt_coeff_a3_c1;
  }

  __device__
      fixnum *
      get_mnt_coeff_a3_c2()
  {
    return this->mnt_coeff_a3_c2;
  }
};

void init_params(HostParams &params);
__device__ GpuParams &get_gpu_params();