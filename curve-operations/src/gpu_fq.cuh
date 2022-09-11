#pragma once

#include "gpu_constants.cuh"

class GpuFq
{
public:
  fixnum data;
  modnum &mod;

public:
  __device__
  GpuFq(const fixnum &data, modnum &mod) : data(data), mod(mod) {}

  __device__ static GpuFq zero(modnum &mod)
  {
    return GpuFq(mod.zero(), mod);
  }

  __device__ static GpuFq one(modnum &mod)
  {
    return GpuFq(mod.one(), mod);
  }

  __device__ static GpuFq load(const fixnum &data, modnum &mod)
  {
    fixnum result;
    mod.to_modnum(result, data);
    return GpuFq(result, mod);
  }

  __device__ __forceinline__ void save(fixnum &result)
  {
    this->mod.from_modnum(result, this->data);
  }

  __device__ __forceinline__ GpuFq operator*(const GpuFq &other) const
  {
    fixnum result;
    this->mod.mul(result, this->data, other.data);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ GpuFq operator+(const GpuFq &other) const
  {
    fixnum result;
    this->mod.add(result, this->data, other.data);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ GpuFq operator-(const GpuFq &other) const
  {
    fixnum result;
    this->mod.sub(result, this->data, other.data);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ bool operator==(const GpuFq &other) const
  {
    return fixnum::cmp(this->data, other.data) == 0;
  }

  __device__ __forceinline__ GpuFq squared() const
  {
    fixnum result;
    this->mod.sqr(result, this->data);
    return GpuFq(result, this->mod);
  }

  __device__ __forceinline__ bool is_zero() const
  {
    return fixnum::is_zero(this->data);
  }

  __device__ __forceinline__ GpuFq inverse() const
  {
    // FIXME: not imnpemented
    return GpuFq::zero(this->mod);
  }
};
