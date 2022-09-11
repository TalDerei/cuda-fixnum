#pragma once

#include "gpu_fq.cuh"

class GpuFq2
{
  GpuFq c0, c1;

  GpuFq &non_residue;

public:
  __device__
  GpuFq2(const GpuFq &c0, const GpuFq &c1, GpuFq &non_residue) : c0(c0), c1(c1), non_residue(non_residue) {}

  __device__ __forceinline__ void save(fixnum &c0, fixnum &c1)
  {
    this->c0.save(c0);
    this->c1.save(c1);
  }

  __device__ __forceinline__ GpuFq2 operator*(const GpuFq2 &other) const
  {
    GpuFq a0_b0 = this->c0 * other.c0;
    GpuFq a1_b1 = this->c1 * other.c1;

    GpuFq a0_plus_a1 = this->c0 + this->c1;
    GpuFq b0_plus_b1 = other.c0 + other.c1;

    GpuFq c = a0_plus_a1 * b0_plus_b1;

    return GpuFq2(a0_b0 + a1_b1 * this->non_residue, c - a0_b0 - a1_b1, this->non_residue);
  }

  __device__ __forceinline__ GpuFq2 operator+(const GpuFq2 &other) const
  {
    return GpuFq2(this->c0 + other.c0, this->c1 + other.c1, this->non_residue);
  }

  __device__ __forceinline__ GpuFq2 operator-(const GpuFq2 &other) const
  {
    return GpuFq2(this->c0 - other.c0, this->c1 - other.c1, this->non_residue);
  }

  __device__ __forceinline__ bool operator==(const GpuFq2 &other) const
  {
    return (this->c0 == other.c0 && this->c1 == other.c1);
  }

  __device__ GpuFq2 squared() const
  {
    /* Devegili OhEig Scott Dahab --- Multiplication and Squaring on Pairing-Friendly Fields.pdf; Section 3 (Complex squaring) */
    const GpuFq &a = this->c0, &b = this->c1;
    const GpuFq ab = a * b;

    return GpuFq2((a + b) * (a + this->non_residue * b) - ab - this->non_residue * ab, ab + ab, this->non_residue);
  }

  __device__ __forceinline__ bool is_zero() const { return this->c0.is_zero() && this->c1.is_zero(); }
};
