#pragma once

#include "gpu_fq.cuh"
#include "gpu_fq2.cuh"

template <typename GpuFx>
class GpuG2
{
public:
  GpuFx X, Y, Z;
  GpuFx &coeff_a;

  __device__
  GpuG2(const GpuFx &X, const GpuFx &Y, const GpuFx &Z, GpuFx &coeff_a) : X(X), Y(Y), Z(Z), coeff_a(coeff_a) {}

  __device__ __forceinline__ bool is_zero() const
  {
    return this->X.is_zero() && this->Z.is_zero();
  }

  __device__
      GpuG2
      operator+(const GpuG2 &other) const
  {
    // handle special cases having to do with O
    if (this->is_zero())
    {
      return other;
    }

    if (other.is_zero())
    {
      return *this;
    }

    const GpuFx X1Z2 = this->X * other.Z; // X1Z2 = X1*Z2
    const GpuFx X2Z1 = this->Z * other.X; // X2Z1 = X2*Z1

    // (used both in add and double checks)

    const GpuFx Y1Z2 = this->Y * other.Z; // Y1Z2 = Y1*Z2
    const GpuFx Y2Z1 = this->Z * other.Y; // Y2Z1 = Y2*Z1

    if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1)
    {
      // perform dbl case
      const GpuFx XX = this->X.squared();                  // XX  = X1^2
      const GpuFx ZZ = this->Z.squared();                  // ZZ  = Z1^2
      const GpuFx w = this->coeff_a * ZZ + (XX + XX + XX); // w   = a*ZZ + 3*XX
      const GpuFx Y1Z1 = this->Y * this->Z;
      const GpuFx s = Y1Z1 + Y1Z1;                       // s   = 2*Y1*Z1
      const GpuFx ss = s.squared();                      // ss  = s^2
      const GpuFx sss = s * ss;                          // sss = s*ss
      const GpuFx R = this->Y * s;                       // R   = Y1*s
      const GpuFx RR = R.squared();                      // RR  = R^2
      const GpuFx B = (this->X + R).squared() - XX - RR; // B   = (X1+R)^2 - XX - RR
      const GpuFx h = w.squared() - (B + B);             // h   = w^2 - 2*B
      const GpuFx X3 = h * s;                            // X3  = h*s
      const GpuFx Y3 = w * (B - h) - (RR + RR);          // Y3  = w*(B-h) - 2*RR
      const GpuFx Z3 = sss;                              // Z3  = sss

      return GpuG2(X3, Y3, Z3, this->coeff_a);
    }

    // if we have arrived here we are in the add case
    const GpuFx Z1Z2 = this->Z * other.Z;      // Z1Z2 = Z1*Z2
    const GpuFx u = Y2Z1 - Y1Z2;               // u    = Y2*Z1-Y1Z2
    const GpuFx uu = u.squared();              // uu   = u^2
    const GpuFx v = X2Z1 - X1Z2;               // v    = X2*Z1-X1Z2
    const GpuFx vv = v.squared();              // vv   = v^2
    const GpuFx vvv = v * vv;                  // vvv  = v*vv
    const GpuFx R = vv * X1Z2;                 // R    = vv*X1Z2
    const GpuFx A = uu * Z1Z2 - (vvv + R + R); // A    = uu*Z1Z2 - vvv - 2*R
    const GpuFx X3 = v * A;                    // X3   = v*A
    const GpuFx Y3 = u * (R - A) - vvv * Y1Z2; // Y3   = u*(R-A) - vvv*Y1Z2
    const GpuFx Z3 = vvv * Z1Z2;               // Z3   = vvv*Z1Z2

    return GpuG2(X3, Y3, Z3, this->coeff_a);
  }
};
