#pragma once

#include "gpu_fq.cuh"

class GpuG1
{
  GpuFq X, Y, Z;
  GpuFq &coeff_a;

public:
  __device__
  GpuG1(const GpuFq &X, const GpuFq &Y, const GpuFq &Z, GpuFq &coeff_a) : X(X), Y(Y), Z(Z), coeff_a(coeff_a) {}

  __device__ __forceinline__ void save(fixnum &c0, fixnum &c1, fixnum &c2)
  {
    this->X.save(c0);
    this->Y.save(c1);
    this->Z.save(c2);
  }

  __device__ __forceinline__ bool is_zero() const
  {
    return this->X.is_zero() && this->Z.is_zero();
  }

  __device__
      GpuG1
      operator+(const GpuG1 &other) const
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

    const GpuFq X1Z2 = this->X * other.Z; // X1Z2 = X1*Z2
    const GpuFq X2Z1 = this->Z * other.X; // X2Z1 = X2*Z1

    // (used both in add and double checks)

    const GpuFq Y1Z2 = this->Y * other.Z; // Y1Z2 = Y1*Z2
    const GpuFq Y2Z1 = this->Z * other.Y; // Y2Z1 = Y2*Z1

    if (X1Z2 == X2Z1 && Y1Z2 == Y2Z1)
    {
      // perform dbl case
      const GpuFq XX = this->X.squared();                  // XX  = X1^2
      const GpuFq ZZ = this->Z.squared();                  // ZZ  = Z1^2
      const GpuFq w = this->coeff_a * ZZ + (XX + XX + XX); // w   = a*ZZ + 3*XX
      const GpuFq Y1Z1 = this->Y * this->Z;
      const GpuFq s = Y1Z1 + Y1Z1;                       // s   = 2*Y1*Z1
      const GpuFq ss = s.squared();                      // ss  = s^2
      const GpuFq sss = s * ss;                          // sss = s*ss
      const GpuFq R = this->Y * s;                       // R   = Y1*s
      const GpuFq RR = R.squared();                      // RR  = R^2
      const GpuFq B = (this->X + R).squared() - XX - RR; // B   = (X1+R)^2 - XX - RR
      const GpuFq h = w.squared() - (B + B);             // h   = w^2 - 2*B
      const GpuFq X3 = h * s;                            // X3  = h*s
      const GpuFq Y3 = w * (B - h) - (RR + RR);          // Y3  = w*(B-h) - 2*RR
      const GpuFq Z3 = sss;                              // Z3  = sss

      return GpuG1(X3, Y3, Z3, this->coeff_a);
    }

    // if we have arrived here we are in the add case
    const GpuFq Z1Z2 = this->Z * other.Z;      // Z1Z2 = Z1*Z2
    const GpuFq u = Y2Z1 - Y1Z2;               // u    = Y2*Z1-Y1Z2
    const GpuFq uu = u.squared();              // uu   = u^2
    const GpuFq v = X2Z1 - X1Z2;               // v    = X2*Z1-X1Z2
    const GpuFq vv = v.squared();              // vv   = v^2
    const GpuFq vvv = v * vv;                  // vvv  = v*vv
    const GpuFq R = vv * X1Z2;                 // R    = vv*X1Z2
    const GpuFq A = uu * Z1Z2 - (vvv + R + R); // A    = uu*Z1Z2 - vvv - 2*R
    const GpuFq X3 = v * A;                    // X3   = v*A
    const GpuFq Y3 = u * (R - A) - vvv * Y1Z2; // Y3   = u*(R-A) - vvv*Y1Z2
    const GpuFq Z3 = vvv * Z1Z2;               // Z3   = vvv*Z1Z2

    return GpuG1(X3, Y3, Z3, this->coeff_a);
  }

  __device__ __forceinline__
      GpuG1
      to_affine_coordinates(modnum &mod)
  {
    if (this->is_zero())
    {
      return GpuG1(GpuFq::zero(mod), GpuFq::one(mod), GpuFq::zero(mod), this->coeff_a);
    }
    else
    {
      const GpuFq Z_inv = Z.inverse();
      return GpuG1(this->X * Z_inv, this->Y * Z_inv, GpuFq::one(mod), this->coeff_a);
    }
  }
};
