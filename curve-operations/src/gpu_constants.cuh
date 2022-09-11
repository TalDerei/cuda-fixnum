#pragma once

#include "./cuda-fixnum/fixnum/warp_fixnum.cu"
#include "./cuda-fixnum/array/fixnum_array.h"
#include "./cuda-fixnum/functions/modexp.cu"
#include "./cuda-fixnum/functions/multi_modexp.cu"
#include "./cuda-fixnum/modnum/modnum_monty_redc.cu"
#include "./cuda-fixnum/modnum/modnum_monty_cios.cu"

#include "constants.hpp"

using namespace cuFIXNUM;

typedef warp_fixnum<bytes_per_elem, u64_fixnum> fixnum;
typedef fixnum_array<fixnum> my_fixnum_array;
// redc may be worth trying over cios
typedef modnum_monty_redc<fixnum> modnum;
