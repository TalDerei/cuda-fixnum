#pragma once

#include <cstdint>
#include "gpu_constants.cuh"

uint8_t *get_1D_fixnum_array(my_fixnum_array *res, int n);
uint8_t *get_2D_fixnum_array(my_fixnum_array *res0, my_fixnum_array *res1, int nelts);
uint8_t *get_3D_fixnum_array(my_fixnum_array *res0, my_fixnum_array *res1, my_fixnum_array *res2, int nelts);