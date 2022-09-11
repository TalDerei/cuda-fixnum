#pragma once

#include <cstdint>

uint8_t *reduce_g1(uint8_t *a, int nelts);
uint8_t *reduce_mnt4_g2(uint8_t *a, int nelts);
uint8_t *reduce_mnt6_g2(uint8_t *a, int nelts);