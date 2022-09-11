#include <cstdio>

#include "utils.hpp"
#include "constants.hpp"

void print_array(uint8_t *a)
{
  for (int j = 0; j < 128; j++)
  {
    printf("%x ", ((uint8_t *)(a))[j]);
  }
  printf("\n");
}

void print_double_array(uint8_t *a)
{
  printf("c0: ");
  print_array(a);
  printf("c1: ");
  print_array(a + bytes_per_elem);
}

void print_tripple_array(uint8_t *a)
{
  printf("c0: ");
  print_array(a);
  printf("c1: ");
  print_array(a + bytes_per_elem);
  printf("c2: ");
  print_array(a + 2 * bytes_per_elem);
}

void printG1(uint8_t *src)
{
  printf("X:\n");
  print_array(src);
  printf("Y:\n");
  print_array(src + bytes_per_elem);
  printf("Z:\n");
  print_array(src + 2 * bytes_per_elem);
}

void print_mnt4_G2(uint8_t *src)
{
  printf("X:\n");
  print_double_array(src);
  printf("Y:\n");
  print_double_array(src + 2 * bytes_per_elem);
  printf("Z:\n");
  print_double_array(src + 4 * bytes_per_elem);
}

void print_mnt6_G2(uint8_t *src)
{
  printf("X:\n");
  print_tripple_array(src);
  printf("Y:\n");
  print_tripple_array(src + 3 * bytes_per_elem);
  printf("Z:\n");
  print_tripple_array(src + 6 * bytes_per_elem);
}
