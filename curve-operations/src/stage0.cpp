#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "constants.hpp"
#include "io.hpp"
#include "params.hpp"
#include "fq_mul.hpp"

using namespace std;

void stage_0(FILE *inputs, FILE *outputs)
{
  size_t n;

  while (true)
  {
    size_t elts_read = fread((void *)&n, sizeof(size_t), 1, inputs);
    if (elts_read == 0)
    {
      break;
    }

    uint8_t *x0 = new uint8_t[n * bytes_per_elem];
    memset(x0, 0, n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt_fq(x0 + i * bytes_per_elem, inputs);
    }

    uint8_t *x1 = new uint8_t[n * bytes_per_elem];
    memset(x1, 0, n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt_fq(x1 + i * bytes_per_elem, inputs);
    }    

    init_params_mnt4();
    uint8_t *res_x = fq_mul(x0, x1, n);

    for (size_t i = 0; i < n; ++i)
    {
      write_mnt_fq(res_x + i * bytes_per_elem, outputs);
    }

    delete[] x0;
    delete[] x1;
    delete[] res_x;

    uint8_t *y0 = new uint8_t[n * bytes_per_elem];
    memset(y0, 0, n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt_fq(y0 + i * bytes_per_elem, inputs);
    }

    uint8_t *y1 = new uint8_t[n * bytes_per_elem];
    memset(y1, 0, n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt_fq(y1 + i * bytes_per_elem, inputs);
    }

    init_params_mnt6();
    uint8_t *res_y = fq_mul(y0, y1, n);

    for (size_t i = 0; i < n; ++i)
    {
      write_mnt_fq(res_y + i * bytes_per_elem, outputs);
    }

    delete[] y0;
    delete[] y1;
    delete[] res_y;
  }
}
