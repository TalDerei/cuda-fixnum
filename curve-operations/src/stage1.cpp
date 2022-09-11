#include <cstdint>
#include <cstdio>
#include <cstring>

#include "constants.hpp"
#include "io.hpp"
#include "params.hpp"
#include "fq_mul.hpp"

void stage_1(FILE *inputs, FILE *outputs)
{
  size_t n;

  while (true)
  {
    size_t elts_read = fread((void *)&n, sizeof(size_t), 1, inputs);
    if (elts_read == 0)
    {
      break;
    }

    uint8_t *x0 = new uint8_t[2 * n * bytes_per_elem];
    memset(x0, 0, 2 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt_fq2(x0 + 2 * i * bytes_per_elem, inputs);
    }

    uint8_t *x1 = new uint8_t[2 * n * bytes_per_elem];
    memset(x1, 0, 2 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt_fq2(x1 + 2 * i * bytes_per_elem, inputs);
    }

    init_params_mnt4();
    uint8_t *res_x = fq2_mul(x0, x1, n);

    for (size_t i = 0; i < n; ++i)
    {
      write_mnt_fq2(res_x + 2 * i * bytes_per_elem, outputs);
    }

    delete[] x0;
    delete[] x1;
    delete[] res_x;
  }
}
