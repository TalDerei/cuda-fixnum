#include "retrieve_utils.cuh"
#include "constants.hpp"

uint8_t *get_1D_fixnum_array(my_fixnum_array *res, int n)
{
  uint8_t *local_results = new uint8_t[bytes_per_elem * n];
  int ret_nelts;
  res->retrieve_all(local_results, bytes_per_elem * n, &ret_nelts);
  return local_results;
}

uint8_t *get_2D_fixnum_array(my_fixnum_array *res0, my_fixnum_array *res1, int nelts)
{
  int lrl = bytes_per_elem * nelts;
  uint8_t *local_results0 = new uint8_t[lrl];
  uint8_t *local_results1 = new uint8_t[lrl];
  int ret_nelts;
  for (int i = 0; i < lrl; i++)
  {
    local_results0[i] = 0;
    local_results1[i] = 0;
  }

  res0->retrieve_all(local_results0, bytes_per_elem * nelts, &ret_nelts);
  res1->retrieve_all(local_results1, bytes_per_elem * nelts, &ret_nelts);

  uint8_t *local_results = new uint8_t[2 * lrl];

  for (int i = 0; i < nelts; i++)
  {
    mempcpy(local_results + 2 * i * bytes_per_elem, local_results0 + i * bytes_per_elem, bytes_per_elem);
    mempcpy(local_results + 2 * i * bytes_per_elem + bytes_per_elem, local_results1 + i * bytes_per_elem, bytes_per_elem);
  }

  delete local_results0;
  delete local_results1;
  return local_results;
}

uint8_t *get_3D_fixnum_array(my_fixnum_array *res0, my_fixnum_array *res1, my_fixnum_array *res2, int nelts)
{
  int lrl = bytes_per_elem * nelts;
  uint8_t *local_results0 = new uint8_t[lrl];
  uint8_t *local_results1 = new uint8_t[lrl];
  uint8_t *local_results2 = new uint8_t[lrl];
  int ret_nelts;
  for (int i = 0; i < lrl; i++)
  {
    local_results0[i] = 0;
    local_results1[i] = 0;
    local_results2[i] = 0;
  }

  res0->retrieve_all(local_results0, bytes_per_elem * nelts, &ret_nelts);
  res1->retrieve_all(local_results1, bytes_per_elem * nelts, &ret_nelts);
  res2->retrieve_all(local_results2, bytes_per_elem * nelts, &ret_nelts);

  uint8_t *local_results = new uint8_t[3 * lrl];

  for (int i = 0; i < nelts; i++)
  {
    mempcpy(local_results + 3 * i * bytes_per_elem, local_results0 + i * bytes_per_elem, bytes_per_elem);
    mempcpy(local_results + 3 * i * bytes_per_elem + bytes_per_elem, local_results1 + i * bytes_per_elem, bytes_per_elem);
    mempcpy(local_results + 3 * i * bytes_per_elem + 2 * bytes_per_elem, local_results2 + i * bytes_per_elem, bytes_per_elem);
  }

  delete[] local_results0;
  delete[] local_results1;
  delete[] local_results2;

  return local_results;
}
