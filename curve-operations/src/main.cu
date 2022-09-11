#include <cstdint>
#include <cstdio>
#include <cstring>

#include "io.hpp"
#include "stages.hpp"

using namespace std;

uint8_t *reduce_g2(uint8_t *a, int nelts)
{
  return NULL;
}

int main(int argc, char *argv[])
{
  setbuf(stdout, NULL);

  // Initialize all public parameters
  init_libff();

  bool is_stage_0 = strcmp(argv[1], "compute-stage-0") == 0;
  bool is_stage_1 = strcmp(argv[1], "compute-stage-1") == 0;
  bool is_stage_2 = strcmp(argv[1], "compute-stage-2") == 0;
  bool is_stage_3 = strcmp(argv[1], "compute-stage-3") == 0 || strcmp(argv[1], "compute") == 0;

  FILE *inputs = fopen(argv[2], "r");
  FILE *outputs = fopen(argv[3], "w");

  if (is_stage_0)
  {
    stage_0(inputs, outputs);
  }
  else if (is_stage_1)
  {
    stage_1(inputs, outputs);
  }
  else if (is_stage_2)
  {
    stage_2(inputs, outputs);
  }
  else if (is_stage_3)
  {
    stage_3(inputs, outputs);
  }

  return 0;
}
