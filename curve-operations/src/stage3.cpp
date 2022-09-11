#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "constants.hpp"
#include "io.hpp"
#include "reduce.hpp"
#include "utils.hpp"
#include "params.hpp"

using namespace std;

void stage_3(FILE *inputs, FILE *outputs)
{
  size_t n;

  while (true)
  {
    size_t elts_read = fread((void *)&n, sizeof(size_t), 1, inputs);
    if (elts_read == 0)
    {
      break;
    }

    std::cerr << n << std::endl;

    cout << "-----------------------------------------------------------------" << endl;
    cout << "Starting MNT G1 Calculations" << endl;

    // MNT4 G1 curve: (Fq, Fq)
    // Initializing size to 3 * n * bytes_per_elem to account for X,Y,Z arrays
    cout << "Reading the elements from file for MNT4 G1 curve" << endl;
    uint8_t *x0 = new uint8_t[3 * n * bytes_per_elem];
    memset(x0, 0, 3 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {     
      // read the elements of mnt4 g1 that are in montgomery form
      read_mnt4_g1_montgomery(x0 + 3 * i * bytes_per_elem, inputs);
    }
    /*
    Example output of printing G1;
    96 bytes for each element (X, Y) w/ 32 bytes of padding. 
    This represents 16 limbs of 8 bytes each ~

    X:
    e6 d5 8f 1a c6 d9 13 3a fc 6e da a3 91 19 53 7f 7d 15 3e db b0 eb bf 54 7 96 ec 53 64 e6 b 93 
    f9 50 c6 4c ef 74 af 16 d8 9e db e1 f6 cd 99 38 c5 7e 3a b3 1 5d 70 25 59 79 a0 29 8c 8f a4 5e 
    e3 be 86 c6 12 f9 12 40 f0 b1 13 50 39 1d fa 91 fe 4a e 64 96 82 c3 fc ac 8f 51 58 b8 97 1 0 
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    Y:
    b 4 73 80 aa e3 3 dc 3b fb d7 a2 a2 7a 38 bb ae 18 ce f 91 f 90 73 4f 2a 26 5c d4 43 6 c5 c5 
    49 41 a8 3f 1a 2c 3d 10 eb fb f3 7c 98 bd f1 8c 59 9d 53 f2 70 81 44 4e 9c 3d 63 6 1f 54 33 
    e7 8d e f5 6b 2 ad 2e f9 72 bf 76 f2 72 fc 2e 5a 36 c0 c1 a5 dc d4 92 42 bf 66 4c 8d af 1 0 
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    Z:
    1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    */
    printG1(x0);

    // initialize the parameters for mnt4 
    cout << "Initializing parameters for mnt4 G1" << endl;
    init_params_mnt4();

    cout << "Starting reduce_g1" << endl;
    uint8_t *res_x0 = reduce_g1(x0, n);
    printG1(res_x0);

    // Converting to affine coordinates
    cout << "Converting to mnt G1 to affine coordinates" << endl;
    uint8_t *res_x0_affine = mnt4_g1_to_affine(res_x0);
    cout << "Results:" << endl;
    printG1(res_x0_affine);

    // Writing to output
    cout << "Writing mnt G1 to output" << endl;
    write_mnt4_g1_montgomery(res_x0_affine, outputs);

    cout << "-----------------------------------------------------------------" << endl;
    cout << "Starting MNT G2 Calculations" << endl;

    // mnt 4 G2
    uint8_t *y0 = new uint8_t[6 * n * bytes_per_elem];
    memset(y0, 0, 6 * n * bytes_per_elem);
    cout << "Reading the elements from file for MNT4 G2 curve" << endl;
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt4_g2_montgomery(y0 + 6 * i * bytes_per_elem, inputs);
    }
    /* 
    Example output of printing G2;
    96 bytes for each element (X, Y) w/ 32 bytes of padding. 
    This represents 16 limbs of 8 bytes each ~

    X,Y,Z each have c0, and c1 sub components from the underlying Fq field

    X:
    c0: db b5 2b fd 84 55 6 c7 b0 af b7 e3 7b 28 1c c5 61 ef b1 e9 6b 8e 2e af 1 9d ef a3 8c f8 87 e0 2c 29 8c 35 
    7c c9 75 45 98 4e 24 ad ae c4 7f 20 8a 70 ca cf 60 b3 8b 61 2e 3b cc 6c 6d 4f 8d 43 ca c0 c7 93 4a 63 88 f6 70 
    ff 79 8c b8 ab 3b 1a 78 72 a8 1a c4 d5 8d 87 d6 b0 7a e1 31 46 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    0 0 0 0 0 0 0 0 0 0 
    c1: 9c f9 6f 4 2 fb 8a 9f c1 29 d0 ad ec a5 e1 bf 57 3f 8c 4b 75 83 1e ac 7 63 96 2a bf 4f 3b 2 b7 d4 3c ba ff 
    a7 2c 37 a6 27 cc 1e 57 c4 b2 aa 7b 4d 7f d4 fa 6e fc 22 a 7 5b 21 bc cc b2 e6 49 3b 2e 5c 4b 47 4a fb 2 3c 91 
    93 80 bb 6 83 3a 45 ae b7 70 9c bc b6 e5 35 48 e1 72 c 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    0 0 0 0 0 0 
    Y:
    c0: 73 cd 80 24 36 db 61 f3 95 61 67 cc e3 c7 46 1 71 28 b3 66 b6 39 23 59 59 26 56 9e 7c dc 24 ab d4 57 ce 7b d5 
    20 be a6 73 42 27 4a b9 35 b8 79 ca fd 3a de a1 39 33 a9 90 6e 3 dd 81 cb 1e 2a cf ba b4 3b 24 7d 96 d8 39 a9 d4 
    73 a0 ff 7a 8b 7d 6e b8 98 7a 86 7f e 9e 9e 19 d5 bb c0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    0 0 0 0 0 
    c1: 42 37 2d aa df 90 48 a2 cd 70 45 1e 49 6c 2f 8b c2 28 5d 93 9 b2 34 56 4b 40 e f6 c3 75 99 2c 29 d8 51 56 55 
    79 66 f3 f5 43 e0 b9 c3 4 3b 61 c0 5b c7 d7 3a e5 c 80 a4 bb f6 d5 6d 90 ab 82 82 6d 5e 2d d1 f1 ac cf f0 75 99 45 
    7f bd cb f1 b6 3d 30 44 cd b4 d3 38 e4 59 7d 39 c9 4a 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    0 0 0 
    Z:
    c0: 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    c1: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    */
    print_mnt4_G2(y0);

    cout << "Initializing parameters for mnt4 G2" << endl;
    init_params_mnt4();
    
    cout << "Starting reduce_mnt4_g2" << endl;
    uint8_t *res_y0 = reduce_mnt4_g2(y0, n);

    cout << "Converting to mnt G2 to affine coordinates" << endl;
    uint8_t *res_y0_affine = mnt4_g2_to_affine(res_y0);
    print_mnt4_G2(res_y0_affine);

    cout << "Writing mnt G2 to output" << endl;
    write_mnt4_g2_montgomery(res_y0_affine, outputs);

    // mnt 6 G1
    uint8_t *x1 = new uint8_t[3 * n * bytes_per_elem];
    memset(x1, 0, 3 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt6_g1_montgomery(x1 + 3 * i * bytes_per_elem, inputs);
    }

    // mnt 6 G1 same as mnt 4 G1 just on a different finite field modulus
    init_params_mnt6();
    uint8_t *res_x1 = reduce_g1(x1, n);

    uint8_t *res_x1_affine = mnt6_g1_to_affine(res_x1);
    write_mnt6_g1_montgomery(res_x1_affine, outputs);
    //printG1(res_x1_affine);

    // mnt 6 G2
    uint8_t *y1 = new uint8_t[9 * n * bytes_per_elem];
    memset(y1, 0, 9 * n * bytes_per_elem);
    for (size_t i = 0; i < n; ++i)
    {
      read_mnt6_g2_montgomery(y1 + 9 * i * bytes_per_elem, inputs);
    }

    init_params_mnt6();
    // There's an error here for some reason ~
    uint8_t *res_y1 = reduce_mnt6_g2(y1, n);

    uint8_t *res_y1_affine = mnt6_g2_to_affine(res_y1);
    write_mnt6_g2_montgomery(res_y1_affine, outputs);
    //print_mnt6_G2(res_y1_affine);

    delete[] x0;
    delete[] x1;
    delete[] y0;
    delete[] y1;
    delete[] res_x0;
    delete[] res_x1;
    delete[] res_y0;
    delete[] res_y1;
  }
}
