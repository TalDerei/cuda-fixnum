Not sure why the cuda-fixnum repo (https://github.com/MinaProtocol/cuda-fixnum) which has the CUDA reference implementation does not produce the same output as the CPU reference solution ((https://github.com/MinaProtocol/snark-challenge/tree/master/reference-01-field-arithmetic)  on the same input file? I thought the cuda repo was supposed to already be implemented for the first challenge (i.e. field arithmetic), but the 'shasum outputs' don't match for some reason.
     - the reason was because of the way I was interpreting the input file --> interpreting inputs as montgomery representation vs. ordinary numbers

In the cuda implementation reference, theres two pre-initialized arrays of integers which is important for the correctness of satisfying the checksum for some reason. Still not sure why we defined the modulus that way and how it relates to the reference CPU implementation. Because if I change just one of those elemts in the array, the checksum fails. 

For phase 1:
     Still need to determine exactly how the multiplication works in the 'mul_and_convert' struct when calling the modnum_monty_cios class.
          - see implementation of GPU functions

     Additionally need to figure out the formate 'x0 + 2 * i * bytes_per_elem' every time we're performing a read
          - The format makes sense. Instead of using a vector and using emplace_back function, we're manually moving the pointer in the dyanmically allocated array. We're also doing 2 * i * bytes_per_elem since every element in a Fq2 field is a pair of two 96 byte elements. 