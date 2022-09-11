#include <cstdio>
#include <cstdint>

void read_mnt_fq(uint8_t *dest, FILE *inputs);
void read_mnt_fq2(uint8_t *dest, FILE *inputs);
void read_mnt_fq3(uint8_t *dest, FILE *inputs);

void read_mnt4_g2(uint8_t *dest, FILE *inputs);
void read_mnt6_g2(uint8_t *dest, FILE *inputs);

void write_mnt_fq(uint8_t *fq, FILE *outputs);
void write_mnt_fq2(uint8_t *src, FILE *outputs);
void write_mnt_fq3(uint8_t *src, FILE *outputs);
void write_mnt4_g2(uint8_t *src, FILE *outputs);
void write_mnt6_g2(uint8_t *src, FILE *outputs);

void read_mnt4_fq_montgomery(uint8_t *dest, FILE *inputs);
void read_mnt6_fq_montgomery(uint8_t *dest, FILE *inputs);
void read_mnt4_g1_montgomery(uint8_t *dest, FILE *inputs);
void read_mnt6_g1_montgomery(uint8_t *dest, FILE *inputs);
void read_mnt4_g2_montgomery(uint8_t *dest, FILE *inputs);
void read_mnt6_g2_montgomery(uint8_t *dest, FILE *inputs);
void write_mnt4_g1_montgomery(uint8_t *src, FILE *outputs);
void write_mnt6_g1_montgomery(uint8_t *src, FILE *outputs);
void write_mnt4_g2_montgomery(uint8_t *src, FILE *outputs);
void write_mnt6_g2_montgomery(uint8_t *src, FILE *outputs);

void init_libff();

uint8_t *mnt4_g1_to_affine(uint8_t *src);
uint8_t *mnt6_g1_to_affine(uint8_t *src);
uint8_t *mnt4_g2_to_affine(uint8_t *src);
uint8_t *mnt6_g2_to_affine(uint8_t *src);