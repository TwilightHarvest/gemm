#include "gemm/gemm.h"

namespace gemm {

// Textbook i-j-k triple loop: the baseline every other kernel is measured
// against.
void sgemm_naive(int M, int N, int K, const float *A, int lda, const float *B,
                 int ldb, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += A[i * lda + k] * B[k * ldb + j];
      }
      C[i * ldc + j] = acc;
    }
  }
}

} // namespace gemm
