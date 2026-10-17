#pragma once

// Single-precision GEMM kernels.
//
// All kernels compute C = A * B for row-major matrices:
//   A is M x K (leading dimension lda >= K)
//   B is K x N (leading dimension ldb >= N)
//   C is M x N (leading dimension ldc >= N), overwritten.

namespace gemm {

using SgemmFn = void (*)(int M, int N, int K,
                         const float* A, int lda,
                         const float* B, int ldb,
                         float* C, int ldc);

void sgemm_naive(int M, int N, int K,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float* C, int ldc);

}  // namespace gemm
