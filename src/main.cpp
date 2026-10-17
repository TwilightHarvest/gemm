// Minimal SGEMM benchmark harness.
//
// For each problem size, every kernel is checked against OpenBLAS and then timed.
// Reported GFLOP/s uses 2*M*N*K flops; "best" is the fastest repetition, which is
// the most stable number for comparing kernels on a quiet machine.
//
// Usage: gemm_bench [options] [size ...]
//   size            N (square) or MxNxK; default: 64 128 256 512 1024
//   -k, --kernel S  run only kernels whose name contains S (repeatable)
//   -t, --threads N OpenBLAS thread count (default 1, to match our kernels)
//   --min-time S    minimum timed seconds per kernel/size (default 0.5)
//   --csv           machine-readable output

#include <cblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "gemm/gemm.h"

namespace {

struct Kernel {
  const char* name;
  gemm::SgemmFn fn;
  long max_flops;  // skip sizes above this (keeps slow kernels from taking minutes)
};

void sgemm_openblas(int M, int N, int K, const float* A, int lda, const float* B, int ldb,
                    float* C, int ldc) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, lda, B, ldb, 0.0f,
              C, ldc);
}

// Add new kernels here.
const Kernel kKernels[] = {
    {"openblas", sgemm_openblas, std::numeric_limits<long>::max()},
    {"naive", gemm::sgemm_naive, 2L * 1024 * 1024 * 1024},
};

struct Size {
  int M, N, K;
};

struct FreeDeleter {
  void operator()(float* p) const { std::free(p); }
};
using Buffer = std::unique_ptr<float[], FreeDeleter>;

// 64-byte aligned so kernels can assume cache-line alignment of the base pointer.
Buffer alloc_matrix(std::size_t count) {
  std::size_t bytes = (count * sizeof(float) + 63) / 64 * 64;
  auto* p = static_cast<float*>(std::aligned_alloc(64, bytes));
  if (!p) {
    std::fprintf(stderr, "allocation of %zu bytes failed\n", bytes);
    std::exit(1);
  }
  return Buffer(p);
}

void fill_random(float* p, std::size_t count, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < count; ++i) p[i] = dist(rng);
}

double now_seconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

bool parse_size(const char* s, Size& out) {
  int m, n, k;
  char tail;
  if (std::sscanf(s, "%dx%dx%d%c", &m, &n, &k, &tail) == 3) {
    out = {m, n, k};
  } else if (std::sscanf(s, "%d%c", &n, &tail) == 1) {
    out = {n, n, n};
  } else {
    return false;
  }
  return out.M > 0 && out.N > 0 && out.K > 0;
}

[[noreturn]] void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [-k kernel]... [-t threads] [--min-time s] [--csv] [N | MxNxK]...\n"
               "kernels:",
               argv0);
  for (const auto& k : kKernels) std::fprintf(stderr, " %s", k.name);
  std::fprintf(stderr, "\n");
  std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<Size> sizes;
  std::vector<std::string> filters;
  int threads = 1;
  double min_time = 0.5;
  bool csv = false;

  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    auto next = [&] {
      if (i + 1 >= argc) usage(argv[0]);
      return argv[++i];
    };
    if (!std::strcmp(a, "-k") || !std::strcmp(a, "--kernel")) {
      filters.emplace_back(next());
    } else if (!std::strcmp(a, "-t") || !std::strcmp(a, "--threads")) {
      threads = std::atoi(next());
    } else if (!std::strcmp(a, "--min-time")) {
      min_time = std::atof(next());
    } else if (!std::strcmp(a, "--csv")) {
      csv = true;
    } else {
      Size s;
      if (!parse_size(a, s)) usage(argv[0]);
      sizes.push_back(s);
    }
  }
  if (sizes.empty()) {
    for (int n : {64, 128, 256, 512, 1024}) sizes.push_back({n, n, n});
  }

  std::vector<const Kernel*> kernels;
  for (const auto& k : kKernels) {
    bool match = filters.empty();
    for (const auto& f : filters) match |= std::strstr(k.name, f.c_str()) != nullptr;
    if (match) kernels.push_back(&k);
  }
  if (kernels.empty()) usage(argv[0]);

  openblas_set_num_threads(threads);

  if (csv) {
    std::printf("kernel,M,N,K,best_gflops,median_gflops,reps,max_abs_err,ok\n");
  } else {
    std::printf("openblas threads: %d\n", threads);
    std::printf("%-10s %16s %10s %10s %8s %6s %10s\n", "kernel", "MxNxK", "best GF/s",
                "med GF/s", "%blas", "reps", "max err");
  }

  std::mt19937 rng(42);
  bool all_ok = true;

  for (const Size& s : sizes) {
    const auto [M, N, K] = s;
    const double flops = 2.0 * M * N * K;
    auto A = alloc_matrix(std::size_t(M) * K);
    auto B = alloc_matrix(std::size_t(K) * N);
    auto C = alloc_matrix(std::size_t(M) * N);
    auto Cref = alloc_matrix(std::size_t(M) * N);
    fill_random(A.get(), std::size_t(M) * K, rng);
    fill_random(B.get(), std::size_t(K) * N, rng);
    sgemm_openblas(M, N, K, A.get(), K, B.get(), N, Cref.get(), N);

    // Entries are in [-1, 1], so |c| <= K and a forward-error bound is ~K * eps.
    const float tol = 2.0f * K * std::numeric_limits<float>::epsilon();
    double blas_best = 0.0;
    char dims[48];
    std::snprintf(dims, sizeof dims, "%dx%dx%d", M, N, K);

    for (const Kernel* k : kernels) {
      if (flops > double(k->max_flops)) {
        if (!csv) std::printf("%-10s %16s %10s\n", k->name, dims, "skipped");
        continue;
      }

      // Correctness: poison C first so a kernel that skips elements is caught.
      std::fill_n(C.get(), std::size_t(M) * N, std::numeric_limits<float>::quiet_NaN());
      k->fn(M, N, K, A.get(), K, B.get(), N, C.get(), N);
      float max_err = 0.0f;
      for (std::size_t i = 0; i < std::size_t(M) * N; ++i) {
        float e = std::fabs(C[i] - Cref[i]);
        if (!(e <= max_err)) max_err = e;  // also propagates NaN
      }
      const bool ok = max_err <= tol;
      all_ok &= ok;

      // Timing: at least 3 reps and min_time seconds (the correctness run doubles as warmup).
      std::vector<double> times;
      double total = 0.0;
      while (times.size() < 3 || total < min_time) {
        double t0 = now_seconds();
        k->fn(M, N, K, A.get(), K, B.get(), N, C.get(), N);
        double dt = now_seconds() - t0;
        times.push_back(dt);
        total += dt;
      }
      std::sort(times.begin(), times.end());
      const double best = flops / times.front() * 1e-9;
      const double median = flops / times[times.size() / 2] * 1e-9;
      if (std::strcmp(k->name, "openblas") == 0) blas_best = best;

      if (csv) {
        std::printf("%s,%d,%d,%d,%.3f,%.3f,%zu,%.3g,%d\n", k->name, M, N, K, best, median,
                    times.size(), max_err, ok);
      } else {
        char pct[16] = "-";
        if (blas_best > 0) std::snprintf(pct, sizeof pct, "%.1f%%", 100.0 * best / blas_best);
        std::printf("%-10s %16s %10.2f %10.2f %8s %6zu %10.2g%s\n", k->name, dims, best, median,
                    pct, times.size(), max_err, ok ? "" : "  FAIL");
      }
      std::fflush(stdout);
    }
  }

  return all_ok ? 0 : 1;
}
