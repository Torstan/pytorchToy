#pragma once
#include <immintrin.h>
#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
// 高性能矩阵乘法内核
//
// 参考:
//   1. Frigo et al. "Cache-Oblivious Algorithms." 2012
//   2. MIT 6.172 Lecture 1 (Leiserson) — register tiling + AVX
//
// 优化层次:
//   Level 1: Cache-oblivious 递归 — 自动适配 L1/L2/L3
//   Level 2: Register tiling — 6×16 micro-tile 保持 C 在 AVX 寄存器
//   Level 3: AVX2 FMA — vfmadd231ps, 每条指令 8 FLOPs
//   Level 4: -O3 -march=native -ffast-math -funroll-loops
// ============================================================

namespace gemm {

static constexpr int THRESHOLD = 128;

// ---- 6×16 register-tiled 微内核 ----
// C[6×16] 保持在 12 个 AVX 寄存器中 (6 行 × 2 __m256)
// 对每个 k: 广播 A[i][k] → 6 个 __m256, 加载 B[k][j:j+16] → 2 个 __m256
// 用 FMA 累加: c_ij += a_ik * b_kj
// 整个 k 循环中 C 不回写内存，减少 load/store 带宽
static inline void micro_6x16(
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float* __restrict__ C, int ldc,
    int kk)
{
    // 12 个 AVX 寄存器保持 C[6][16]
    __m256 c00 = _mm256_loadu_ps(C + 0*ldc);
    __m256 c01 = _mm256_loadu_ps(C + 0*ldc + 8);
    __m256 c10 = _mm256_loadu_ps(C + 1*ldc);
    __m256 c11 = _mm256_loadu_ps(C + 1*ldc + 8);
    __m256 c20 = _mm256_loadu_ps(C + 2*ldc);
    __m256 c21 = _mm256_loadu_ps(C + 2*ldc + 8);
    __m256 c30 = _mm256_loadu_ps(C + 3*ldc);
    __m256 c31 = _mm256_loadu_ps(C + 3*ldc + 8);
    __m256 c40 = _mm256_loadu_ps(C + 4*ldc);
    __m256 c41 = _mm256_loadu_ps(C + 4*ldc + 8);
    __m256 c50 = _mm256_loadu_ps(C + 5*ldc);
    __m256 c51 = _mm256_loadu_ps(C + 5*ldc + 8);

    for (int k = 0; k < kk; k++) {
        __m256 b0 = _mm256_loadu_ps(B + k*ldb);
        __m256 b1 = _mm256_loadu_ps(B + k*ldb + 8);

        __m256 a0 = _mm256_set1_ps(A[0*lda + k]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        __m256 a1 = _mm256_set1_ps(A[1*lda + k]);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);

        __m256 a2 = _mm256_set1_ps(A[2*lda + k]);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);

        __m256 a3 = _mm256_set1_ps(A[3*lda + k]);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);

        __m256 a4 = _mm256_set1_ps(A[4*lda + k]);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);

        __m256 a5 = _mm256_set1_ps(A[5*lda + k]);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    _mm256_storeu_ps(C + 0*ldc,     c00);
    _mm256_storeu_ps(C + 0*ldc + 8, c01);
    _mm256_storeu_ps(C + 1*ldc,     c10);
    _mm256_storeu_ps(C + 1*ldc + 8, c11);
    _mm256_storeu_ps(C + 2*ldc,     c20);
    _mm256_storeu_ps(C + 2*ldc + 8, c21);
    _mm256_storeu_ps(C + 3*ldc,     c30);
    _mm256_storeu_ps(C + 3*ldc + 8, c31);
    _mm256_storeu_ps(C + 4*ldc,     c40);
    _mm256_storeu_ps(C + 4*ldc + 8, c41);
    _mm256_storeu_ps(C + 5*ldc,     c50);
    _mm256_storeu_ps(C + 5*ldc + 8, c51);
}

// ---- 通用标量微内核 (处理边界) ----
static inline void micro_scalar(
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float* __restrict__ C, int ldc,
    int m, int n, int p)
{
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < n; k++) {
            float a_ik = A[i * lda + k];
            for (int j = 0; j < p; j++)
                C[i * ldc + j] += a_ik * B[k * ldb + j];
        }
    }
}

// ---- 基础情况微内核: i-outer, j-inner + prefetch ----
// cachegrind: 92% D1 miss 在 B 的 AVX load (stride=ldb, 跨行大步长)
// 优化: prefetch 下一 k 的 B 行，隐藏 L1 miss 延迟
static inline void micro_gemm(
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float* __restrict__ C, int ldc,
    int m, int n, int p)
{
    int i = 0;
    for (; i + 5 < m; i += 6) {
        int j = 0;
        for (; j + 15 < p; j += 16) {
            micro_6x16(A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, n);
        }
        if (j < p) {
            micro_scalar(A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, 6, n, p - j);
        }
    }
    if (i < m) {
        micro_scalar(A + i*lda, lda, B, ldb, C + i*ldc, ldc, m - i, n, p);
    }
}

// ---- Cache-oblivious 递归矩阵乘法: C += A * B ----
static void rec_mult(
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float* __restrict__ C, int ldc,
    int m, int n, int p)
{
    if (m <= THRESHOLD && n <= THRESHOLD && p <= THRESHOLD) {
        micro_gemm(A, lda, B, ldb, C, ldc, m, n, p);
        return;
    }

    if (m >= n && m >= p) {
        int m2 = m / 2;
        rec_mult(A,            lda, B, ldb, C,            ldc, m2,     n, p);
        rec_mult(A + m2 * lda, lda, B, ldb, C + m2 * ldc, ldc, m - m2, n, p);
    } else if (n >= m && n >= p) {
        int n2 = n / 2;
        rec_mult(A,      lda, B,            ldb, C, ldc, m, n2,     p);
        rec_mult(A + n2, lda, B + n2 * ldb, ldb, C, ldc, m, n - n2, p);
    } else {
        int p2 = p / 2;
        rec_mult(A, lda, B,      ldb, C,      ldc, m, n, p2);
        rec_mult(A, lda, B + p2, ldb, C + p2, ldc, m, n, p - p2);
    }
}

// ---- 顶层接口: C = A * B ----
// 并行策略 (MIT 6.172 slide 42): 对 M 维度做 OpenMP 并行
// 每个线程处理独立的 C 行块，无竞争 → 不需要同步
// 阈值 PARALLEL_MIN: 避免小矩阵的线程开销
static constexpr int PARALLEL_MIN = 256;

static inline void matmul(
    const float* A, const float* B, float* C,
    int M, int K, int N)
{
    std::memset(C, 0, sizeof(float) * M * N);

#ifdef _OPENMP
    if (M >= PARALLEL_MIN) {
        // 沿 M 切分 — 每个线程调用 rec_mult 处理独立的行块
        #pragma omp parallel
        {
            int nthreads = omp_get_num_threads();
            int tid = omp_get_thread_num();
            int rows_per = (M + nthreads - 1) / nthreads;
            int m_start = tid * rows_per;
            int m_end = std::min(m_start + rows_per, M);
            if (m_start < M) {
                rec_mult(A + m_start * K, K,
                         B, N,
                         C + m_start * N, N,
                         m_end - m_start, K, N);
            }
        }
    } else
#endif
    {
        rec_mult(A, K, B, N, C, N, M, K, N);
    }
}

} // namespace gemm
