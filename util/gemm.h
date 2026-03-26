#pragma once

// ============================================================
// Cache-Oblivious 矩阵乘法 (REC-MULT)
//
// 参考: Frigo, Leiserson, Prokop, Ramachandran.
//       "Cache-Oblivious Algorithms." ACM Trans. Algorithms, 2012.
//
// 递归分治策略: 沿最大维度二分，直到子问题适配 cache。
// 无需知道 cache 参数（cache-oblivious），自动适配 L1/L2/L3。
//
// Cache 复杂度: Q(m,n,p) = Θ(m+n+p + (mn+np+mp)/B + mnp/B√M)
// 与 cache-aware 的 tiled 算法相同，是理论最优。
// ============================================================

namespace gemm {

// 基础情况阈值 — 子问题所有维度 ≤ THRESHOLD 时用 i-k-j 微内核
// 论文建议 coarsened base case 以减少递归开销
// 32~64 是实践中的甜点区间
static constexpr int THRESHOLD = 64;

// ---- i-k-j 微内核: C[m×p] += A[m×n] * B[n×p] ----
// 操作子矩阵，通过 leading dimension (stride) 描述布局，避免数据拷贝
// i-k-j 循环序: A 按行扫描，B 和 C 内层按行连续访问，cache 友好
// -O2 下编译器可对内层 j 循环做 SIMD 向量化
static inline void micro_gemm(
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

// ---- Cache-oblivious 递归矩阵乘法: C += A * B ----
// A: m×n (leading dim = lda)
// B: n×p (leading dim = ldb)
// C: m×p (leading dim = ldc)
static void rec_mult(
    const float* __restrict__ A, int lda,
    const float* __restrict__ B, int ldb,
    float* __restrict__ C, int ldc,
    int m, int n, int p)
{
    // 基础情况: 所有维度 ≤ THRESHOLD，用微内核
    if (m <= THRESHOLD && n <= THRESHOLD && p <= THRESHOLD) {
        micro_gemm(A, lda, B, ldb, C, ldc, m, n, p);
        return;
    }

    // 沿最大维度切分
    if (m >= n && m >= p) {
        // Case 1: m 最大 — 水平切 A 和 C
        //   C1 += A1 * B
        //   C2 += A2 * B
        int m2 = m / 2;
        rec_mult(A,            lda, B, ldb, C,            ldc, m2,     n, p);
        rec_mult(A + m2 * lda, lda, B, ldb, C + m2 * ldc, ldc, m - m2, n, p);
    } else if (n >= m && n >= p) {
        // Case 2: n 最大 — 切 A 的列和 B 的行，两次累加到同一 C
        //   C += A1 * B1 + A2 * B2
        int n2 = n / 2;
        rec_mult(A,      lda, B,            ldb, C, ldc, m, n2,     p);
        rec_mult(A + n2, lda, B + n2 * ldb, ldb, C, ldc, m, n - n2, p);
    } else {
        // Case 3: p 最大 — 垂直切 B 和 C
        //   C1 += A * B1
        //   C2 += A * B2
        int p2 = p / 2;
        rec_mult(A, lda, B,      ldb, C,      ldc, m, n, p2);
        rec_mult(A, lda, B + p2, ldb, C + p2, ldc, m, n, p - p2);
    }
}

// ---- 顶层接口: C = A * B (连续布局) ----
// A: M×K, B: K×N, C: M×N (所有 row-major 连续)
static inline void matmul(
    const float* A, const float* B, float* C,
    int M, int K, int N)
{
    // C 清零（rec_mult 是累加模式 C += A*B）
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;
    rec_mult(A, K, B, N, C, N, M, K, N);
}

} // namespace gemm
