// matmul 单元测试
// 编译: make test/test_gemm && ./test/test_gemm
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include "util/gemm.h"

static float* alloc_mat(int rows, int cols) {
    float* p = (float*)aligned_alloc(32, sizeof(float) * rows * cols);
    for (int i = 0; i < rows * cols; i++)
        p[i] = (float)rand() / RAND_MAX - 0.5f;
    return p;
}

static float* alloc_zero(int rows, int cols) {
    float* p = (float*)aligned_alloc(32, sizeof(float) * rows * cols);
    std::memset(p, 0, sizeof(float) * rows * cols);
    return p;
}

// 朴素 O(n^3) 矩阵乘法作为参考实现
static void naive_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float a = A[i * K + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += a * B[k * N + j];
        }
}

// 检查两个矩阵是否近似相等
static bool check_close(const float* A, const float* B, int size, float rtol, float atol) {
    for (int i = 0; i < size; i++) {
        float diff = std::fabs(A[i] - B[i]);
        float tol = atol + rtol * std::fabs(B[i]);
        if (diff > tol) {
            printf("  MISMATCH at [%d]: got %.6f, expected %.6f, diff=%.6e\n", i, A[i], B[i], diff);
            return false;
        }
    }
    return true;
}

static int passed = 0, failed = 0;

static void run_test(const char* name, int M, int K, int N) {
    printf("%-50s ", name);

    srand(42);  // 固定种子保证可复现
    float* A = alloc_mat(M, K);
    float* B = alloc_mat(K, N);
    float* C_ref = alloc_zero(M, N);
    float* C_test = alloc_zero(M, N);

    naive_matmul(A, B, C_ref, M, K, N);
    gemm::matmul(A, B, C_test, M, K, N);

    // -ffast-math 会引入微小误差，用宽松容差
    bool ok = check_close(C_test, C_ref, M * N, 1e-3f, 1e-4f);
    if (ok) {
        printf("PASS\n");
        passed++;
    } else {
        printf("FAIL\n");
        failed++;
    }

    free(A); free(B); free(C_ref); free(C_test);
}

int main() {
    printf("=== gemm::matmul 单元测试 ===\n\n");

    // 1. 边界情况: 1×1
    run_test("1x1x1 (最小矩阵)", 1, 1, 1);

    // 2. 小矩阵: 不触发 AVX 微内核
    run_test("3x4x5 (纯标量路径)", 3, 4, 5);

    // 3. 刚好触发 6×16 微内核
    run_test("6x8x16 (单个 6x16 tile)", 6, 8, 16);

    // 4. 多个完整 tile
    run_test("12x8x32 (2x2 tiles)", 12, 8, 32);

    // 5. 不对齐 6/16 的边界
    run_test("7x9x17 (余数行+余数列)", 7, 9, 17);
    run_test("13x10x19 (余数行+余数列)", 13, 10, 19);

    // 6. 单行/单列
    run_test("1x100x1 (向量内积)", 1, 100, 1);
    run_test("100x1x100 (外积)", 100, 1, 100);
    run_test("1x100x100 (单行 × 矩阵)", 1, 100, 100);

    // 7. 刚好等于 THRESHOLD (128)
    run_test("128x128x128 (= THRESHOLD)", 128, 128, 128);

    // 8. 超过 THRESHOLD, 触发递归分治
    run_test("129x129x129 (> THRESHOLD, 递归)", 129, 129, 129);
    run_test("256x256x256 (多层递归)", 256, 256, 256);

    // 9. 非方阵 + 递归
    run_test("200x50x300 (宽矩阵)", 200, 50, 300);
    run_test("50x300x200 (高矩阵)", 50, 300, 200);

    // 10. Transformer 典型尺寸 (缩小版, 加速测试)
    run_test("256x64x256 (mini transformer)", 256, 64, 256);
    run_test("512x128x512 (small transformer)", 512, 128, 512);

#ifdef _OPENMP
    // 11. 大矩阵触发 OpenMP 路径 (M >= PARALLEL_MIN=256)
    run_test("512x256x512 (OpenMP path)", 512, 256, 512);
#endif

    printf("\n=== 结果: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
