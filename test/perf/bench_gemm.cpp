// 独立 GEMM benchmark — 可用 cachegrind 分析 cache miss
// 用法:
//   g++ -std=c++17 -O3 -march=native -ffast-math -funroll-loops -I. -o bench_gemm bench_gemm.cpp
//   valgrind --tool=cachegrind ./bench_gemm
//   cg_annotate cachegrind.out.* --auto=yes | head -100
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "util/gemm.h"

static float* alloc_mat(int rows, int cols) {
    float* p = (float*)aligned_alloc(32, sizeof(float) * rows * cols);
    for (int i = 0; i < rows * cols; i++)
        p[i] = (float)rand() / RAND_MAX - 0.5f;
    return p;
}

int main() {
    // 模拟 transformer 中典型的矩阵尺寸
    struct { int M, K, N; const char* name; } shapes[] = {
        {2048, 640, 2560, "linear_forward (2048x640 @ 640x2560)"},
        {2048, 2560, 640, "linear_backward (2048x2560 @ 2560x640)"},
        {640, 2048, 2048, "grad_weight (640x2048 @ 2048x2048)"},
        {2048, 640, 640,  "projection (2048x640 @ 640x640)"},
    };

    for (auto& s : shapes) {
        float* A = alloc_mat(s.M, s.K);
        float* B = alloc_mat(s.K, s.N);
        float* C = alloc_mat(s.M, s.N);

        // warmup
        gemm::matmul(A, B, C, s.M, s.K, s.N);

        auto t0 = std::chrono::high_resolution_clock::now();
        gemm::matmul(A, B, C, s.M, s.K, s.N);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gflops = 2.0 * s.M * s.K * s.N / (ms * 1e6);
        printf("%-45s  %7.1f ms  %5.1f GFLOPS\n", s.name, ms, gflops);

        free(A); free(B); free(C);
    }
    return 0;
}
