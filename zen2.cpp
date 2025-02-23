#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>

#include "/opt/intel/oneapi/mkl/latest/include/mkl.h"

using f32 = float;
namespace chrono = std::chrono;

void sgemm0(std::size_t N, std::size_t M, std::size_t K, f32 alpha, f32 *A,
            f32 *B, f32 beta, f32 *C) {
    for (std::size_t row = 0; row < N; row++) {
        for (std::size_t col = 0; col < K; col++) {
            C[row * K + col] = beta * C[row * K + col];
            for (std::size_t inner = 0; inner < M; inner++) {
                C[row * K + col] +=
                    alpha * A[row * M + inner] * B[inner * K + col];
            }
        }
    }
}

void print_matrix(f32 *data, std::size_t rows, std::size_t cols) {
    int i = 0;
    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            printf("%.1f ", data[i++]);
        }

        printf("\n");
    }
}

int main(void) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<f32> dist(0.0f, 100.0f);

    constexpr std::size_t N = 512;
    constexpr std::size_t M = 512;
    constexpr std::size_t K = 512;
    const f32 alpha = 1.0f;
    const f32 beta = 0.0f;

    std::vector<f32> A(N * M), B(M * K), C(N * K), D(N * K);
    const auto seed = 1.0; // dist(rng);
    for (std::size_t i = 0; i < A.size(); i++) A[i] = seed * (i % 10);
    for (std::size_t i = 0; i < B.size(); i++) B[i] = seed * (i % 10);
    // print_matrix(A.data(), N, M);
    // printf("\n");
    // print_matrix(B.data(), M, K);
    // printf("\n");

    // first check if correct
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, alpha,
                A.data(), N, B.data(), K, beta, C.data(), K);
    sgemm0(N, M, K, 1.0f, A.data(), B.data(), 0.0f, D.data());
    // print_matrix(C.data(), N, K);
    // printf("\n");

    assert(C.size() == D.size());
    for (std::size_t i = 0; i < C.size(); i++) {
        if (std::fabs(C[i] - D[i]) > std::numeric_limits<f32>::epsilon()) {
            printf("incorrect value at element %ld: ref = %f, test = %f\n", i,
                   C[i], D[i]);
            return 0;
        }
    }

    long min = std::numeric_limits<long>::max();
    for (int iteration = 0; iteration < 10; iteration++) {
        const auto start = chrono::high_resolution_clock::now();
        sgemm0(N, M, K, alpha, A.data(), B.data(), beta, C.data());
        const auto stop = chrono::high_resolution_clock::now();
        const auto duration =
            chrono::duration_cast<chrono::milliseconds>(stop - start);

        // compare result again, mostly to make sure clang doesn't optimize out
        // the entire operation
        for (std::size_t i = 0; i < C.size(); i++) {
            if (std::fabs(C[i] - D[i]) > std::numeric_limits<f32>::epsilon()) {
                printf("iteration %d incorrect value at element %ld: ref = %f, "
                       "test = %f\n",
                       iteration, i, D[i], C[i]);
                return 0;
            }
        }

        printf("iteration %d: %ld ms\n", iteration + 1, duration.count());
        min = std::min(min, duration.count());
    }

    printf("best: %ld ms\n", min);
}
