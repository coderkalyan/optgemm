#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>

#include "/opt/intel/oneapi/mkl/latest/include/mkl.h"

using f32 = float;
namespace chrono = std::chrono;

int main(void) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<f32> dist(0.0f, 100.0f);

    constexpr std::size_t N = 2048;
    constexpr std::size_t M = 2048;
    constexpr std::size_t K = 2048;

    std::vector<f32> A(N * M), B(M * K), C(N * K);
    const auto seed = dist(rng);
    for (size_t i = 0; i < A.size(); i++) A[i] = seed * i;
    for (size_t i = 0; i < B.size(); i++) B[i] = seed * i;

    long min = std::numeric_limits<long>::max();
    for (int i = 0; i < 10; i++) {
        const auto start = chrono::high_resolution_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, M, 1.0f,
                    A.data(), N, B.data(), M, 0.0f, C.data(), N);
        const auto stop = chrono::high_resolution_clock::now();

        const auto duration =
            chrono::duration_cast<chrono::milliseconds>(stop - start);
        printf("iteration %d: %ld ms\n", i + 1, duration.count());

        min = std::min(min, duration.count());
    }

    printf("best: %ld ms\n", min);
}
