#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <limits>
#include <random>

#include "/opt/intel/oneapi/mkl/latest/include/mkl.h"

using f32 = float;
namespace chrono = std::chrono;

void sgemm0(std::size_t N, std::size_t M, std::size_t K, f32 alpha, f32 *A,
            f32 *B, f32 beta, f32 *C) {
    for (std::size_t row = 0; row < N; row++) {
        for (std::size_t col = 0; col < K; col++) {
            f32 accumulator = beta * C[row * K + col];
            for (std::size_t inner = 0; inner < M; inner++) {
                accumulator += alpha * A[row * M + inner] * B[inner * K + col];
            }
        }
    }
}

void sgemm1(std::size_t N, std::size_t M, std::size_t K, f32 alpha, f32 *A,
            f32 *B, f32 beta, f32 *C) {
    constexpr std::size_t TILE = 64;

    const auto n_tiles = (N + TILE - 1) / TILE;
    const auto m_tiles = (M + TILE - 1) / TILE;
    const auto k_tiles = (K + TILE - 1) / TILE;

    for (std::size_t tiler = 0; tiler < n_tiles; tiler++) {
        for (std::size_t tilec = 0; tilec < k_tiles; tilec++) {
            // scale C for entire tile by beta
            for (std::size_t i = 0; i < TILE; i++) {
                for (std::size_t j = 0; j < TILE; j++) {
                    const auto row = tiler * TILE + i;
                    const auto col = tilec * TILE + j;
                    if ((row >= N) || (col >= K)) continue;
                    C[row * K + col] = beta * C[row * K + col];
                }
            }

            // compute tiled inner product
            for (std::size_t tilei = 0; tilei < m_tiles; tilei++) {
                for (std::size_t i = 0; i < TILE; i++) {
                    const auto row = tiler * TILE + i;
                    if ((row >= N)) continue;

                    for (std::size_t j = 0; j < TILE; j++) {
                        const auto col = tilec * TILE + j;
                        if ((col >= K)) continue;

                        f32 accumulator = 0.0f;
                        for (std::size_t k = 0; k < TILE; k++) {
                            const auto inner = tilei * TILE + k;
                            accumulator +=
                                alpha * A[row * M + inner] * B[inner * K + col];
                        }
                        C[row * K + col] += accumulator;
                    }
                }
            }
        }
    }
}

void sgemm2(std::size_t N, std::size_t M, std::size_t K, f32 alpha, f32 *A,
            f32 *B, f32 beta, f32 *C) {
    constexpr std::size_t TILE = 32;

    const auto n_tiles = (N + TILE - 1) / TILE;
    const auto m_tiles = (M + TILE - 1) / TILE;
    const auto k_tiles = (K + TILE - 1) / TILE;

    std::fill(C, C + (N * K), 0.0f);

#pragma omp parallel for
    for (std::size_t tiler = 0; tiler < n_tiles; tiler++) {
        for (std::size_t tilec = 0; tilec < k_tiles; tilec++) {
            // compute tiled inner product
            for (std::size_t tilei = 0; tilei < m_tiles; tilei++) {
                for (std::size_t i = 0; i < TILE; i++) {
                    const auto row = tiler * TILE + i;
                    if ((row >= N)) continue;

                    for (std::size_t j = 0; j < TILE; j++) {
                        const auto col = tilec * TILE + j;
                        if ((col >= K)) continue;

                        f32 accumulator = 0.0f;
                        for (std::size_t k = 0; k < TILE; k++) {
                            const auto inner = tilei * TILE + k;
                            accumulator +=
                                A[row * M + inner] * B[inner * K + col];
                        }
                        C[row * K + col] += accumulator;
                    }
                }
            }
        }
    }
}

void sgemm3(std::size_t N, std::size_t M, std::size_t K, f32 alpha, f32 *A,
            f32 *B, f32 beta, f32 *C) {
    constexpr std::size_t TILE = 32;

    const auto n_tiles = (N + TILE - 1) / TILE;
    const auto m_tiles = (M + TILE - 1) / TILE;
    const auto k_tiles = (K + TILE - 1) / TILE;

    std::fill(C, C + (N * K), 0.0f);

#pragma omp parallel for
    for (std::size_t tiler = 0; tiler < n_tiles; tiler++) {
        for (std::size_t tilec = 0; tilec < k_tiles; tilec++) {
            // compute tiled inner product
            for (std::size_t tilei = 0; tilei < m_tiles; tilei++) {
                for (std::size_t i = 0; i < TILE; i++) {
                    const auto row = tiler * TILE + i;
                    if (row >= N) continue;

                    for (std::size_t k = 0; k < TILE; k++) {
                        const auto inner = tilei * TILE + k;
                        if (inner >= M) continue;

                        for (std::size_t j = 0; j < TILE; j += 8) {
                            // if (col >= K) continue;

                            const auto col = tilec * TILE + j;
                            // for (std::size_t l = 0; l < 8; l += 1) {
                            //     const auto col = tilec * TILE + (j + l);
                            //     C[row * K + col] +=
                            //         A[row * M + inner] * B[inner * K + col];
                            // }
                            const __m256d a =
                                _mm256_loadu_ps(&A[row * M + inner]);
                            const __m256d b =
                                _mm256_loadu_ps(&B[inner * K + col]);
                            const __m256d c =
                                _mm256_loadu_ps(&C[row * K + col]);
                            const __m256d fma = _mm256_fmadd_ps(a, b, c);
                            _mm256_storeu_ps(&C[row * K + col], fma);
                        }
                    }
                }
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
    sgemm3(N, M, K, 1.0f, A.data(), B.data(), 0.0f, D.data());
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
        sgemm3(N, M, K, alpha, A.data(), B.data(), beta, C.data());
        const auto stop = chrono::high_resolution_clock::now();
        const auto duration =
            chrono::duration_cast<chrono::microseconds>(stop - start);

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

        printf("iteration %d: %ld us\n", iteration + 1, duration.count());
        min = std::min(min, duration.count());
    }

    printf("best: %ld us\n", min);
}
