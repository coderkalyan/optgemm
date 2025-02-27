const std = @import("std");
const mkl = @cImport({
    @cDefine("MKL_ILP64", "1");
    @cInclude("/opt/intel/oneapi/mkl/latest/include/mkl.h");
});

fn printMatrix(data: [*]f32, rows: usize, cols: usize) void {
    var i: usize = 0;
    for (0..rows) |_| {
        for (0..cols) |_| {
            std.debug.print("{d} ", .{data[i]});
            i += 1;
        }

        std.debug.print("\n", .{});
    }
}

fn gemm0(M: usize, N: usize, K: usize, a: [*]f32, b: [*]f32, c: [*]f32) void {
    for (0..M) |row| {
        for (0..K) |col| {
            var accumulator: f32 = 0.0;
            for (0..N) |inner| accumulator += a[row * N + inner] * b[inner * K + col];
            c[row * K + col] = accumulator;
        }
    }
}

const block_size = 64;
threadlocal var local_a: [block_size][block_size]f32 align(64) = undefined;
threadlocal var local_b: [block_size][block_size]f32 align(64) = undefined;
threadlocal var local_c: [block_size][block_size]f32 align(64) = undefined;

fn gemm1(comptime m: usize, comptime n: usize, comptime k: usize, a_ptr: *align(64) const [m][n]f32, b_ptr: *align(64) const [n][k]f32, noalias c_ptr: *align(64) [m][k]f32) void {
    // should have padding instead but for simplicity assume the sizes are nice
    std.debug.assert(m % block_size == 0);
    std.debug.assert(n % block_size == 0);
    std.debug.assert(k % block_size == 0);

    const m_blocks = m / block_size;
    const n_blocks = n / block_size;
    const k_blocks = k / block_size;
    const a: *align(64) const [m_blocks][block_size][n_blocks][block_size]f32 = @ptrCast(a_ptr);
    const b: *align(64) const [n_blocks][block_size][k_blocks][block_size]f32 = @ptrCast(b_ptr);
    const c: *align(64) [m_blocks][block_size][k_blocks][block_size]f32 = @ptrCast(c_ptr);

    for (0..m_blocks) |bi| {
        for (0..k_blocks) |bj| {
            // clear out local c (accumulator)
            for (0..block_size) |ii| {
                for (0..block_size) |jj|
                    local_c[ii][jj] = 0.0;
            }

            for (0..n_blocks) |bk| {
                // make local copies of a and b blocks to improve locality and prefetching
                for (0..block_size) |ii| {
                    for (0..block_size) |jj| {
                        local_a[ii][jj] = a[bi][ii][bk][jj];
                        local_b[ii][jj] = b[bk][ii][bj][jj];
                    }
                }

                for (0..block_size) |ii| {
                    for (0..block_size) |kk| {
                        const multiplier = local_a[ii][kk];
                        for (0..block_size) |jj|
                            local_c[ii][jj] += multiplier * local_b[kk][jj];
                    }
                }
            }

            for (0..block_size) |ii| {
                for (0..block_size) |jj|
                    c[bi][ii][bj][jj] = local_c[ii][jj];
            }
        }
    }
}

const threadpool_size = 8;

fn gemm2(comptime m: usize, comptime n: usize, comptime k: usize, a_ptr: *align(64) const [m][n]f32, b_ptr: *align(64) const [n][k]f32, noalias c_ptr: *align(64) [m][k]f32) void {
    // should have padding instead but for simplicity assume the sizes are nice
    std.debug.assert(m % block_size == 0);
    std.debug.assert(n % block_size == 0);
    std.debug.assert(k % block_size == 0);

    const m_blocks = m / block_size;
    const chunk = m_blocks / threadpool_size;

    var threads: [threadpool_size]std.Thread = undefined;
    // std.debug.print("m: {} m_blocks: {} chunk: {}\n", .{ m, m_blocks, chunk });
    for (0..threadpool_size) |i| {
        const start = i * chunk;
        const end = if (i == threadpool_size - 1) m_blocks else start + chunk;
        // std.debug.print("{} {}\n", .{ start, end });
        threads[i] = std.Thread.spawn(.{}, ugemm2, .{ m, n, k, a_ptr, b_ptr, c_ptr, start, end }) catch unreachable;
    }

    for (threads) |thread| thread.join();
}

inline fn ugemm2(comptime m: usize, comptime n: usize, comptime k: usize, a_ptr: *align(64) const [m][n]f32, b_ptr: *align(64) const [n][k]f32, noalias c_ptr: *align(64) [m][k]f32, start: usize, end: usize) void {
    // should have padding instead but for simplicity assume the sizes are nice
    std.debug.assert(m % block_size == 0);
    std.debug.assert(n % block_size == 0);
    std.debug.assert(k % block_size == 0);

    const m_blocks = m / block_size;
    const n_blocks = n / block_size;
    const k_blocks = k / block_size;
    const a: *align(64) const [m_blocks][block_size][n_blocks][block_size]f32 = @ptrCast(a_ptr);
    const b: *align(64) const [n_blocks][block_size][k_blocks][block_size]f32 = @ptrCast(b_ptr);
    const c: *align(64) [m_blocks][block_size][k_blocks][block_size]f32 = @ptrCast(c_ptr);

    for (start..end) |bi| {
        for (0..k_blocks) |bj| {
            // clear out local c (accumulator)
            for (0..block_size) |ii| {
                for (0..block_size) |jj|
                    local_c[ii][jj] = 0.0;
            }

            for (0..n_blocks) |bk| {
                // make local copies of a and b blocks to improve locality and prefetching
                for (0..block_size) |ii| {
                    for (0..block_size) |jj| {
                        local_a[ii][jj] = a[bi][ii][bk][jj];
                        local_b[ii][jj] = b[bk][ii][bj][jj];
                    }
                }

                for (0..block_size) |ii| {
                    for (0..block_size) |kk| {
                        const multiplier = local_a[ii][kk];
                        for (0..block_size) |jj|
                            local_c[ii][jj] += multiplier * local_b[kk][jj];
                    }
                }
            }

            for (0..block_size) |ii| {
                for (0..block_size) |jj|
                    c[bi][ii][bj][jj] = local_c[ii][jj];
            }
        }
    }
}

pub fn main() !void {
    var allocator = std.heap.GeneralPurposeAllocator(.{}){};
    var gpa = allocator.allocator();

    // const N: usize = 512;
    // const M: usize = 512;
    // const K: usize = 512;
    const M: usize = 2048;
    const N: usize = M;
    const K: usize = M;

    const alpha = 1.0;
    const beta = 0.0;

    const a = try gpa.alignedAlloc(f32, 64, M * N * 2);
    const b = try gpa.alignedAlloc(f32, 64, N * K * 2);
    const c = try gpa.alignedAlloc(f32, 64, M * K * 2);
    const d = try gpa.alignedAlloc(f32, 64, M * K * 2);
    // var rand = std.Random.init(0);
    // const seed = rand.float(f32);
    const seed = 1.0;
    for (0..a.len) |i| a[i] = seed * @as(f32, @floatFromInt(i % 10));
    for (0..b.len) |i| b[i] = seed * @as(f32, @floatFromInt(i % 10));
    @memset(c, 0.0);

    {
        const start = std.time.microTimestamp();
        mkl.cblas_sgemm(mkl.CblasRowMajor, mkl.CblasNoTrans, mkl.CblasNoTrans, M, K, N, alpha, a.ptr, M, b.ptr, K, beta, c.ptr, K);
        // gemm0(N, M, K, @ptrCast(a.ptr), @ptrCast(b.ptr), @ptrCast(c.ptr));
        const end = std.time.microTimestamp();
        const duration = end - start;
        std.debug.print("evaluating reference took {}us (no warmup)\n", .{duration});
    }

    var discrepancy: f32 = 0.0;
    gemm2(M, N, K, @ptrCast(a.ptr), @ptrCast(b.ptr), @ptrCast(d.ptr));
    for (c, d) |ref, res| discrepancy += @abs(ref - res);
    std.debug.print("evaluating test function, discrepancy: {d}\n", .{discrepancy});

    var min: i64 = std.math.maxInt(i64);
    for (0..10) |i| {
        const start = std.time.milliTimestamp();
        gemm2(M, N, K, @ptrCast(a.ptr), @ptrCast(b.ptr), @ptrCast(d.ptr));
        const end = std.time.milliTimestamp();
        const duration = end - start;
        std.debug.print("iteration {}: {}ms\n", .{ i + 1, duration });
        min = @min(min, duration);
    }

    std.debug.print("best: {}ms\n", .{min});

    // printMatrix(c.ptr, N, K);
}
