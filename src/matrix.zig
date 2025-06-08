//! Module containing matrix operations.
//! Matrices can be initialized using as follows:
//! ```zig
//! A = try Matrix(f32).init(allocator, 3, 3);
//! ```
//! Do not forget to de-initialize the matrices.
//! ```zig
//! A.deinit();
//! ```
//!

const std = @import("std");
const Allocator = std.mem.Allocator;

const MatrixError = error{
    SizeMismatch,
    SingularMatrix,
};

/// A matrix based on 2D array.
/// Type T must be provided at compile time.
pub fn Matrix(comptime T: type) type {
    return struct {
        const Self = @This();

        size: std.meta.Tuple(&.{ usize, usize }),
        array: [][]T,
        allocator: Allocator,

        /// Creates a matrix of size n x m.
        /// The matrix will be allocated on the heap using the provided allocator.
        /// This means that the matrix must be deallocated to avoid memory leaks.
        pub fn init(allocator: Allocator, n: usize, m: usize) !Self {
            const array = try allocator.alloc([]T, n);
            for (array) |*row| {
                row.* = try allocator.alloc(T, m);
            }

            return .{
                .size = .{ n, m },
                .array = array,
                .allocator = allocator,
            };
        }

        /// Destroy the matrix by releasing the 2D array from the memory
        pub fn deinit(self: *const Self) void {
            for (self.array) |row| {
                self.allocator.free(row);
            }
            self.allocator.free(self.array);
        }

        /// Uses the provided slice to populate the matrix.
        /// If the size of the slice does not match that of the matrix, a SizeMismatch error is returned
        pub fn fromSlice(self: *const Self, slice: []const []const T) !void {
            if (slice.len == 0) {
                return MatrixError.SizeMismatch;
            }

            if (slice.len != self.size[0] or slice[0].len != self.size[1]) {
                return MatrixError.SizeMismatch;
            }

            for (0..self.size[0]) |i| {
                for (0..self.size[1]) |j| {
                    self.set(i, j, slice[i][j]);
                }
            }
        }

        /// Takes in another matrix and returns true if both matrices equal.
        /// Two matrices are equal if they are equal element wise.
        pub fn equals(self: *const Self, other: Matrix(T)) !bool {
            if (self.size[0] != other.size[0] or self.size[1] != other.size[1]) {
                return MatrixError.SizeMismatch;
            }

            for (0..self.size[0]) |i| {
                for (0..self.size[1]) |j| {
                    if (self.get(i, j) != other.get(i, j)) {
                        return false;
                    }
                }
            }
            return true;
        }

        /// Prints the content of the matrix.
        /// Each row will be printed on a separate row.
        pub fn print(self: *const Self) void {
            for (0..self.size[0]) |i| {
                for (0..self.size[1]) |j| {
                    std.debug.print("{} ", .{self.get(i, j)});
                }
                std.debug.print("\n", .{});
            }
        }

        /// Fills the entire matrix with val.
        pub fn fill(self: *const Self, val: T) void {
            for (0..self.size[0]) |i| {
                for (0..self.size[1]) |j| {
                    self.array[i][j] = val;
                }
            }
        }

        /// Get the element at i-th row and j-th column
        pub inline fn get(self: *const Self, i: usize, j: usize) T {
            return self.array[i][j];
        }

        /// Set the element at i-th row and j-th column to val
        pub inline fn set(self: *const Self, i: usize, j: usize, val: T) void {
            self.array[i][j] = val;
        }

        /// Performs matrix addition.
        /// If the two matrices do not have the same size, a SizeMismatch error is returned.
        pub fn add(self: *const Self, other: Matrix(T)) !Matrix(T) {
            if (self.size[0] != other.size[0] or self.size[1] != other.size[1]) {
                return MatrixError.SizeMismatch;
            }
            const n, const m = self.size;

            const result = try Matrix(T).init(self.allocator, n, m);
            for (0..n) |i| {
                for (0..m) |j| {
                    result.set(i, j, self.get(i, j) + other.get(i, j));
                }
            }

            return result;
        }

        /// Performs matrix subtraction.
        /// If the two matrices do not have the same size, a SizeMismatch error is returned.
        pub fn sub(self: *const Self, other: Matrix(T)) !Matrix(T) {
            if (self.size[0] != other.size[0] or self.size[1] != other.size[1]) {
                return MatrixError.SizeMismatch;
            }
            const n, const m = self.size;

            const result = try Matrix(T).init(self.allocator, n, m);
            for (0..n) |i| {
                for (0..m) |j| {
                    result.set(i, j, self.get(i, j) - other.get(i, j));
                }
            }

            return result;
        }

        /// Performs scalar multiplication.
        pub fn scalarMult(self: *const Self, scalar: T) !Matrix(T) {
            const n, const m = self.size;
            const result = try Matrix(T).init(self.allocator, n, m);

            for (0..n) |i| {
                for (0..m) |j| {
                    result.set(i, j, scalar * self.get(i, j));
                }
            }

            return result;
        }

        /// Performs matrix multiplication.
        /// If the number of columns of the current matrix does not equal number of rows of the provided matrix, a SizeMismatch error is returned.
        pub fn mult(self: *const Self, other: Matrix(T)) !Matrix(T) {
            const a, const b = other.size;
            if (a != self.size[1]) {
                return MatrixError.SizeMismatch;
            }

            const result = try Matrix(T).init(self.allocator, self.size[0], b);
            for (0..self.size[0]) |i| {
                for (0..b) |j| {
                    var sum: T = 0;
                    for (0..self.size[1]) |k| {
                        sum += self.get(i, k) * other.get(k, j);
                    }
                    result.set(i, j, sum);
                }
            }

            return result;
        }

        /// Returns two matrices L and U which are the result of the LU decomposition of the matrix.
        /// Uses Crout's method for matrix decomposition.
        /// If the matrix is not a square matrix, a SizeMismatch error is returned.
        pub fn LU(self: *const Self) !std.meta.Tuple(&.{ Matrix(f32), Matrix(f32) }) {
            if (self.size[0] != self.size[1]) {
                return MatrixError.SizeMismatch;
            }
            const n = self.size[0];
            var sum: f32 = 0;

            const L = try Matrix(f32).init(self.allocator, n, n);
            const U = try Matrix(f32).init(self.allocator, n, n);
            L.fill(0);
            U.fill(0);

            for (0..n) |k| {
                U.set(k, k, 1);

                for (k..n) |j| {
                    sum = 0;
                    for (0..k) |i| {
                        sum += L.get(j, i) * U.get(i, k);
                    }
                    L.set(j, k, self.get(j, k) - sum);
                }

                for (k + 1..n) |j| {
                    sum = 0;
                    for (0..k) |i| {
                        sum += L.get(k, i) * U.get(i, j);
                    }
                    U.set(k, j, (self.get(k, j) - sum) / L.get(k, k));
                }
            }

            return .{ L, U };
        }

        /// Returns the determinant of the matrix.
        /// If the matrix is not a square matrix, a SizeMismatch error is returned.
        pub fn det(self: *const Self) !f32 {
            var ans: f32 = 1;

            const L, const U = try self.LU();
            defer {
                L.deinit();
                U.deinit();
            }
            const n = self.size[0];

            for (0..n) |i| {
                ans *= L.get(i, i);
            }
            for (0..n) |i| {
                ans *= U.get(i, i);
            }

            return ans;
        }

        /// Returns the inverse matrix.
        /// NOTE: Calculating the inverse matrix should be avoided when possible due to poor numerical stability.
        /// If the matrix is not a square matrix, a SizeMismatch error is returned.
        pub fn inv(self: *const Self) !Matrix(f32) {
            if (self.size[0] != self.size[1]) {
                return MatrixError.SizeMismatch;
            }
            const n = self.size[0];

            const L, const U = try self.LU();
            defer {
                L.deinit();
                U.deinit();
            }

            const L_inv = try Matrix(f32).init(self.allocator, n, n);
            const U_inv = try Matrix(f32).init(self.allocator, n, n);
            defer {
                L_inv.deinit();
                U_inv.deinit();
            }
            L_inv.fill(0);
            U_inv.fill(0);

            for (0..n) |i| {
                U_inv.set(i, i, 1);
            }

            for (0..n) |i| {
                for (n - i..n) |j| {
                    var sum: f32 = 0;
                    for (n - i..j + 1) |k| {
                        sum += U.get(n - 1 - i, k) * U_inv.get(k, j);
                    }
                    U_inv.set(n - 1 - i, j, -sum);
                }
            }

            for (0..n) |i| {
                L_inv.set(i, i, 1.0 / L.get(i, i));

                for (i + 1..n) |j| {
                    var sum: T = 0;
                    for (i..j) |k| {
                        sum += L.get(j, k) * L_inv.get(k, i);
                    }
                    L_inv.set(j, i, -sum / L.get(j, j));
                }
            }

            return U_inv.mult(L_inv);
        }

        /// Returns the transpose of the matrix.
        pub fn transpose(self: *const Self) Matrix(T) {
            const result = try Matrix(T).init(self.allocator, self.size[0], self.size[1]);
            for (0..self.size[0]) |i| {
                for (0..self.size[1]) |j| {
                    result.set(i, j, self.get(j, j));
                }
            }

            return result;
        }
    };
}

/// Returns the identity matrix of size n x n.
pub fn I(comptime T: type, n: usize, allocator: Allocator) !Matrix(T) {
    const matrix = try Matrix(T).init(allocator, n, n);
    for (0..n) |i| {
        for (0..n) |j| {
            if (i == j) {
                matrix.set(i, j, 1);
            } else {
                matrix.set(i, j, 0);
            }
        }
    }

    return matrix;
}

test "Matrix filled with 1s" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leak_check = gpa.deinit();
        if (leak_check == .leak) @panic("FAIL");
    }
    const allocator = gpa.allocator();

    const A = try Matrix(i32).init(allocator, 2, 2);
    defer A.deinit();
    A.fill(1);

    const expected = try Matrix(i32).init(allocator, 2, 2);
    defer expected.deinit();
    try expected.fromSlice(&[_][]const i32{
        &[_]i32{ 1, 1 },
        &[_]i32{ 1, 1 },
    });

    try std.testing.expect(try A.equals(expected));
}

test "Matrix from array" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leak_check = gpa.deinit();
        if (leak_check == .leak) @panic("FAIL");
    }
    const allocator = gpa.allocator();

    const A = try Matrix(f32).init(allocator, 2, 2);
    defer A.deinit();
    try A.fromSlice(&[_][]const f32{
        &[_]f32{ 4, 3 },
        &[_]f32{ 6, 3 },
    });

    try std.testing.expect(A.get(0, 0) == 4 and A.get(0, 1) == 3 and A.get(1, 0) == 6 and A.get(1, 1) == 3);
}

test "Matrix multiplication" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leak_check = gpa.deinit();
        if (leak_check == .leak) @panic("FAIL");
    }
    const allocator = gpa.allocator();

    const A = try Matrix(i32).init(allocator, 2, 3);
    const B = try Matrix(i32).init(allocator, 3, 2);
    defer {
        A.deinit();
        B.deinit();
    }

    try A.fromSlice(&[_][]const i32{
        &[_]i32{ 1, 2, 3 },
        &[_]i32{ 4, 5, 6 },
    });
    try B.fromSlice(&[_][]const i32{ &[_]i32{ 7, 8 }, &[_]i32{ 9, 10 }, &[_]i32{ 11, 12 } });

    const expected = try Matrix(i32).init(allocator, 2, 2);
    defer expected.deinit();
    try expected.fromSlice(&[_][]const i32{
        &[_]i32{ 58, 64 },
        &[_]i32{ 139, 154 },
    });

    const result = try A.mult(B);
    defer result.deinit();
    try std.testing.expect(try expected.equals(result));
}

test "LU decomposition" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leak_check = gpa.deinit();
        if (leak_check == .leak) @panic("FAIL");
    }
    const allocator = gpa.allocator();

    const A = try Matrix(f32).init(allocator, 2, 2);
    defer A.deinit();
    try A.fromSlice(&[_][]const f32{
        &[_]f32{ 4, 3 },
        &[_]f32{ 6, 3 },
    });

    const expectedL = try Matrix(f32).init(allocator, 2, 2);
    const expectedU = try Matrix(f32).init(allocator, 2, 2);
    defer {
        expectedL.deinit();
        expectedU.deinit();
    }
    try expectedL.fromSlice(&[_][]const f32{
        &[_]f32{ 4, 0 },
        &[_]f32{ 6, -1.5 },
    });
    try expectedU.fromSlice(&[_][]const f32{
        &[_]f32{ 1, 0.75 },
        &[_]f32{ 0, 1 },
    });

    const resultL, const resultU = try A.LU();
    defer {
        resultL.deinit();
        resultU.deinit();
    }
    try std.testing.expect(try resultL.equals(expectedL));
    try std.testing.expect(try resultU.equals(expectedU));
}

test "Matrix determinant" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leak_check = gpa.deinit();
        if (leak_check == .leak) @panic("FAIL");
    }
    const allocator = gpa.allocator();

    const A = try Matrix(f32).init(allocator, 4, 4);
    defer A.deinit();
    try A.fromSlice(&[_][]const f32{
        &[_]f32{ 5, -7, 2, 2 },
        &[_]f32{ 0, 3, 0, -4 },
        &[_]f32{ -5, -8, 0, 3 },
        &[_]f32{ 0, 5, 0, -6 },
    });

    const result = try A.det();
    try std.testing.expect(result - 20 < 0.0001);
}

test "Matrix inverse" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leak_check = gpa.deinit();
        if (leak_check == .leak) @panic("FAIL");
    }
    const allocator = gpa.allocator();

    const A = try Matrix(f32).init(allocator, 3, 3);
    defer A.deinit();
    try A.fromSlice(&[_][]const f32{
        &[_]f32{ 1, 2, -1 },
        &[_]f32{ 2, 1, 2 },
        &[_]f32{ -1, 2, 1 },
    });

    const inverse = try A.inv();
    defer inverse.deinit();

    const expected = try Matrix(f32).init(allocator, 3, 3);
    defer expected.deinit();
    try expected.fromSlice(&[_][]const f32{
        &[_]f32{ 0.1875, 0.25, -0.3125 },
        &[_]f32{ 0.25, 0.0, 0.25 },
        &[_]f32{ -0.3125, 0.25, 0.1875 },
    });

    try std.testing.expect(try inverse.equals(expected));
}

test "Identity matrix initiated properly" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leak_check = gpa.deinit();
        if (leak_check == .leak) @panic("FAIL");
    }
    const allocator = gpa.allocator();

    const identity = try I(i32, 3, allocator);
    defer identity.deinit();

    const expected = try Matrix(i32).init(allocator, 3, 3);
    defer expected.deinit();
    try expected.fromSlice(&[_][]const i32{
        &[_]i32{ 1, 0, 0 },
        &[_]i32{ 0, 1, 0 },
        &[_]i32{ 0, 0, 1 },
    });

    try std.testing.expect(try identity.equals(expected));
}
