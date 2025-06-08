# ZLA

ZLA (Zig Linear Algebra) is a linear algebra library created for the Zig programming language. The library aims to be efficient while supporting a wide array of functions.

## Example

```zig
const std = @import("std");
const zla = @import("zla");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        const deinit_status = gpa.deinit();

        if (deinit_status == .leak) @panic("LEAK DETECTED");
    }

    const A = try zla.matrix.Matrix(f32).init(allocator, 2, 2);
    defer A.deinit();

    try A.fromSlice(&[_][]const f32{
        &[_]f32 { 4, 3 },
        &[_]f32 { 6, 3 },
    });

    const L, const U = try zla.matrix.LU(A);
    defer {
        L.deinit();
        U.deinit();
    }
    L.print();
    U.print();
}

```

## Supported Features

Below is a list of features currently supported.

* Integer and float matrices
* Transpose
* Equality check
* Matrix addition and subtraction
* Matrix scalar multiplications
* Matrix multiplication
* LU Decomposition
* Determinant
* Matrix Inverse

## Future Features

* QR decomposition
* Singular Value Decomposition
* Pseudo-Inverse
* Spectral Decomposition
* Complex matrices
* Integer and float vectors
* dot and cross products
* Vector norms
* Linear system solver
