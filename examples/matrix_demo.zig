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

    try A.fromArray(&[_][]const f32{
        &[_]f32{ 4, 3 },
        &[_]f32{ 6, 3 },
    });

    std.debug.print("{}\n", A.det());
}
