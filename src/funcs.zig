const std = @import("std");

pub fn leviCivita(indices: []const i32) i32 {
    var res: i32 = 1;
    for (0..indices.len) |q| {
        for (q + 1..indices.len) |p| {
            if (indices[p] == indices[q]) {
                return 0;
            }
            const denom: i32 = @intCast(@abs(indices[p] - indices[q]));
            res *= @divExact(indices[p] - indices[q], denom);
        }
    }
    return res;
}

test "Levi-Civita symbol with 2 indices" {
    try std.testing.expect(leviCivita(&.{ 1, 2 }) == 1);

    try std.testing.expect(leviCivita(&.{ 2, 1 }) == -1);

    try std.testing.expect(leviCivita(&.{ 1, 1 }) == 0);
    try std.testing.expect(leviCivita(&.{ 2, 2 }) == 0);
}

test "Levi-Civita symbol with 3 indices" {
    try std.testing.expect(leviCivita(&.{ 1, 2, 3 }) == 1);
    try std.testing.expect(leviCivita(&.{ 2, 3, 1 }) == 1);
    try std.testing.expect(leviCivita(&.{ 3, 1, 2 }) == 1);

    try std.testing.expect(leviCivita(&.{ 3, 2, 1 }) == -1);
    try std.testing.expect(leviCivita(&.{ 1, 3, 2 }) == -1);
    try std.testing.expect(leviCivita(&.{ 2, 1, 3 }) == -1);

    try std.testing.expect(leviCivita(&.{ 1, 1, 2 }) == 0);
}
