const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const module = b.addModule("zla", .{
        .root_source_file = b.path("src/zla.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{},
    });

    const exe = b.addExecutable(.{
        .name = "example",
        .root_source_file = b.path("examples/matrix_demo.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("zla", module);

    b.installArtifact(exe);
}
