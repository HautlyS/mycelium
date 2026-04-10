// Matrix multiplication compute shader for Mycelium
// C = A × B where A is M×K, B is K×N, C is M×N
// Each workgroup computes one element of C

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

// Push constants for dimensions would go here but WGSL doesn't support them
// in all implementations, so we use uniform buffer approach

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = gid.x;
    let n = gid.y;

    // We'll read dimensions from the first 3 elements
    // In production: use uniform buffer for dimensions
    let dim_m = u32(a[0]);
    let dim_k = u32(a[1]);
    let dim_n = u32(b[0]);

    if (m >= dim_m || n >= dim_n) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < dim_k; k = k + 1u) {
        sum = sum + a[m * dim_k + k] * b[k * dim_n + n];
    }
    c[m * dim_n + n] = sum;
}
