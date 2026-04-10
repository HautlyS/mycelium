// Multi-head attention compute shader for Mycelium
// Computes Q*K^T / sqrt(d_k), applies causal mask, softmax, * V
// Optimized for MiniMax M2.5: 48 heads, head_dim=128

@group(0) @binding(0) var<storage, read> query: array<f32>;    // [seq_len, num_heads, head_dim]
@group(0) @binding(1) var<storage, read> key: array<f32>;      // [seq_len, num_heads, head_dim]
@group(0) @binding(2) var<storage, read> value: array<f32>;    // [seq_len, num_heads, head_dim]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [seq_len, hidden_dim]

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head_idx = gid.x;
    let seq_pos = gid.y;

    let head_dim = 128u;
    let num_heads = 48u;
    let scale = 1.0 / sqrt(f32(head_dim));

    if (head_idx >= num_heads || seq_pos >= 1u) {
        return;
    }

    // Compute attention scores for this head
    var max_score: f32 = -1e30;
    var scores: array<f32, 128>; // head_dim max length

    for (var k_pos: u32 = 0u; k_pos <= seq_pos; k_pos = k_pos + 1u) {
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            let q_val = query[seq_pos * num_heads * head_dim + head_idx * head_dim + d];
            let k_val = key[k_pos * num_heads * head_dim + head_idx * head_dim + d];
            dot = dot + q_val * k_val;
        }
        let score = dot * scale;
        scores[k_pos] = score;
        if (score > max_score) {
            max_score = score;
        }
    }

    // Softmax
    var sum_exp: f32 = 0.0;
    for (var k_pos: u32 = 0u; k_pos <= seq_pos; k_pos = k_pos + 1u) {
        scores[k_pos] = exp(scores[k_pos] - max_score);
        sum_exp = sum_exp + scores[k_pos];
    }
    for (var k_pos: u32 = 0u; k_pos <= seq_pos; k_pos = k_pos + 1u) {
        scores[k_pos] = scores[k_pos] / sum_exp;
    }

    // Weighted sum of values
    for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
        var val: f32 = 0.0;
        for (var k_pos: u32 = 0u; k_pos <= seq_pos; k_pos = k_pos + 1u) {
            let v_val = value[k_pos * num_heads * head_dim + head_idx * head_dim + d];
            val = val + scores[k_pos] * v_val;
        }
        output[seq_pos * num_heads * head_dim + head_idx * head_dim + d] = val;
    }
}
