"""Metal shader source for Flide MPS hot paths.

State layout for GDN matches the eager path: S[h, v, k] (Hv, Dv, Dk).
Update: S = exp(g) * S; sk = S @ k; S += (v - sk) * beta @ k^T; o = S @ q
"""

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Depthwise causal 1D conv over x [B, C, T], weight [C, W] -> out [B, C, T].
kernel void causal_conv1d_fwd_f16(
    device const half* x        [[buffer(0)]],
    device const half* weight   [[buffer(1)]],
    device const half* bias     [[buffer(2)]],
    device       half* out      [[buffer(3)]],
    constant int& B             [[buffer(4)]],
    constant int& C             [[buffer(5)]],
    constant int& T             [[buffer(6)]],
    constant int& W             [[buffer(7)]],
    constant int& has_bias      [[buffer(8)]],
    constant int& apply_silu    [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
    int t = gid.x;
    int c = gid.y;
    int b = gid.z;
    if (b >= B || c >= C || t >= T) return;
    float acc = 0.0f;
    for (int w = 0; w < W; w++) {
        int src = t - (W - 1 - w);
        if (src >= 0)
            acc += float(x[b * C * T + c * T + src]) * float(weight[c * W + w]);
    }
    if (has_bias) acc += float(bias[c]);
    if (apply_silu) {
        float sig = 1.0f / (1.0f + exp(-acc));
        acc = acc * sig;
    }
    out[b * C * T + c * T + t] = half(acc);
}

kernel void causal_conv1d_fwd_f32(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device       float* out     [[buffer(3)]],
    constant int& B             [[buffer(4)]],
    constant int& C             [[buffer(5)]],
    constant int& T             [[buffer(6)]],
    constant int& W             [[buffer(7)]],
    constant int& has_bias      [[buffer(8)]],
    constant int& apply_silu    [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]])
{
    int t = gid.x;
    int c = gid.y;
    int b = gid.z;
    if (b >= B || c >= C || t >= T) return;
    float acc = 0.0f;
    for (int w = 0; w < W; w++) {
        int src = t - (W - 1 - w);
        if (src >= 0)
            acc += x[b * C * T + c * T + src] * weight[c * W + w];
    }
    if (has_bias) acc += bias[c];
    if (apply_silu) {
        float sig = 1.0f / (1.0f + exp(-acc));
        acc = acc * sig;
    }
    out[b * C * T + c * T + t] = acc;
}

// Fused T-step gated delta rule. One threadgroup per head (B=1 typical for Flide segments).
// q,k: [B, H, T, K] half; v: [B, H, T, V] half; g: [B, H, T] float (log-decay);
// beta: [B, H, T] half; state: [B, H, V, K] float32 in-place; out: [B, H, T, V] half.
kernel void gdn_recurrent_fused(
    device const half* q      [[buffer(0)]],
    device const half* k      [[buffer(1)]],
    device const half* v      [[buffer(2)]],
    device const float* g     [[buffer(3)]],
    device const half* beta   [[buffer(4)]],
    device       float* state [[buffer(5)]],
    device       half* out    [[buffer(6)]],
    constant int& B           [[buffer(7)]],
    constant int& H           [[buffer(8)]],
    constant int& T_val       [[buffer(9)]],
    constant int& K           [[buffer(10)]],
    constant int& V           [[buffer(11)]],
    uint2 tgid   [[threadgroup_position_in_grid]],
    uint2 tid2   [[thread_position_in_threadgroup]],
    uint2 tg_dim [[threads_per_threadgroup]])
{
    uint tid = tid2.x;
    uint tg_size = tg_dim.x;
    int h = tgid.x;
    int b = tgid.y;
    if (b >= B || h >= H) return;

    int bh = b * H + h;
    device float* S = state + bh * V * K;

    for (int t = 0; t < T_val; t++) {
        int bht = bh * T_val + t;
        device const half* qt = q + bht * K;
        device const half* kt = k + bht * K;
        device const half* vt = v + bht * V;
        device       half* ot = out + bht * V;
        float decay = exp(g[bht]);
        float beta_t = float(beta[bht]);

        for (uint vi = tid; vi < (uint)V; vi += tg_size) {
            float sk = 0.0f;
            for (int ki = 0; ki < K; ki++) {
                float s = S[vi * K + ki] * decay;
                S[vi * K + ki] = s;
                sk += s * float(kt[ki]);
            }
            float delta = (float(vt[vi]) - sk) * beta_t;
            for (int ki = 0; ki < K; ki++) {
                S[vi * K + ki] += delta * float(kt[ki]);
            }
            float o = 0.0f;
            for (int ki = 0; ki < K; ki++) {
                o += S[vi * K + ki] * float(qt[ki]);
            }
            ot[vi] = half(o);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}
"""
