"""Metal shader source for Flide MPS hot paths.

GDN state layout matches eager: S[h, v, k] (Hv, Dv, Dk).
"""

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Contiguous batch causal conv: x [B, C, T], weight [C, W] -> out [B, C, T].
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
        acc *= sig;
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
        acc *= sig;
    }
    out[b * C * T + c * T + t] = acc;
}

// Packed-stream varlen causal conv. x/out are [T, C] row-major.
// cu [N+1]; init_state/out_state [N, C, pad] with pad = W-1 (pad may be 0).
// hist[0]=oldest previous input, hist[pad-1]=most recent previous input.
// One thread per (channel, segment). W must be <= 16.
kernel void causal_conv1d_varlen_f16(
    device const half* x          [[buffer(0)]],
    device const half* weight     [[buffer(1)]],
    device const half* bias       [[buffer(2)]],
    device const int*  cu         [[buffer(3)]],
    device const half* init_state [[buffer(4)]],
    device       half* out        [[buffer(5)]],
    device       half* out_state  [[buffer(6)]],
    constant int& N               [[buffer(7)]],
    constant int& C               [[buffer(8)]],
    constant int& W               [[buffer(9)]],
    constant int& has_bias        [[buffer(10)]],
    constant int& apply_silu      [[buffer(11)]],
    constant int& has_init        [[buffer(12)]],
    constant int& write_state     [[buffer(13)]],
    uint2 gid [[thread_position_in_grid]])
{
    int c = gid.x;
    int seg = gid.y;
    if (seg >= N || c >= C) return;
    int lo = cu[seg];
    int hi = cu[seg + 1];
    int L = hi - lo;
    int pad = W - 1;

    float hist[16];
    for (int i = 0; i < 16; i++) hist[i] = 0.0f;
    if (has_init && pad > 0) {
        device const half* is = init_state + ((seg * C + c) * pad);
        for (int i = 0; i < pad; i++) hist[i] = float(is[i]);
    }

    for (int t = 0; t < L; t++) {
        float x_t = float(x[(lo + t) * C + c]);
        float acc = 0.0f;
        for (int w = 0; w < W; w++) {
            int lookback = W - 1 - w; // w=0 oldest, w=W-1 = x_t
            float xv;
            if (lookback == 0) {
                xv = x_t;
            } else {
                // lookback=1 -> hist[pad-1], lookback=pad -> hist[0]
                xv = hist[pad - lookback];
            }
            acc += xv * float(weight[c * W + w]);
        }
        if (has_bias) acc += float(bias[c]);
        if (apply_silu) {
            float sig = 1.0f / (1.0f + exp(-acc));
            acc *= sig;
        }
        out[(lo + t) * C + c] = half(acc);
        if (pad > 0) {
            for (int i = 0; i < pad - 1; i++) hist[i] = hist[i + 1];
            hist[pad - 1] = x_t;
        }
    }
    if (write_state && pad > 0) {
        device half* os = out_state + ((seg * C + c) * pad);
        for (int i = 0; i < pad; i++) os[i] = half(hist[i]);
    }
}

// Single-segment fused GDN (kept for tiny calls).
// q,k: [B, H, T, K] half; v: [B, H, T, V]; g: [B, H, T] float; beta: [B, H, T] half;
// state: [B, H, V, K] float32; out: [B, H, T, V] half.
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
            for (int ki = 0; ki < K; ki++) S[vi * K + ki] += delta * float(kt[ki]);
            float o = 0.0f;
            for (int ki = 0; ki < K; ki++) o += S[vi * K + ki] * float(qt[ki]);
            ot[vi] = half(o);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// Packed-stream varlen GDN. Layouts:
// q,k: [T, H, K] half; v: [T, H, V] half; g: [T, H] float; beta: [T, H] half;
// state: [N, H, V, K] float; out: [T, H, V] half; cu: [N+1] int.
// One threadgroup per (head, segment).
kernel void gdn_recurrent_varlen(
    device const half* q      [[buffer(0)]],
    device const half* k      [[buffer(1)]],
    device const half* v      [[buffer(2)]],
    device const float* g     [[buffer(3)]],
    device const half* beta   [[buffer(4)]],
    device const int*  cu     [[buffer(5)]],
    device       float* state [[buffer(6)]],
    device       half* out    [[buffer(7)]],
    constant int& N           [[buffer(8)]],
    constant int& H           [[buffer(9)]],
    constant int& K           [[buffer(10)]],
    constant int& V           [[buffer(11)]],
    uint2 tgid   [[threadgroup_position_in_grid]],
    uint2 tid2   [[thread_position_in_threadgroup]],
    uint2 tg_dim [[threads_per_threadgroup]])
{
    uint tid = tid2.x;
    uint tg_size = tg_dim.x;
    int h = tgid.x;
    int seg = tgid.y;
    if (seg >= N || h >= H) return;
    int lo = cu[seg];
    int hi = cu[seg + 1];
    device float* S = state + ((seg * H + h) * V * K);
    for (int t = lo; t < hi; t++) {
        int th = t * H + h;
        device const half* qt = q + th * K;
        device const half* kt = k + th * K;
        device const half* vt = v + th * V;
        device       half* ot = out + th * V;
        float decay = exp(g[th]);
        float beta_t = float(beta[th]);
        for (uint vi = tid; vi < (uint)V; vi += tg_size) {
            float sk = 0.0f;
            for (int ki = 0; ki < K; ki++) {
                float s = S[vi * K + ki] * decay;
                S[vi * K + ki] = s;
                sk += s * float(kt[ki]);
            }
            float delta = (float(vt[vi]) - sk) * beta_t;
            for (int ki = 0; ki < K; ki++) S[vi * K + ki] += delta * float(kt[ki]);
            float o = 0.0f;
            for (int ki = 0; ki < K; ki++) o += S[vi * K + ki] * float(qt[ki]);
            ot[vi] = half(o);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}
"""
