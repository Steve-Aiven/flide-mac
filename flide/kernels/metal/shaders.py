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

// ---- Pointwise (Qwen3.5 / OneJev hot path). Arithmetic in float32, store half. ----

// y = RMSNorm(x); optional residual add stores h = (x+res) rounded to half first (matches Triton/eager).
// x,y,h,res: [T, N] row-major half; w: [N] half. One threadgroup per row.
kernel void rms_norm_f16(
    device const half* x   [[buffer(0)]],
    device const half* res [[buffer(1)]],
    device const half* w   [[buffer(2)]],
    device       half* h   [[buffer(3)]],
    device       half* y   [[buffer(4)]],
    constant int& T        [[buffer(5)]],
    constant int& N        [[buffer(6)]],
    constant float& eps    [[buffer(7)]],
    constant int& has_res  [[buffer(8)]],
    constant int& offset   [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tg   [[threads_per_threadgroup]])
{
    int row = tgid;
    if (row >= T) return;
    threadgroup float partial[256];
    device const half* xr = x + row * N;
    float local = 0.0f;
    if (has_res) {
        device const half* rr = res + row * N;
        device half* hr = h + row * N;
        for (uint i = tid; i < (uint)N; i += tg) {
            half hv = half(float(xr[i]) + float(rr[i]));
            hr[i] = hv;
            float v = float(hv);
            local += v * v;
        }
    } else {
        for (uint i = tid; i < (uint)N; i += tg) {
            float v = float(xr[i]);
            local += v * v;
        }
    }
    partial[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(N) + eps);
    device half* yr = y + row * N;
    if (has_res) {
        device half* hr = h + row * N;
        for (uint i = tid; i < (uint)N; i += tg) {
            float ww = float(w[i]);
            if (offset) ww = 1.0f + ww;
            yr[i] = half(float(hr[i]) * inv * ww);
        }
    } else {
        for (uint i = tid; i < (uint)N; i += tg) {
            float ww = float(w[i]);
            if (offset) ww = 1.0f + ww;
            yr[i] = half(float(xr[i]) * inv * ww);
        }
    }
}

// out = silu(gu[:, :n]) * gu[:, n:]; gu [T, 2n], out [T, n].
kernel void swiglu_f16(
    device const half* gu [[buffer(0)]],
    device       half* out [[buffer(1)]],
    constant int& T        [[buffer(2)]],
    constant int& N        [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int col = gid.x;
    int row = gid.y;
    if (row >= T || col >= N) return;
    float a = float(gu[row * (2 * N) + col]);
    float b = float(gu[row * (2 * N) + N + col]);
    float s = a / (1.0f + exp(-a));
    out[row * N + col] = half(s * b);
}

// o [T, H, DV], z [T, H*DV], w [DV] -> out [T, H*DV]: RMSNorm(o)*silu(z).
kernel void gated_rms_f16(
    device const half* o [[buffer(0)]],
    device const half* z [[buffer(1)]],
    device const half* w [[buffer(2)]],
    device       half* out [[buffer(3)]],
    constant int& T        [[buffer(4)]],
    constant int& H        [[buffer(5)]],
    constant int& DV       [[buffer(6)]],
    constant float& eps    [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    int h = gid.x;
    int t = gid.y;
    if (t >= T || h >= H) return;
    device const half* orow = o + (t * H + h) * DV;
    float sumsq = 0.0f;
    for (int i = 0; i < DV; i++) {
        float v = float(orow[i]);
        sumsq += v * v;
    }
    float inv = rsqrt(sumsq / float(DV) + eps);
    int base = t * (H * DV) + h * DV;
    for (int i = 0; i < DV; i++) {
        float y = float(orow[i]) * inv * float(w[i]);
        float zv = float(z[base + i]);
        float sg = zv / (1.0f + exp(-zv));
        out[base + i] = half(y * sg);
    }
}

// Per-head RMSNorm (weight 1+w) then partial RoPE on first R dims.
// x [T, Heads, HD], cos/sin [T, R], w [HD], out [T, Heads, HD].
kernel void norm_rope_f16(
    device const half* x   [[buffer(0)]],
    device const half* cos [[buffer(1)]],
    device const half* sin [[buffer(2)]],
    device const half* w   [[buffer(3)]],
    device       half* out [[buffer(4)]],
    constant int& T        [[buffer(5)]],
    constant int& Heads    [[buffer(6)]],
    constant int& HD       [[buffer(7)]],
    constant int& R        [[buffer(8)]],
    constant float& eps    [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    int h = gid.x;
    int t = gid.y;
    if (t >= T || h >= Heads) return;
    device const half* xr = x + (t * Heads + h) * HD;
    device half* yr = out + (t * Heads + h) * HD;
    float sumsq = 0.0f;
    for (int i = 0; i < HD; i++) {
        float v = float(xr[i]);
        sumsq += v * v;
    }
    float inv = rsqrt(sumsq / float(HD) + eps);
    int half_r = R / 2;
    for (int i = 0; i < HD; i++) {
        float yi = float(xr[i]) * inv * (1.0f + float(w[i]));
        if (i < R) {
            int partner = (i < half_r) ? (i + half_r) : (i - half_r);
            float yp = float(xr[partner]) * inv * (1.0f + float(w[partner]));
            float c = float(cos[t * R + i]);
            float s = float(sin[t * R + i]);
            float sign = (i < half_r) ? -1.0f : 1.0f;
            yi = yi * c + sign * yp * s;
        }
        yr[i] = half(yi);
    }
}

// out[t, h*HD + d] = o[t,h,d] * sigmoid(g[t,h,d]); contiguous last dim assumed after pack.
kernel void gate_f16(
    device const half* o [[buffer(0)]],
    device const half* g [[buffer(1)]],
    device       half* out [[buffer(2)]],
    constant int& T        [[buffer(3)]],
    constant int& H        [[buffer(4)]],
    constant int& HD       [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int h = gid.x;
    int t = gid.y;
    if (t >= T || h >= H) return;
    int oh = (t * H + h) * HD;
    int oo = t * (H * HD) + h * HD;
    for (int i = 0; i < HD; i++) {
        float gv = float(g[oh + i]);
        float sg = 1.0f / (1.0f + exp(-gv));
        out[oo + i] = half(float(o[oh + i]) * sg);
    }
}
"""
