import std;

//#define FORWARD_MODE

static const float PI = 3.14159265359;
static const float SCALE = 200.0;

[Differentiable]
float plasma(float2 p, no_diff float t) {
    let sico = float2(sin(t), cos(t));
    let o = (1.0 + float2(-sico.x, sico.y)) * float2(320, 160);
    return 0.5 - 0.5 * cos(PI / 2.0 * (sin(0.02 * dot(sico, p) + t) + cos(0.03 * length(p - o))) + t);
}

float sdLine(float2 p, float2 a, float2 b) {
    let pa = p - a;
    let ba = b - a;
    return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}

float2 calcUv(uint2 coord, uint2 size) {
    var uv = float2(coord) / float(size.y);
    uv.y = 1.0 - uv.y;
    return uv;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint2 size;
    screen.GetDimensions(size.x, size.y);
    if (any(id.xy >= size)) return;

    let t = time.elapsed * 0.25;
    let uv = calcUv(id.xy, size);
    var color = float3(0.0);

    let cellCenter = (floor(uv * 16.0) + 0.5) / 16.0;
    float2 grad;

    #ifdef FORWARD_MODE
    color += plasma(uv * SCALE, t) * float3(1.0, 0.8, 0.0);
    let diffPos = cellCenter * SCALE;
    grad.x = fwd_diff(plasma)(diffPair(diffPos, float2(1.0, 0.0)), t).d;
    grad.y = fwd_diff(plasma)(diffPair(diffPos, float2(0.0, 1.0)), t).d;

    #else // BACKWARD_MODE
    color += plasma(uv * SCALE, t) * float3(0.0, 0.5, 0.8);
    var diffPos = diffPair(cellCenter * SCALE);
    bwd_diff(plasma)(diffPos, t, 1.0);
    grad = diffPos.d;

    #endif

    let dist = sdLine(uv, cellCenter, cellCenter + grad) - 0.005;
    color = lerp(float3(1.0, 0.0, 0.0), color, smoothstep(0.0, 1.0 / float(size.y), dist));

    screen[id.xy] = float4(pow(color, float3(2.2)), 1.0);
}