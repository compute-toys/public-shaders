import std;


float2 flow_map(float2 p) {
    float t = time.elapsed * .1;
    float2x2 rot = float2x2(cos(t), sin(t), -sin(t), cos(t));
    p = mul(rot, p);
    return float2(p.y, -p.x) + float2(sin(p.x*2), cos(p.y*4));
}

interface Kernel {
    // Given u \in [-1, 1] return a value which weights the position along the LIC
    float eval(float u);
}

struct UniformKernel : Kernel {
    float eval(float u) {
        return 1;
    }
}

struct GaussianKernel : Kernel {
    float eval(float u) {
        return exp(-u*u * 4);
    }
}

struct TriKernel : Kernel {
    float eval(float u) {
        return saturate(1-abs(u));
    }
}

float3 compute_LIC(float2 p_start, float dist, int steps, Kernel k = UniformKernel()) {
    float step_size = 2*dist/steps;
    float4 col;

    // Forwards
    float2 pos = p_start;
    for (float step = 0; step <= dist; step += dist / steps) {
        float w = k.eval(step);
        float2 dir = flow_map(pos) * step_size;
        float4 sample = channel0.SampleGrad(bilinear_repeat, pos, dir, 0);
        col += float4(sample.rgb * w, w);
        pos += dir;
    }

    // Backwards
    pos = p_start;
    for (float step = 0; step <= dist; step += dist / steps) {
        float w = k.eval(step);
        float2 dir = -flow_map(pos) * step_size;
        float4 sample = channel0.SampleGrad(bilinear_repeat, pos, dir, 0);
        col += float4(sample.rgb * w, w);
        pos += dir;
    }
    return col.rgb / col.w;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID) {
    // Viewport resolution (in pixels)
    uint width, height;
    screen.GetDimensions(width, height);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= width || id.y >= height)
        return;

    // Pixel coordinates (centre of pixel, origin at bottom left)
    float2 fragCoord = float2(float(id.x) + 0.5, float(height - id.y) - 0.5);

    // Normalised pixel coordinates (from 0 to 1)
    float2 uv = fragCoord / float2(width, height);

    float2 pos = (uv * 2 - 1) * 3;
    float3 col = compute_LIC(pos, .2, 50, TriKernel());

    // Output to screen (linear colour space)
    screen[id.xy] = float4(pow(col.rgb, 2.2), 1.0);
}
