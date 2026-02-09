import std;

// https://www.shadertoy.com/view/4djSRW
float2 hash22(float2 p)
{
	float3 p3 = fract(float3(p.xyx) * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

// https://www.shadertoy.com/view/4djSRW
float3 hash31(float p)
{
    float3 p3 = fract(float3(p) * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xxy+p3.yzz)*p3.zyx); 
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID) {
    uint2 size;
    screen.GetDimensions(size.x, size.y);
    if (any(id.xy >= size)) return;

    float2 uv = (float2(id.xy) - 0.5 * float2(size)) / float(size.y);
    float3 color = float3(0.0);

    float t = time.elapsed;
    float s = floor(t);
    float flash_pulse = exp(-10.0 * fract(t));
    for (int i = 0; i < 10; i++) {
        float2 pos = hash22(float2(s, i)) - 0.5;
        float3 flash_color = 20.0 * normalize(hash31(s));
        color += flash_color * exp(-30.0 * length(uv - pos)) * flash_pulse;
    }

    screen[id.xy] = float4(color, 1.0);
}
