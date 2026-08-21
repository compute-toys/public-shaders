import std;

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    // Viewport resolution (in pixels)
    uint width, height;
    screen.GetDimensions(width, height);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= width || id.y >= height)
        return;


    float2 U = float2(float(id.x) + 0.5, float(height - id.y) - 0.5);
    float2 r = float2(width, height);

    float2 uv = U / float2(width, height);
    float2 cuv = (2.*U-r)/r.y;
    float3 col = float3(.2);
    col.xy = cuv*.1;

    screen[id.xy] = float4(pow(col, 2.2), 1.0); // clear screen

    if (id.x > 24 || id.y > 12) return;

    float2 a = float2(cos(float(id.x)+0.+time.elapsed), sin(float(id.x)+0.+time.elapsed));
    float2 b = float2(cos(float(id.x)+2.+time.elapsed), sin(float(id.x)+2.+time.elapsed));
    float2 c = float2(cos(float(id.x)+4.+time.elapsed), sin(float(id.x)+4.+time.elapsed));

    a /= 20.;
    b /= 20.;
    c /= 20.;

    a += float2(id.xy*float2(1,-1))/8. - float2(1.5, -.8);
    b += float2(id.xy*float2(1,-1))/8. - float2(1.5, -.8);
    c += float2(id.xy*float2(1,-1))/8. - float2(1.5, -.8);


    float4 aabb = float4(
        min(a.x, min(b.x, c.x)),
        min(a.y, min(b.y, c.y)),
        max(a.x, max(b.x, c.x)),
        max(a.y, max(b.y, c.y))
    );
    
    uint4 ibb = uint4(
        uint2(((aabb.xy * r.y)+r)/2.),
        uint2(((aabb.zw * r.y)+r)/2.)
    );

    uint2 sbb = uint2(
        ibb.z-ibb.x,
        ibb.w-ibb.y
    );

    for (uint x = ibb.x; x < ibb.z; x++){
        for (uint y = ibb.y; y < ibb.w; y++){


            float2 U = float2(float(x) + 0.5, float(y) - 0.5);
            float2 cuv = (2.*U-r)/r.y;

            float4 bary = float4(
                dot(normalize(b-a).yx * float2(-1,1), cuv-a),
                dot(normalize(c-b).yx * float2(-1,1), cuv-b),
                dot(normalize(a-c).yx * float2(-1,1), cuv-c),
                0.
            ) / dot(normalize(b-a).yx * float2(-1,1), c-a);

            if (bary.x < 0. || bary.y < 0. || bary.z < 0.){
                bary.w = 0.;
            } else {
                bary.w = 1.;
            }

            if (bary.w > 0.) 
                screen[uint2(x, height-y)] = float4(bary);
        }
    }
}
