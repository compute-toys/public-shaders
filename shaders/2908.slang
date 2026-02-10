import std;

float smoothstep(float edge0, float edge1, float x){
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float4 tex(float2 U){
    float2 f = fract(U);
    return float4(
        lerp(
            lerp(channel0[int2(floor(U) + float2(0, 0))], channel0[int2(U + float2(1, 0))], f.x),
            lerp(channel0[int2(floor(U) + float2(0, 1))], channel0[int2(U + float2(1, 1))], f.x),
            f.y
        )
    );
}

float4 character(float2 uv, uint i){
    uint x = i%16u;
    uint y = i/16u;

    if (uv.x < 0. || uv.y < 0. || uv.x > 1. || uv.y > 1.){
        return float4(0,0,0,1);
    }

    return tex(uv*float2(64,-64) + float2(0,64) + float2(x, y)*64.);
}


[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    // Viewport resolution (in pixels)
    uint width, height;
    screen.GetDimensions(width, height);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= width || id.y >= height) return;

    // Pixel coordinates (centre of pixel, origin at bottom left)
    float2 fragCoord = float2(float(id.x) + 0.5, float(height - id.y) - 0.5);
    float2 uv = fragCoord / float2(width, height);


    float4 col = character(fragCoord/float(height)*3. - float2(.3 + (uv.x > .5?2.7:0.), 1), uv.x>.5?72:83);
    float4 temp = character(fragCoord/float(height)*3. - float2(.8 + (uv.x > .5?2.7:0.),1), 68);
    col = float4(col.xyz + temp.xyz, min(col.w, temp.w));
    temp = character(fragCoord/float(height)*3. - float2(1.25 + (uv.x > .5?2.7:0.),1), 82);
    col = float4(col.xyz + temp.xyz, min(col.w, temp.w));

    float4 O = float4(smoothstep(.5, .5 - 2./height, col.w)*.5);


    if (time.elapsed%1.5/1.5 > .5){
        O *= 0.;
    }
    
    O += .5;

    if (uv.x > .5){
        O *= 2.;
    }

    screen[id.xy] = O;
}
