import std;

float smoothstep(float edge0, float edge1, float x){
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

float4 tex(float2 U){
    float2 f = fract(U);
    bool2 edge = bool2(0);
    if (int(U.x)/64 != (int(U.x)+1)/64){
        edge.x = true;
    }
    if (int(U.y)/64 != (int(U.y)+1)/64){
        edge.y = true;
    }  // if you wanna remove the edge cleaning thing, delete from here to bool2 edge
      // above, and replace !edge.* below with 1

    return float4(
        lerp(
            lerp(channel0[int2(U + float2(0,       0))], channel0[int2(U + float2(!edge.x,       0))], f.x),
            lerp(channel0[int2(U + float2(0, !edge.y))], channel0[int2(U + float2(!edge.x, !edge.y))], f.x),
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


    float4 col = float4(0,0,0,1),temp = col;

    float t = time.elapsed*4.;
    int c = int(t);
    int[7] charList = int[](
        (c + 0)%(16*16), 
        (c + 1)%(16*16), 
        (c + 2)%(16*16),
        (c + 3)%(16*16),
        (c + 4)%(16*16),
        (c + 5)%(16*16),
        (c + 6)%(16*16)
    );
    
    for (int i = 7; i --> 0;){
        temp = character(fragCoord/float(width) * 4. - float2((i - (t)%1) * .6,0), charList[i]); 
        col = float4(col.xyz + temp.xyz, min(col.w, temp.w));

    }

    // note: character spacing in demo is 0.6 which is decent for text but not so much for 
    //       larger special characters like the arrows. those end up overlapping

    screen[id.xy] = float4(smoothstep(.5, .5 - 2./width, col.w));
}
