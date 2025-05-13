import std;
import shadertoy;

#define iChannel0 0
#define iChannel1 1
#define iChannel2 2
#define iChannel3 3

////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////

void mainImageA( out vec4 fragColor, in vec2 fragCoord ) {
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // Time varying pixel color
    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));

    // Output to screen
    fragColor = vec4(col,1.0);
}

void mainImageB( out vec4 fragColor, in vec2 fragCoord ) {
    fragColor = texture(iChannel0, fragCoord / iResolution.xy);
}

void mainImageC( out vec4 fragColor, in vec2 fragCoord ) {
    fragColor = texelFetch(iChannel1, ivec2(fragCoord), 0);
}

void mainImageD( out vec4 fragColor, in vec2 fragCoord ) {
    fragColor = textureLod(iChannel2, fragCoord / iResolution.xy, 0);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    fragColor = texture(iChannel3, fragCoord / iResolution.xy);
}

////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////

[shader("compute")]
[numthreads(16, 16, 1)]
void BufferA(uint3 id : SV_DispatchThreadID) {
    vec2 fragCoord = vec2(id.xy) + 0.5;
    vec4 fragColor;
    mainImageA(fragColor, fragCoord);
    pass_out[int3(id.xy, 0)] = fragColor;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void BufferB(uint3 id : SV_DispatchThreadID) {
    vec2 fragCoord = vec2(id.xy) + 0.5;
    vec4 fragColor;
    mainImageB(fragColor, fragCoord);
    pass_out[int3(id.xy, 1)] = fragColor;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void BufferC(uint3 id : SV_DispatchThreadID) {
    vec2 fragCoord = vec2(id.xy) + 0.5;
    vec4 fragColor;
    mainImageC(fragColor, fragCoord);
    pass_out[int3(id.xy, 2)] = fragColor;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void BufferD(uint3 id : SV_DispatchThreadID) {
    vec2 fragCoord = vec2(id.xy) + 0.5;
    vec4 fragColor;
    mainImageD(fragColor, fragCoord);
    pass_out[int3(id.xy, 3)] = fragColor;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void Image(uint3 id : SV_DispatchThreadID) {
    vec2 fragCoord = vec2(id.xy) + 0.5;
    vec4 fragColor;
    mainImage(fragColor, fragCoord);
    screen[ivec2(id.x, SCREEN_SIZE.y - 1 - id.y)] = pow(fragColor, 2.2);
}