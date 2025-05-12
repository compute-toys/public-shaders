////////////////////////////////////////////////////
///// SHADERTOY TO COMPUTE.TOYS TEMPLATE v0.2 //////
////////////////////////////////////////////////////
import std;

// This template assumes that channels are in the following order both for inputs and outputs of the buffers:
#define iChannel0 0
#define iChannel1 1
#define iChannel2 2
#define iChannel3 3

// The template is not perfect, as you can't the channels used, 
// but it should be a good starting point for porting shaders from shadertoy to compute.toys

// types
typealias vec2  = float2;   typealias vec3  = float3;   typealias vec4  = float4;
typealias ivec2 = int2;     typealias ivec3 = int3;     typealias ivec4 = int4;
typealias uvec2 = uint2;    typealias uvec3 = uint3;    typealias uvec4 = uint4;
typealias bvec2 = bool2;    typealias bvec3 = bool3;    typealias bvec4 = bool4;
typealias mat2  = float2x2; typealias mat3  = float3x3; typealias mat4  = float4x4;

mat2 operator*(mat2 a, mat2 b) { return mul(a,b); }
vec2 operator*(mat2 a, vec2 b) { return mul(a,b); }
mat3 operator*(mat3 a, mat3 b) { return mul(a,b); }
vec3 operator*(mat3 a, vec3 b) { return mul(a,b); }
mat4 operator*(mat4 a, mat4 b) { return mul(a,b); }
vec4 operator*(mat4 a, vec4 b) { return mul(a,b); }

#define mix lerp
#define fract frac
#define inversesqrt rsqrt
#define floatBitsToUint  asuint
#define floatBitsToInt   asint
#define uintBitsToFloat  asfloat
#define intBitsToFloat   asfloat
#define mod(x, y) ((x) - floor((y)/(x))*(y))  

#define iResolution vec2(SCREEN_SIZE)
#define iTime (time.elapsed)
#define iTimeDelta (time.delta)
#define iFrameRate (time.frame)
#define iFrame (time.frame)
#define iMousePos (vec2(mouse.pos.x, SCREEN_SIZE.y - 1 - mouse.pos.y))
#define iMouse vec4(iMousePos, (mouse.click == 1)?iMousePos:0.0)

vec4 texelFetch(int channel, ivec2 coord, int lod) {
    return pass_in[ivec3(coord, channel)];
}

vec4 textureLod(int channel, vec2 coord, float lod) {
    coord = coord*iResolution - 0.5;
    ivec2 icoord = ivec2(floor(coord));
    vec2 fcoord = fract(coord);
    vec4 v00 = texelFetch(channel, icoord + ivec2(0,0), 0);
    vec4 v01 = texelFetch(channel, icoord + ivec2(0,1), 0);
    vec4 v10 = texelFetch(channel, icoord + ivec2(1,0), 0);
    vec4 v11 = texelFetch(channel, icoord + ivec2(1,1), 0);
    return mix(mix(v00, v01, fcoord.y), mix(v10, v11, fcoord.y), fcoord.x); 
}

vec4 texture(int channel, vec2 coord) {
    return textureLod(channel, coord, 0.0);
}

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