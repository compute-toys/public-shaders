import std;

// Almost all GLSL/Shadertoy features are mapped to Slang in this module
// https://github.com/compute-toys/compute.toys/blob/master/lib/shaders/shadertoy.slang
import shadertoy;

// There are, however, some differences from Shadertoy:

// 1. All samples from iChannels that are mapped to Buffer A-D
//    must be replaced with ChBufferA-D
//    channel0/channel1 can be used normally, 
//    but note they are called channel0 not iChannel0
#define ChBufferA 0
#define ChBufferB 1
#define ChBufferC 2
#define ChBufferD 3

// 2. Globals must be static
static const vec3 someConst = vec3(0,2,4); 
static float someParameter = custom.Speed;
static float someGlobal = someParameter*iTime;

// 3. Compute shaders don't support screen derivatives, so dFdx/dFdy/fwidth won't work
//    They must be emulated, either analytically, or doing multiple passes
//    This also means that texture(ch, p) always samples the 0th MIP level
//    Technically can be fully emulated using subgroup operations and clever mapping,
//    but not supported here at the moment.

// 4. There are no sampler2D or other GLSL texture types
//    Those must be either replaced by an int (for the buffer index)
//    Or replaced with Texture2D<float4> which is the type of the input channel0/1

void mainImageA( out vec4 fragColor, in vec2 fragCoord ) {
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    // Time varying pixel color
    vec3 col = 0.5 + 0.5*cos(someGlobal+uv.xyx+someConst);

    // Output to screen
    fragColor = vec4(col,1.0);
}

void mainImageB( out vec4 fragColor, in vec2 fragCoord ) {
    fragColor = texture(ChBufferA, fragCoord / iResolution.xy);
}

void mainImageC( out vec4 fragColor, in vec2 fragCoord ) {
    fragColor = texelFetch(ChBufferB, ivec2(fragCoord), 0);
}

void mainImageD( out vec4 fragColor, in vec2 fragCoord ) {
    fragColor = textureLod(ChBufferC, fragCoord / iResolution.xy, 0);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    fragColor = texture(ChBufferD, fragCoord / iResolution.xy);
}

////////////////////////////////////////////////////////////
//Compute dispatches emulating fullscreen fragment shaders//
////////////////////////////////////////////////////////////

// 5. Unlike fragment shaders here you actually can control how the GPU spawns
//    the "pixel" threads.  Depending on the algorithm it might be preferrable 
//    to change the size of the workgroups

// Workgroup size 
static const int WG_X = 16;
static const int WG_Y = 16;

#define ComputeFragColor(bfunc, id) \
    if(any(id.xy >= SCREEN_SIZE)) return; \
    vec4 fragColor; \
    bfunc(fragColor, vec2(id.xy) + 0.5);

#define BufferPass(bid, bfunc, id)  \
    ComputeFragColor(bfunc, id) \
    pass_out[int3(id.xy, bid)] = fragColor; 

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void BufferA(uint3 id : SV_DispatchThreadID) { BufferPass(0, mainImageA, id) }

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void BufferB(uint3 id : SV_DispatchThreadID) { BufferPass(1, mainImageB, id) }

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void BufferC(uint3 id : SV_DispatchThreadID) {  BufferPass(2, mainImageC, id) }

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void BufferD(uint3 id : SV_DispatchThreadID) {  BufferPass(3, mainImageD, id) }

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void Image(uint3 id : SV_DispatchThreadID) {
    ComputeFragColor(mainImage, id)
    screen[ivec2(id.x, SCREEN_SIZE.y - 1 - id.y)] = pow(fragColor, 2.2);
}