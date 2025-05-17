import std;
import shadertoy;

#define MAX_PIXEL_COUNT (1920*1080)
[StorageBuffer(MAX_PIXEL_COUNT)]
RWStructuredBuffer<vec4> ChBufferA;
[StorageBuffer(MAX_PIXEL_COUNT)]
RWStructuredBuffer<vec4> ChBufferB;
[StorageBuffer(MAX_PIXEL_COUNT)]
RWStructuredBuffer<vec4> ChBufferC;
[StorageBuffer(MAX_PIXEL_COUNT)]
RWStructuredBuffer<vec4> ChBufferD;

uint PixelID(ivec2 id) { return id.x + id.y * SCREEN_SIZE.x; }

public vec4 texelFetch(RWStructuredBuffer<vec4> channel, ivec2 coord, int lod) {
    return channel[PixelID(coord)];
}

public vec4 textureLod(RWStructuredBuffer<vec4> channel, vec2 coord, float lod) {
    coord = coord*iResolution - 0.5;
    ivec2 icoord = ivec2(floor(coord));
    vec2 fcoord = fract(coord);
    vec4 v00 = texelFetch(channel, icoord + ivec2(0,0), 0);
    vec4 v01 = texelFetch(channel, icoord + ivec2(0,1), 0);
    vec4 v10 = texelFetch(channel, icoord + ivec2(1,0), 0);
    vec4 v11 = texelFetch(channel, icoord + ivec2(1,1), 0);
    return mix(mix(v00, v01, fcoord.y), mix(v10, v11, fcoord.y), fcoord.x); 
}

public vec4 texture(RWStructuredBuffer<vec4> channel, vec2 coord) {
    return textureLod(channel, coord, 0.0);
}

static const vec3 someConst = vec3(0,2,4); 
static float someParameter = custom.Speed;
static float someGlobal = someParameter*iTime;

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

// Workgroup size 
static const int WG_X = 16;
static const int WG_Y = 16;

#define ComputeFragColor(bfunc, id) \
    if(any(id.xy >= SCREEN_SIZE)) return; \
    vec4 fragColor; \
    bfunc(fragColor, vec2(id.xy) + 0.5);

#define BufferPass(bid, bfunc, id)  \
    ComputeFragColor(bfunc, id) \
    bid[PixelID(id.xy)] = fragColor; 

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void BufferA(uint3 id : SV_DispatchThreadID) { BufferPass(ChBufferA, mainImageA, id) }

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void BufferB(uint3 id : SV_DispatchThreadID) { BufferPass(ChBufferB, mainImageB, id) }

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void BufferC(uint3 id : SV_DispatchThreadID) {  BufferPass(ChBufferC, mainImageC, id) }

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void BufferD(uint3 id : SV_DispatchThreadID) {  BufferPass(ChBufferD, mainImageD, id) }

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void Image(uint3 id : SV_DispatchThreadID) {
    ComputeFragColor(mainImage, id)
    screen[ivec2(id.x, SCREEN_SIZE.y - 1 - id.y)] = pow(fragColor, 2.2);
}