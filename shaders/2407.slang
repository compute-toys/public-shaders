import std;

static const float PI = 3.14159265358979323846f;

static const uint CANVAS_W = 1500;
static const uint CANVAS_H = 1500;
static const uint ELEMENTS = CANVAS_W * CANVAS_H;

static const uint IMG_WG_SIZE = 16;
static const uint WG_COUNT_IMAGE_X = (CANVAS_W + IMG_WG_SIZE - 1) / IMG_WG_SIZE;
static const uint WG_COUNT_IMAGE_Y = (CANVAS_H + IMG_WG_SIZE - 1) / IMG_WG_SIZE;

// Curve params
static const uint CURVE_WG_SIZE   = 128;
static const uint CURVE_WG_COUNT  = 256;
static const uint TOTAL_CURVES    = CURVE_WG_SIZE * CURVE_WG_COUNT;
static const uint SAMPLES_PER_CURVE = 1024;
static const int  STAMP_RADIUS    = 1;
static const int  FLOAT_TO_INT_SCALE = 1024;
static const float PHI_RANGE = 2.0;

// one atomic flag per pixel (0→1 once)
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<Atomic<uint>> IC_written;

// seed payload (first writer wins)
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float2> IC_seeds;

// JFA ping-pong + nearest-seed integer coords
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float2> IC_JFA_0;
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float2> IC_JFA_1;
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<uint2>  IC_SEEDCOORD_0;
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<uint2>  IC_SEEDCOORD_1;

static const uint  INVALID_U = 0xffffffffu;
static const float2 INVALID_F2 = float2(-1e30, -1e30);

uint linearIndex(uint2 p) { return p.x + p.y * CANVAS_W; }

void scatter_ic(uint2 pix, float2 ic, uint2 seedCoord)
{
    uint idx = linearIndex(pix);
    // first-writer-wins using atomic compare-exchange on uint
    uint old = IC_written[idx].compareExchange(0u, 1u); // succeeds if old==0
    if (old == 0u) {
        IC_seeds[idx]       = ic;
        IC_SEEDCOORD_0[idx] = seedCoord;
        IC_JFA_0[idx]       = ic;           // JFA starts seeded
    }
}

[WorkgroupCount(WG_COUNT_IMAGE_X, WG_COUNT_IMAGE_Y, 1)]
[shader("compute")]
[numthreads(IMG_WG_SIZE, IMG_WG_SIZE, 1)]
void clear_ic(uint3 id : SV_DispatchThreadID) {
    if (id.x >= CANVAS_W || id.y >= CANVAS_H) return;
    uint idx = linearIndex(id.xy);
    IC_written[idx].store(0u);
    IC_seeds[idx]       = INVALID_F2;
    IC_JFA_0[idx]       = INVALID_F2;
    IC_JFA_1[idx]       = INVALID_F2;
    IC_SEEDCOORD_0[idx] = uint2(INVALID_U, INVALID_U);
    IC_SEEDCOORD_1[idx] = uint2(INVALID_U, INVALID_U);
}


static const float EPS = 1e-6;

float2 get_uv0(float u0, float angle) { return u0 * float2(1.0, -tan(angle)); }
float get_angle0(float2 uv) { return atan2(-uv.y, uv.x); }

float3 hsv2rgb(float3 hsv) {
    float4 K = float4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    float3 p = abs(frac(hsv.xxx + K.yzw) * 6.0 - K.www);
    return hsv.z * lerp(K.xxx, saturate(p - K.xxx), hsv.y);
}

float3 jet(float t)
{
    t = saturate(t);
    float x = 4.0 * t;
    float r = saturate(1.5 - abs(x - 3.0));
    float g = saturate(1.5 - abs(x - 2.0));
    float b = saturate(1.5 - abs(x - 1.0));
    return float3(r, g, b); // dark blue -> cyan -> yellow -> dark red
}

float3 jet(float t, float min, float max)
{
    t = clamp((t - min) / (max - min), 0.0, 1.0);
    return jet(t);
}

float2 duv(float2 uv) { return float2(uv.y, 1.5*uv.x*uv.x - uv.x); }

void rk4_step_uv(inout float2 uv, float h)
{
    float2 k1 = duv(uv);
    float2 k2 = duv(uv + 0.5*h*k1);
    float2 k3 = duv(uv + 0.5*h*k2);
    float2 k4 = duv(uv + h*k3);
    uv += (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4);
}

void draw_one_curve(float2 uv0, float step)
{
    float phi = 0.0;
    float2 uv = uv0;

    for (uint i=0u; i<SAMPLES_PER_CURVE; ++i) {
        float h = max(pow(uv.x, 0.2), 0.01) * step / (1.0 + abs(uv.y));
        float x = (phi) / (PHI_RANGE*PI);
        float y = uv.x / uv0.x;
        y = y / (y + 1.0);
        float2 pos = float2(x * float(CANVAS_W - 1), y * float(CANVAS_H - 1));
        //float4 color = float4(jet(sin(100.0*((uv0.y - uv.y)))*0.5 + 0.5), 1.0);
        float angle0 = get_angle0(uv0);
        float angle = get_angle0(uv);
        float4 color = float4(jet(3.0*(uv0.y - uv.y)*0.5 + 0.5), 1.0);
        bool outside = uv.x >= 1.0 || uv.x <= 0.0;
        //float4 color = float4(jet(sin(100.0*((angle0 - angle)))*0.5 + 0.5), 1.0);
        if (pos.x >= 0 && pos.y >= 0 && pos.x < int(CANVAS_W) && pos.y < int(CANVAS_H))
        {
            scatter_ic(uint2(uint(pos.x), uint(pos.y)), float2(uv0.x, angle0 /*angle - angle0 - phi*/), uint2(uint(pos.x), uint(pos.y)));
        } 

        rk4_step_uv(uv, h);
        phi += h;
        if (uv.x <= 0.0) break;
    }
}

void draw_bidir(float2 uv0)
{
    draw_one_curve(uv0, 0.01);
    //draw_one_curve(uv0, -0.005);
}

float2 phi2dir(float dir) {
    return -float2(sin(2*PI*dir), cos(2*PI*dir))*0.5 + 0.5;
}

[WorkgroupCount(CURVE_WG_COUNT, 1, 1)]
[shader("compute")]
[numthreads(CURVE_WG_SIZE, 1, 1)]
void draw_curves(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint curveId = gid.x * CURVE_WG_SIZE + gtid.x;
    if (curveId < TOTAL_CURVES) 
    {   
        float u0 = float(mouse.pos.y)/float(SCREEN_SIZE.y);
        float phi = float(curveId)/float(TOTAL_CURVES);
        float angle = PI*(phi - 0.5);
        float2 uv0 = get_uv0(u0, angle);
        draw_bidir(uv0);
    }
}


[WorkgroupCount(WG_COUNT_IMAGE_X, WG_COUNT_IMAGE_Y, 1)]
[shader("compute")]
[numthreads(IMG_WG_SIZE, IMG_WG_SIZE, 1)]
void jfa_init(uint3 id : SV_DispatchThreadID) {
    if (id.x >= CANVAS_W || id.y >= CANVAS_H) return;
    uint idx = linearIndex(id.xy);
    if (IC_written[idx].load() == 0u) {
        IC_JFA_0[idx]       = INVALID_F2;
        IC_SEEDCOORD_0[idx] = uint2(INVALID_U, INVALID_U);
    }
    IC_JFA_1[idx]       = IC_JFA_0[idx];
    IC_SEEDCOORD_1[idx] = IC_SEEDCOORD_0[idx];
}

static const uint JFA_ITERS = 4;         // keep even → final in IC_JFA_0

int2 clampPix(int2 p){
    p.x = clamp(p.x, 0, int(CANVAS_W-1));
    p.y = clamp(p.y, 0, int(CANVAS_H-1));
    return p;
}
uint d2(uint2 a, uint2 b){
    int dx = int(a.x) - int(b.x);
    int dy = int(a.y) - int(b.y);
    return uint(dx*dx + dy*dy);
}

[DispatchCount(JFA_ITERS)]
[WorkgroupCount(WG_COUNT_IMAGE_X, WG_COUNT_IMAGE_Y, 1)]
[shader("compute")]
[numthreads(IMG_WG_SIZE, IMG_WG_SIZE, 1)]
void jfa(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= CANVAS_W || tid.y >= CANVAS_H) return;
    uint idx  = tid.x + tid.y * CANVAS_W;
    bool ping = ((dispatch.id & 1u) == 0u);

    float2 srcVal  = ping ? IC_JFA_0[idx]       : IC_JFA_1[idx];
    uint2  srcSeed = ping ? IC_SEEDCOORD_0[idx] : IC_SEEDCOORD_1[idx];

    float2 bestVal  = srcVal;
    uint2  bestSeed = srcSeed;
    uint   bestD2   = (srcSeed.x==0xffffffffu) ? 0xffffffffu : d2(uint2(tid.xy), srcSeed);

    // 4-cardinal taps at radii {8,4,2,1} in one pass
    [unroll] for (int si=0; si<8; ++si){
        int s = 1 << si;
        int2 offsets[4] = { int2( s,0), int2(-s,0), int2(0, s), int2(0,-s) };
        [unroll] for (int k=0; k<4; ++k){
            int2 p = clampPix(int2(tid.xy) + offsets[k]);
            uint  nidx  = uint(p.x) + uint(p.y) * CANVAS_W;
            uint2 nseed = ping ? IC_SEEDCOORD_0[nidx] : IC_SEEDCOORD_1[nidx];
            if (nseed.x == 0xffffffffu) continue;
            uint  dd    = d2(uint2(tid.xy), nseed);
            if (dd < bestD2){
                bestD2   = dd;
                bestSeed = nseed;
                bestVal  = ping ? IC_JFA_0[nidx] : IC_JFA_1[nidx];
            }
        }
    }

    if (ping){
        IC_JFA_1[idx]       = bestVal;
        IC_SEEDCOORD_1[idx] = bestSeed;
    } else {
        IC_JFA_0[idx]       = bestVal;
        IC_SEEDCOORD_0[idx] = bestSeed;
    }
}


static const float SLOPE_VIS_K = 0.35; // arctan compression for display

float2 load_ic(uint2 p) {
    p.x = min(p.x, CANVAS_W - 1);
    p.y = min(p.y, CANVAS_H - 1);
    return IC_JFA_0[linearIndex(p)];
}

float3 slope_to_color(float2 uv) {
    float angle = get_angle0(uv);
    return jet(sin(20.0*uv.y), -1.0, 1.0);
}

[shader("compute")]
[numthreads(IMG_WG_SIZE, IMG_WG_SIZE, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    uint sw, sh;
    screen.GetDimensions(sw, sh);

    float2 screenSize = float2(sw, sh);
    float2 canvasSize = float2(CANVAS_W, CANVAS_H);
    float  scale = min(screenSize.x / canvasSize.x, screenSize.y / canvasSize.y);
    float2 scaledSize = canvasSize * scale;
    float2 offset = 0.5 * (screenSize - scaledSize);

    float2 p = float2(id.xy) + 0.5;
    if (any(p < offset) || any(p >= offset + scaledSize)) { screen[id.xy] = float4(0,0,0,1); return; }

    // inverse-map with bilinear on RGBA
    float2 q = (p - offset) / scale - 0.5;
    float2 b = floor(q), f = q - b;

    int2 ib = clamp(int2(b), int2(0,0), int2(int(CANVAS_W - 1), int(CANVAS_H - 1)));
    uint2 i00 = uint2(ib);
    uint2 i10 = uint2(min(ib.x + 1, int(CANVAS_W - 1)), ib.y);
    uint2 i01 = uint2(ib.x, min(ib.y + 1, int(CANVAS_H - 1)));
    uint2 i11 = uint2(min(ib.x + 1, int(CANVAS_W - 1)), min(ib.y + 1, int(CANVAS_H - 1)));

    float2 ic00 = load_ic(i00);
    float2 ic10 = load_ic(i10);
    float2 ic01 = load_ic(i01);
    float2 ic11 = load_ic(i11);
    float2 ic0  = lerp(ic00, ic10, f.x);
    float2 ic1  = lerp(ic01, ic11, f.x);
    float2 ic   = lerp(ic0,  ic1,  f.y);

    // visualize du/dphi from the nearest IC
    float3 rgb = slope_to_color(ic);
    screen[id.xy] = float4(rgb, 1.0);
}