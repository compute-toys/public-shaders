import std;

static const float PI = 3.14159265358979323846f;

static const uint CANVAS_W = 2000;
static const uint CANVAS_H = 1000;
static const uint ELEMENTS = CANVAS_W * CANVAS_H;

static const uint IMG_WG_SIZE = 16;
static const uint WG_COUNT_IMAGE_X = (CANVAS_W + IMG_WG_SIZE - 1) / IMG_WG_SIZE;
static const uint WG_COUNT_IMAGE_Y = (CANVAS_H + IMG_WG_SIZE - 1) / IMG_WG_SIZE;

// Curve params
static const uint CURVE_WG_SIZE   = 64;
static const uint CURVE_WG_COUNT  = 1;
static const uint TOTAL_CURVES    = CURVE_WG_SIZE * CURVE_WG_COUNT;
static const uint SAMPLES_PER_CURVE = 1024;
static const int  STAMP_RADIUS    = 1;
static const int  FLOAT_TO_INT_SCALE = 1024;
static const float PHI_RANGE = 1.5;

[StorageBuffer(ELEMENTS * 4)]
RWStructuredBuffer<Atomic<uint>> canvas;


static const float EPS = 1e-6;

// --- helpers ---
uint linearIndex(uint2 p) { return p.x + p.y * CANVAS_W; }

void stamp_disk(float2 p, int r, float4 color) {
    float rad = float(r), r2 = rad*rad;
    int2 lo = int2(max(0.0, floor(p.x - rad)), max(0.0, floor(p.y - rad)));
    int2 hi = int2(min(float(CANVAS_W - 1), ceil(p.x + rad)),
                   min(float(CANVAS_H - 1), ceil(p.y + rad)));

    for (int y = lo.y; y <= hi.y; ++y) {
        float dy = (float(y) + 0.5) - p.y;
        for (int x = lo.x; x <= hi.x; ++x) {
            float dx = (float(x) + 0.5) - p.x;
            float d2 = dx*dx + dy*dy;
            if (d2 <= r2) {
                float w = exp(-2.0 * d2 / r2);
                float4 c = color * w;
                uint base = (linearIndex(uint2(uint(x), uint(y))) << 2);
                if (c.x > EPS) canvas[base + 0].add((uint)(FLOAT_TO_INT_SCALE * c.x));
                if (c.y > EPS) canvas[base + 1].add((uint)(FLOAT_TO_INT_SCALE * c.y));
                if (c.z > EPS) canvas[base + 2].add((uint)(FLOAT_TO_INT_SCALE * c.z));
                if (c.w > EPS) canvas[base + 3].add((uint)(FLOAT_TO_INT_SCALE * c.w));
            }
        }
    }
}

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

float2 duv(float2 uv, float rs) { return float2(uv.y, 1.5*rs*uv.x*uv.x - uv.x); }

void rk4_step_uv(inout float2 uv, float h, float rs)
{
    float2 k1 = duv(uv, rs);
    float2 k2 = duv(uv + 0.5*h*k1, rs);
    float2 k3 = duv(uv + 0.5*h*k2, rs);
    float2 k4 = duv(uv + h*k3, rs);
    uv += (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4);
}

// u0 = initial u, du0 = initial du/dphi, rs = r_s
void draw_one_curve(float2 uv0, float step)
{
    float rs = 1.0;
    float phi = 0.0;
    float2 uv = uv0;

    for (uint i=0u; i<SAMPLES_PER_CURVE; ++i) {
        float h = step / (1.0 + abs(uv.y));
        float x = (phi + PHI_RANGE*PI) / (2.0*PHI_RANGE*PI);
        float y = uv.x;// / uv0.x;
        //y = y / (y + 1.0);
        float2 pos = float2(x * float(CANVAS_W - 1), y * float(CANVAS_H - 1));
        //float4 color = float4(jet(sin(100.0*((uv0.y - uv.y)))*0.5 + 0.5), 1.0);
        float angle0 = get_angle0(uv0)/PI;
        float angle = get_angle0(uv)/PI;
        float4 color = float4(jet(3.0*(uv0.y - uv.y)*0.5 + 0.5), 1.0);
        bool outside = uv.x >= 1.0 || uv.x <= 0.0;
        if(outside) color = float4(1,0,1,1);
        //float4 color = float4(jet(sin(100.0*((angle0 - angle)))*0.5 + 0.5), 1.0);
        if (pos.x >= 0 && pos.y >= 0 && pos.x < int(CANVAS_W) && pos.y < int(CANVAS_H)) stamp_disk(pos, STAMP_RADIUS, color);

        rk4_step_uv(uv, h, rs);
        phi += h;
        if (uv.x <= 0.0) break;
    }
}

[WorkgroupCount(WG_COUNT_IMAGE_X, WG_COUNT_IMAGE_Y, 1)]
[shader("compute")]
[numthreads(IMG_WG_SIZE, IMG_WG_SIZE, 1)]
void clear_canvas(uint3 id : SV_DispatchThreadID) {
    if (id.x >= CANVAS_W || id.y >= CANVAS_H) return;
    canvas[4*linearIndex(id.xy)].store(0u);
    canvas[4*linearIndex(id.xy)+1].store(0u);
    canvas[4*linearIndex(id.xy)+2].store(0u);
    canvas[4*linearIndex(id.xy)+3].store(0u);
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
        draw_one_curve(uv0, 0.005);
        draw_one_curve(uv0, -0.005);
    }
}

float4 load_color(uint2 p) {
    p.x = min(p.x, CANVAS_W - 1);
    p.y = min(p.y, CANVAS_H - 1);
    uint base = (linearIndex(p) << 2);
    float s = 1.0 / FLOAT_TO_INT_SCALE;
    float r = float(canvas[base + 0].load()) * s;
    float g = float(canvas[base + 1].load()) * s;
    float b = float(canvas[base + 2].load()) * s;
    float a = float(canvas[base + 3].load()) * s;
    return float4(r, g, b, a);
}

float2 fitted(float phi, float u) {
    float u0 = float(mouse.pos.y)/float(SCREEN_SIZE.y);
    float angle = -2.0*PHI_RANGE*PI*(phi - 0.5);
    angle = clamp(angle, -PI, PI);
    return normalize(float2(sin(angle), cos(angle) - 2.0*u));
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

    float4 c00 = load_color(i00);
    float4 c10 = load_color(i10);
    float4 c01 = load_color(i01);
    float4 c11 = load_color(i11);

    float4 c0 = lerp(c00, c10, f.x);
    float4 c1 = lerp(c01, c11, f.x);
    float4 c  = lerp(c0,  c1,  f.y);

    // simple per-channel exposure
    //c = c / c.w;
    //c.xy = normalize(c.xy - 0.5)*0.5 + 0.5;
    q/=canvasSize;
    //if(q.x < 0.5) 
    //c.xy = abs(c.xy - (fitted(q.x, q.y) * 0.5 + 0.5));

    const float k = 1.0;
    float4 v = 1.0 - exp(-k * c);

    screen[id.xy] = float4(v.rgb, saturate(v.a));
}