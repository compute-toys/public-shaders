import std;

static const int SIZE = 1024;
static const int ELEMENTS = SIZE * SIZE;
static const int IMG_WG = 16;
static const int WG_COUNT = SIZE / IMG_WG;

static const int STEPS = 4;
static const float c2 = 1.0;
static const float dt = 0.45;
static const float damp = 0.03;
static const float normK = 0.1;
static const float brushR = 30.0;
static const float brushK = 0.2;

// viz options
static const bool  SHOW_LAPLACIAN = true;
static const float LAPL_GAIN = 2.0;
static const float LEN_GAIN = 2.0;
static const float LEN_GAIN_LAPL = 4.0;
static const float LEN_GAMMA = 0.75;

// random periodic init (integer wavevectors)
static const int   MODES = 16;
static const int   KMIN = 1;
static const int   KMAX = 8;
static const float initAmp = 0.25;
static const float initNormAlpha = 0.85;

[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float4> u0; // .xy = u, .zw = v
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float4> u1;

uint lin(uint2 p){ return p.x + p.y * SIZE; }

float4 loadS(uint2 p, bool from0){ uint i = lin(p); return from0 ? u0[i] : u1[i]; }
void   storeS(uint2 p, float4 s, bool to0){ uint i = lin(p); if(to0) u0[i]=s; else u1[i]=s; }

// ---- periodic (toroidal) addressing ----
uint wrap1(int x){ return uint(x & (SIZE-1)); }          // SIZE is power of two
uint2 wrap2(int2 q){ return uint2(wrap1(q.x), wrap1(q.y)); }

// sim sampling (periodic)
float2 sampleU(int2 q, bool from0){ return loadS(wrap2(q), from0).xy; }
float2 lapU(int2 q, bool from0){
    float2 c = sampleU(q, from0);
    float2 L = sampleU(q + int2(-1,0), from0);
    float2 R = sampleU(q + int2( 1,0), from0);
    float2 D = sampleU(q + int2(0,-1), from0);
    float2 U = sampleU(q + int2(0, 1), from0);
    return (L + R + U + D - 4.0 * c);
}

// tiny hash + soft norm
float rnd(uint s){ return frac(sin(float(s)*12.9898 + 78.233)*43758.5453); }
float2 softNorm(float2 u, float a){ float l2=max(dot(u,u),1e-12); return lerp(u, u*rsqrt(l2), a); }

// strictly periodic random field: sum sin(2π k·x), k integer
float2 initU(uint2 p){
    float2 x = float2(p) / float(SIZE);
    float2 acc = float2(0.0, 0.0);
    for(uint i=0u;i<MODES;++i){
        int kx = KMIN + int(floor(rnd(7919u*i+1u) * float(KMAX-KMIN+1)));
        int ky = KMIN + int(floor(rnd(104729u*i+7u)* float(KMAX-KMIN+1)));
        if (rnd(1223u*i+3u) < 0.5) kx = -kx;
        if (rnd(1229u*i+5u) < 0.5) ky = -ky;
        if (kx==0 && ky==0) ky = 1;
        float phase = 6.2831853 * rnd(15485863u*i+11u);
        float s = sin(6.2831853 * (float(kx)*x.x + float(ky)*x.y) + phase);
        float2 q = float2(cos(6.2831853*rnd(9001u*i+13u)), sin(6.2831853*rnd(9013u*i+17u)));
        float w = initAmp / (1.0 + float(kx*kx + ky*ky)); // de-emphasize high-k
        acc += w * s * q;
    }
    return softNorm(acc, initNormAlpha);
}

// mouse brush on the torus (minimal image distance)
void addBrush(inout float2 v, uint2 p){
    if (mouse.click == 0) return;
    uint W,H; screen.GetDimensions(W,H);
    float s = min(float(W)/float(SIZE), float(H)/float(SIZE));
    float2 view = s * float2(SIZE, SIZE);
    float2 offs = 0.5 * (float2(W,H) - view);
    float2 mPix = float2(mouse.pos);
    if (any(mPix < offs) || any(mPix >= offs + view)) return;
    float2 mSim = (mPix - offs) / s;

    float2 d = (float2(p) + 0.5) - mSim;
    float SZ = float(SIZE);
    if (d.x >  0.5*SZ) d.x -= SZ; else if (d.x < -0.5*SZ) d.x += SZ;
    if (d.y >  0.5*SZ) d.y -= SZ; else if (d.y < -0.5*SZ) d.y += SZ;

    float r2 = dot(d,d);
    v += brushK * exp(-r2/(brushR*brushR)) * ((r2>1e-6)? d*rsqrt(r2):float2(0,0));
}

// ---- integrator ----
[DispatchCount(STEPS)]
[WorkgroupCount(WG_COUNT, WG_COUNT, 1)]
[shader("compute")]
[numthreads(IMG_WG, IMG_WG, 1)]
void step(uint3 id : SV_DispatchThreadID){
    uint2 p = id.xy;
    if (any(p >= uint2(SIZE,SIZE))) return;

    if (time.frame == 0u && dispatch.id == 0u){
        float2 u = initU(p);
        float4 S = float4(u, 0.0, 0.0);
        u0[lin(p)] = S; u1[lin(p)] = S;
        return;
    }

    uint t = time.frame * uint(STEPS) + dispatch.id;
    bool read0  = (t & 1u) == 0u;
    bool write0 = !read0;

    float4 S = loadS(p, read0);                 // S.xy=u, S.zw=v
    addBrush(S.zw, p);

    float2 lap = c2 * lapU(int2(p), read0);
    float n2 = dot(S.xy, S.xy);
    float2 normForce = normK * (1.0 - n2) * S.xy;

    S.zw += dt * (lap + normForce - 0.0*damp * S.zw); // dv/dt
    S.xy += dt * (S.zw - damp * S.xy);                // du/dt

    storeS(p, S, write0);
}

// ---- bilinear sampling with wrap ----
float2 texU(uint2 p, bool from0){ return from0 ? u0[lin(p)].xy : u1[lin(p)].xy; }

float2 bilinearU(float2 sp, bool from0){
    float SZ = float(SIZE);
    sp = frac(sp / SZ) * SZ;
    int2 i0i = int2(floor(sp));
    int2 i1i = i0i + int2(1,1);
    uint2 i0 = wrap2(i0i);
    uint2 i1 = wrap2(i1i);
    float2 t = sp - float2(i0i);
    float2 a = texU(uint2(i0.x, i0.y), from0);
    float2 b = texU(uint2(i1.x, i0.y), from0);
    float2 c = texU(uint2(i0.x, i1.y), from0);
    float2 d = texU(uint2(i1.x, i1.y), from0);
    return lerp(lerp(a, b, t.x), lerp(c, d, t.x), t.y);
}

float2 bilinearLap(float2 sp, bool from0){
    float2 c = bilinearU(sp, from0);
    float2 L = bilinearU(sp + float2(-1,0), from0);
    float2 R = bilinearU(sp + float2( 1,0), from0);
    float2 D = bilinearU(sp + float2(0,-1), from0);
    float2 U = bilinearU(sp + float2(0, 1), from0);
    return (L + R + U + D - 4.0 * c);
}

// hue by angle, brightness by |vec|
float3 hue_to_rgb(float a){
    float t = (a / 3.14159265f + 1.0) * 0.5;
    float3 k=float3(1.0, 2.0/3.0, 1.0/3.0);
    float3 p=abs(frac(float3(t,t,t)+k)*6.0-3.0)-1.0;
    return clamp(p, 0.0, 1.0);
}

[shader("compute")]
[numthreads(IMG_WG, IMG_WG, 1)]
void main_image(uint3 id : SV_DispatchThreadID){
    uint W,H; screen.GetDimensions(W,H);

    float s = min(float(W)/float(SIZE), float(H)/float(SIZE));
    float2 view = s * float2(SIZE, SIZE);
    float2 offs = 0.5 * (float2(W,H) - view);

    float2 pix = float2(id.xy);
    if (any(pix < offs) || any(pix >= offs + view)) return;

    float2 sp = (pix - offs) / s;

    uint last = time.frame * uint(STEPS) + (uint(STEPS) - 1u);
    bool lastRead0 = (last & 1u) == 0u;
    bool latest0 = !lastRead0;

    float2 vec = SHOW_LAPLACIAN ? (LAPL_GAIN * bilinearLap(sp, latest0))
                                : bilinearU(sp, latest0);

    float mag = sqrt(max(dot(vec,vec), 0.0));
    float gain = SHOW_LAPLACIAN ? LEN_GAIN_LAPL : LEN_GAIN;
    float val = pow(clamp(mag * gain, 0.0, 1.0), LEN_GAMMA);

    float3 base = hue_to_rgb(atan2(vec.y, vec.x));
    screen[id.xy] = float4(base * val, 1.0);
}
