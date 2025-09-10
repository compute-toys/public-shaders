import std;

static const int SIZE = 1024;
static const int ELEMENTS = SIZE*SIZE;
static const int IMG_WG = 16;
static const int WG_COUNT = SIZE/IMG_WG;

static const int STEPS = 4;
static const float c2 = 1.0;
static const float dt = 0.45;
static const float damp = 0.03;      // on u only
static const float normK = 0.5;      // |u|→1 potential strength

// viz
static const bool  SHOW_LAPLACIAN = false;
static const float LAPL_GAIN = 2.0;
static const float LEN_GAIN = 2.0;
static const float LEN_GAIN_LAPL = 4.0;
static const float LEN_GAMMA = 0.75;

// init
static const int   MODES = 16;
static const int   KMIN = 1;
static const int   KMAX = 128;
static const float initAmp = 0.25;
static const float initNormAlpha = 0.85;

// brush
static const float brushR = 30.0;
static const float brushK = 0.2;

[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float3> u0_u;
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float3> u0_v;
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float3> u1_u;
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float3> u1_v;

uint lin(uint2 p){ return p.x + p.y*SIZE; }

uint wrap1(int x){ return uint(x & (SIZE-1)); }
uint2 wrap2(int2 q){ return uint2(wrap1(q.x), wrap1(q.y)); }

float3 loadU(uint2 p, bool from0){ uint i=lin(p); return from0 ? u0_u[i] : u1_u[i]; }
float3 loadV(uint2 p, bool from0){ uint i=lin(p); return from0 ? u0_v[i] : u1_v[i]; }
void   storeU(uint2 p, float3 v, bool to0){ uint i=lin(p); if(to0) u0_u[i]=v; else u1_u[i]=v; }
void   storeV(uint2 p, float3 v, bool to0){ uint i=lin(p); if(to0) u0_v[i]=v; else u1_v[i]=v; }

float3 sampleU(int2 q, bool from0){ return loadU(wrap2(q), from0); }
float3 lapU3(int2 q, bool from0){
    float3 c = sampleU(q, from0);
    float3 L = sampleU(q+int2(-1,0), from0);
    float3 R = sampleU(q+int2( 1,0), from0);
    float3 D = sampleU(q+int2(0,-1), from0);
    float3 U = sampleU(q+int2(0, 1), from0);
    return (L+R+U+D - 4.0*c);
}

float rnd(uint s){ return frac(sin(float(s)*12.9898 + 78.233)*43758.5453); }
float3 softNorm3(float3 u, float a){
    float l2 = max(dot(u,u), 1e-12);
    return lerp(u, u*rsqrt(l2), a);
}

float3 initU(uint2 p){
    float2 x = float2(p)/float(SIZE);
    float3 acc = float3(0,0,0);
    for(uint i=0u;i<MODES;++i){
        int kx = KMIN + int(floor(rnd(7919u*i+1u)*float(KMAX-KMIN+1)));
        int ky = KMIN + int(floor(rnd(104729u*i+7u)*float(KMAX-KMIN+1)));
        if (rnd(1223u*i+3u) < 0.5) kx = -kx;
        if (rnd(1229u*i+5u) < 0.5) ky = -ky;
        if (kx==0 && ky==0) ky = 1;
        float phase = 6.2831853*rnd(15485863u*i+11u);
        float s = sin(6.2831853*(float(kx)*x.x + float(ky)*x.y) + phase);
        // random 3D direction
        float a = 6.2831853*rnd(9001u*i+13u);
        float z = 2.0*rnd(9013u*i+17u)-1.0;
        float r = sqrt(max(1.0 - z*z, 0.0));
        float3 q = float3(r*cos(a), r*sin(a), z);
        float w = initAmp/(1.0 + float(kx*kx + ky*ky));
        acc += w*s*q;
    }
    return softNorm3(acc, initNormAlpha);
}

void addBrush(inout float3 v, uint2 p){
    if (mouse.click == 0) return;
    uint W,H; screen.GetDimensions(W,H);
    float s = min(float(W)/float(SIZE), float(H)/float(SIZE));
    float2 view = s*float2(SIZE,SIZE);
    float2 offs = 0.5*(float2(W,H)-view);
    float2 mPix = float2(mouse.pos);
    if (any(mPix < offs) || any(mPix >= offs+view)) return;
    float2 mSim = (mPix - offs)/s;
    float2 d2 = (float2(p)+0.5) - mSim;
    float SZ = float(SIZE);
    if (d2.x >  0.5*SZ) d2.x -= SZ; else if (d2.x < -0.5*SZ) d2.x += SZ;
    if (d2.y >  0.5*SZ) d2.y -= SZ; else if (d2.y < -0.5*SZ) d2.y += SZ;
    float r2 = dot(d2,d2);
    float2 imp2 = brushK * exp(-r2/(brushR*brushR)) * ((r2>1e-6)? d2*rsqrt(r2):float2(0,0));
    v += float3(imp2, 0.0);
}

[DispatchCount(STEPS)]
[WorkgroupCount(WG_COUNT, WG_COUNT, 1)]
[shader("compute")]
[numthreads(IMG_WG, IMG_WG, 1)]
void step(uint3 id : SV_DispatchThreadID){
    uint2 p = id.xy;
    if (any(p >= uint2(SIZE,SIZE))) return;

    if (time.frame==0u && dispatch.id==0u){
        float3 u = initU(p);
        u0_u[lin(p)] = u; u1_u[lin(p)] = u;
        u0_v[lin(p)] = float3(0,0,0); u1_v[lin(p)] = float3(0,0,0);
        return;
    }

    uint t = time.frame*uint(STEPS) + dispatch.id;
    bool read0 = (t & 1u)==0u, write0 = !read0;

    float3 u = loadU(p, read0);
    float3 v = loadV(p, read0);
    addBrush(v, p);

    float3 lap = c2 * lapU3(int2(p), read0);
    float n2 = dot(u,u);
    float3 normForce = normK * (1.0 - n2) * u;

    v += dt * (lap + normForce);     // dv/dt
    u += dt * (v - damp * u);        // du/dt

    storeU(p, u, write0);
    storeV(p, v, write0);
}

float3 texU(uint2 p, bool from0){ return from0 ? u0_u[lin(p)] : u1_u[lin(p)]; }

float3 bilinearU(float2 sp, bool from0){
    float SZ = float(SIZE);
    sp = frac(sp/SZ)*SZ;
    int2 i0i = int2(floor(sp));
    int2 i1i = i0i + int2(1,1);
    uint2 i0 = wrap2(i0i), i1 = wrap2(i1i);
    float2 t = sp - float2(i0i);
    float3 a = texU(uint2(i0.x,i0.y), from0);
    float3 b = texU(uint2(i1.x,i0.y), from0);
    float3 c = texU(uint2(i0.x,i1.y), from0);
    float3 d = texU(uint2(i1.x,i1.y), from0);
    return lerp(lerp(a,b,t.x), lerp(c,d,t.x), t.y);
}

float3 bilinearLap3(float2 sp, bool from0){
    float3 c = bilinearU(sp, from0);
    float3 L = bilinearU(sp+float2(-1,0), from0);
    float3 R = bilinearU(sp+float2( 1,0), from0);
    float3 D = bilinearU(sp+float2(0,-1), from0);
    float3 U = bilinearU(sp+float2(0, 1), from0);
    return (L+R+U+D - 4.0*c);
}

float3 dir_to_rgb(float3 d){
    float3 n = normalize(d + 1e-12);
    return clamp(0.5*(n+1.0), 0.0, 1.0); // direction → color
}

[shader("compute")]
[numthreads(IMG_WG, IMG_WG, 1)]
void main_image(uint3 id : SV_DispatchThreadID){
    uint W,H; screen.GetDimensions(W,H);
    float s = min(float(W)/float(SIZE), float(H)/float(SIZE));
    float2 view = s*float2(SIZE,SIZE);
    float2 offs = 0.5*(float2(W,H)-view);

    float2 pix = float2(id.xy);
    if (any(pix < offs) || any(pix >= offs+view)) return;

    float2 sp = (pix - offs)/s;

    uint last = time.frame*uint(STEPS) + (uint(STEPS)-1u);
    bool latest0 = !((last & 1u)==0u);

    float3 vec = SHOW_LAPLACIAN ? (LAPL_GAIN * bilinearLap3(sp, latest0))
                                : bilinearU(sp, latest0);

    float mag = sqrt(max(dot(vec,vec), 0.0));
    float gain = SHOW_LAPLACIAN ? LEN_GAIN_LAPL : LEN_GAIN;
    float val = pow(clamp(mag*gain, 0.0, 1.0), LEN_GAMMA);

    float3 base = dir_to_rgb(vec);
    screen[id.xy] = float4(base * val, 1.0);
}
