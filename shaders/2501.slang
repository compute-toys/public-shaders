import std;

static const int SIZE = 256, ELEMENTS = SIZE*SIZE;
static const int IMG_WG = 16, WG_COUNT = SIZE / IMG_WG;

static const float dt = 2.0;
static const int   JACOBI_ITERS = 16;     // even → final pressure in p0
static const float pReuse = 0.9999;

static const float brushR = 28.0, brushK = 0.35;
static const float VORT_GAIN = 7.0;
static const float SZF = float(SIZE);

/* obstacle */
static const float OBST_R = 30.0, OBST_RAD = 100.0, OBST_W = 0.015, OBST_WSPIN = 0.0;

[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float2> vel0, vel1;
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float>  p0, p1, divg;
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<uint>   solid; // 0=fluid,1=solid

uint  idx(uint2 p){ return p.x + p.y * SIZE; }
uint  wrap1(int x){ return uint(x & (SIZE-1)); }
uint2 wrap2(int2 q){ return uint2(wrap1(q.x), wrap1(q.y)); }
bool  evenFrame(){ return (time.frame & 1u) == 0u; }

float2 fold(float2 d){ return d - SZF * round(d / SZF); }

float2 obstCenterAt(uint f){ float t=float(f)*OBST_W; return 0.5*SZF.xx + OBST_RAD*float2(cos(t), sin(0.8*t)); }
float2 obstCenter(){ return obstCenterAt(time.frame); }
float2 obstCenterVel(){ return (time.frame==0u)?0.0.xx:(obstCenterAt(time.frame)-obstCenterAt(time.frame-1u))/dt; }
float sdfCircle(float2 x){ return length(fold(x - obstCenter())) - OBST_R; }
float2 obstVelocityAt(float2 x){ float2 vc=obstCenterVel(), r=fold(x - obstCenter()); float w=OBST_WSPIN/dt; return vc + w*float2(-r.y, r.x); }

bool  solidAt(uint2 p){ return solid[idx(p)] != 0u; }
float2 loadV(uint2 p, bool from0){ return from0? vel0[idx(p)] : vel1[idx(p)]; }
void   storeV(uint2 p, float2 v, bool to0){ if (to0) vel0[idx(p)] = v; else vel1[idx(p)] = v; }
float  loadP(uint2 p, bool from0){ return from0? p0[idx(p)] : p1[idx(p)]; }
void   storeP(uint2 p, float v, bool to0){ if (to0) p0[idx(p)] = v; else p1[idx(p)] = v; }

float2 bilinearV(float2 sp, bool from0){
    sp = frac(sp / SZF) * SZF;
    int2 i0i = int2(floor(sp));
    float2 t = sp - float2(i0i);
    uint2 i0 = wrap2(i0i), i1 = wrap2(i0i + int2(1,1));
    float2 a = loadV(i0, from0);
    float2 b = loadV(uint2(i1.x, i0.y), from0);
    float2 c = loadV(uint2(i0.x, i1.y), from0);
    float2 d = loadV(i1, from0);
    return lerp(lerp(a,b,t.x), lerp(c,d,t.x), t.y);
}

/* viewport + brush */
struct Viewport { float s; float2 view, offs; };
Viewport vp(){ uint W,H; screen.GetDimensions(W,H); float s=min(float(W)/SZF,float(H)/SZF); return Viewport(s, s*float2(SIZE,SIZE), 0.5*(float2(W,H)-s*float2(SIZE,SIZE))); }

void addBrushForce(inout float2 v, uint2 p){
    if (mouse.click==0) return;
    Viewport V = vp();
    float2 mPix = float2(mouse.pos);
    if (any(mPix < V.offs) || any(mPix >= V.offs + V.view)) return;
    float2 d = fold((float2(p)+0.5) - (mPix - V.offs)/V.s);
    float r2 = dot(d,d);
    float2 dir = (r2>1e-6)? d*rsqrt(r2):0.0.xx;
    float2 tang = float2(-dir.y, dir.x);
    v += brushK * exp(-r2/(brushR*brushR)) * (0.75*dir + 0.25*tang);
}

/* neighbors */
static const int2 OFFS[4] = { int2(-1,0), int2(1,0), int2(0,-1), int2(0,1) };

/* 0) mask once per frame */
[WorkgroupCount(WG_COUNT, WG_COUNT, 1)]
[shader("compute")]
[numthreads(IMG_WG, IMG_WG, 1)]
void pass_solidmask(uint3 id: SV_DispatchThreadID){
    uint2 p = id.xy; if (any(p>=uint2(SIZE,SIZE))) return;
    solid[idx(p)] = (sdfCircle(float2(p)+0.5) < 0.0) ? 1u : 0u;
}

/* 1) advection + forces + early no-slip clamp */
[WorkgroupCount(WG_COUNT, WG_COUNT, 1)]
[shader("compute")]
[numthreads(IMG_WG, IMG_WG, 1)]
void pass_advect(uint3 id: SV_DispatchThreadID){
    uint2 p = id.xy; if (any(p>=uint2(SIZE,SIZE))) return;
    bool read0 = evenFrame(), write0 = !read0;

    if (time.frame==0u){
        vel0[idx(p)] = 0.0.xx; vel1[idx(p)] = 0.0.xx;
        p0[idx(p)] = 0.0; p1[idx(p)] = 0.0; divg[idx(p)] = 0.0;
        return;
    }

    float2 v0 = bilinearV(float2(p), read0);
    float2 vA = bilinearV(float2(p) - dt*v0, read0);
    addBrushForce(vA, p);
    if (solidAt(p)) vA = obstVelocityAt(float2(p)+0.5);
    storeV(p, vA, write0);
}

/* 2) divergence (plain stencil; solids already rigid) */
[WorkgroupCount(WG_COUNT, WG_COUNT, 1)]
[shader("compute")]
[numthreads(IMG_WG, IMG_WG, 1)]
void pass_divergence(uint3 id: SV_DispatchThreadID){
    uint2 p = id.xy; if (any(p>=uint2(SIZE,SIZE))) return;
    if (solidAt(p)){ divg[idx(p)] = 0.0; return; }
    bool latest0 = !evenFrame();
    int2 q = int2(p);
    float2 uL = loadV(wrap2(q+OFFS[0]), latest0);
    float2 uR = loadV(wrap2(q+OFFS[1]), latest0);
    float2 uD = loadV(wrap2(q+OFFS[2]), latest0);
    float2 uU = loadV(wrap2(q+OFFS[3]), latest0);
    divg[idx(p)] = 0.5 * ((uR.x - uL.x) + (uU.y - uD.y));
}

/* 3) Jacobi with warm-start; final in p0 */
[DispatchCount(JACOBI_ITERS)]
[WorkgroupCount(WG_COUNT, WG_COUNT, 1)]
[shader("compute")]
[numthreads(IMG_WG, IMG_WG, 1)]
void pass_pressure(uint3 id: SV_DispatchThreadID){
    uint2 p = id.xy; if (any(p>=uint2(SIZE,SIZE))) return;

    if (dispatch.id==0u){ float warm = pReuse * p0[idx(p)]; p0[idx(p)] = warm; p1[idx(p)] = warm; }

    bool readP0  = ((dispatch.id & 1u)==0u), writeP0 = !readP0;
    if (solidAt(p)){ storeP(p, 0.0, writeP0); return; }

    int2 q = int2(p);
    float2 acc = 0.0; // (sumP,count)
    [unroll] for (int i=0;i<4;++i){
        uint2 n = wrap2(q + OFFS[i]);
        acc += solidAt(n) ? 0.0 : float2(loadP(n, readP0), 1.0);
    }
    float pNew = (acc.x - divg[idx(p)]) / max(acc.y, 1.0);
    storeP(p, pNew, writeP0);
}

/* 4) projection; re-enforce no-slip */
[WorkgroupCount(WG_COUNT, WG_COUNT, 1)]
[shader("compute")]
[numthreads(IMG_WG, IMG_WG, 1)]
void pass_project(uint3 id: SV_DispatchThreadID){
    uint2 p = id.xy; if (any(p>=uint2(SIZE,SIZE))) return;
    bool latest0 = !evenFrame();
    int2 q = int2(p);

    float2 v = loadV(p, latest0);
    float  pC = p0[idx(p)];
    float  nP[4];
    [unroll] for (int i=0;i<4;++i){
        uint2 n = wrap2(q + OFFS[i]);
        nP[i] = solidAt(n) ? pC : p0[idx(n)];
    }
    v -= 0.5 * float2(nP[1]-nP[0], nP[3]-nP[2]); //substract gradient
    
    if (solidAt(p)){ v = obstVelocityAt(float2(p)+0.5); }
    storeV(p, v, latest0);
}

/* viz */
float3 bwr(float t){ return (t<0.5)? lerp(float3(0,0,1),float3(1,1,1),t/0.5)
                                   : lerp(float3(1,1,1),float3(1,0,0),(t-0.5)/0.5); }

[shader("compute")]
[numthreads(IMG_WG, IMG_WG, 1)]
void main_image(uint3 id: SV_DispatchThreadID){
    Viewport V = vp();
    float2 pix = float2(id.xy);
    if (any(pix < V.offs) || any(pix >= V.offs + V.view)) return;
    float2 sp = (pix - V.offs) / V.s;

    bool latest0 = !evenFrame();
    int2 q = int2(floor(sp));
    float2 uL = loadV(wrap2(q+OFFS[0]), latest0);
    float2 uR = loadV(wrap2(q+OFFS[1]), latest0);
    float2 uD = loadV(wrap2(q+OFFS[2]), latest0);
    float2 uU = loadV(wrap2(q+OFFS[3]), latest0);

    float omega = 0.5*(uR.y - uL.y) - 0.5*(uU.x - uD.x);
    float t = 0.5 + tanh(VORT_GAIN * omega);
    screen[id.xy] = float4(bwr(t), 1.0);
}
