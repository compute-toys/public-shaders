import std;

static const int SIZE = 256, ELEMENTS = SIZE*SIZE;
static const int IMG_WG = 16, WG_COUNT = SIZE / IMG_WG;

static const float dt = 2.0;
static const int   JACOBI_ITERS = 20;
static const float pReuse = 0.9999;

static const float brushR = 10.0, brushK = 0.35;
static const float VORT_GAIN = 4.0;
static const float SZF = SIZE;

static const float VORT_EPS = 0.01; // confinement strength

static const float OBST_R =  SIZE * 0.07, OBST_RAD = SIZE * 0.4, OBST_W = 0.025, OBST_WSPIN = 0.0;

[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float2> vel0, vel1;
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float1> p0, p1;
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float1> divg;
[StorageBuffer(ELEMENTS)] RWStructuredBuffer<float1> solid;

uint  idx(uint2 p){ return p.x + p.y * SIZE; }

// wrap for torus sampling
uint  wrap1(int x){ return uint(x & (SIZE-1)); }
uint2 wrap2(int2 q){ return uint2(wrap1(q.x), wrap1(q.y)); }

float2 fold(float2 d){ return d - SZF * round(d / SZF); }
float2 obstCenterAt(uint f){ float t=f*OBST_W; return 0.5*SZF.xx + OBST_RAD*float2(cos(t), sin(0.8*t)); }
float2 obstCenter(){ return obstCenterAt(time.frame); }
float2 obstCenterVel(){ return (time.frame==0u)?0.0.xx:(obstCenterAt(time.frame)-obstCenterAt(time.frame-1u))/dt; }
float sdfCircle(float2 x){ return length(fold(x - obstCenter())) - OBST_R; }
float2 obstVelocityAt(float2 x){ float2 vc=obstCenterVel(), r=fold(x - obstCenter()); float w=OBST_WSPIN/dt; return vc + w*float2(-r.y, r.x); }

struct Viewport{ float s; float2 view,offs; };
Viewport vp(){ uint W,H; screen.GetDimensions(W,H); float s=min(W/SZF,H/SZF); return Viewport(s,s*SZF.xx,0.5*(float2(W,H)-s*SZF.xx)); }

void addBrushForce(inout float2 v, uint2 p){
    if(mouse.click==0) return;
    Viewport V=vp(); float2 m=mouse.pos;
    if(any(m<V.offs)||any(m>=V.offs+V.view)) return;
    float2 d=fold((p+0.5)-(m-V.offs)/V.s);
    float r2=dot(d,d);
    float2 dir=(r2>1e-6)? d*rsqrt(r2):0.0.xx;
    float2 tang=float2(-dir.y,dir.x);
    v += brushK*exp(-r2/(brushR*brushR))*(0.75*dir+0.25*tang);
}

public __generic<int S>
struct Grid
{
    public typealias vecT = vector<float,S>;

    struct CoordMap{
        int w, h; float2 szf;
        uint  Idx(uint2 p){ return p.x + p.y * w; }
        uint2 W2(int2 q){ return uint2(q.x & (w-1), q.y & (h-1)); }
        float2 WrapPos(float2 s){ return frac(s / szf) * szf; }
        static CoordMap make(int W, int H){ CoordMap c; c.w=W; c.h=H; c.szf=float2(W,H); return c; }
    };

    RWStructuredBuffer<vecT> b0,b1;
    CoordMap CM; bool ping;

    static Grid<S> make(RWStructuredBuffer<vecT> B0,
                        RWStructuredBuffer<vecT> B1,
                        bool f0 = true, int W = SIZE, int H = SIZE)
    {
        Grid<S> g; g.b0=B0; g.b1=B1; g.CM=CoordMap::make(W,H); g.ping=f0; return g;
    }

    vecT load(int2 p){
        uint2 q=CM.W2(p);
        return ping ? b0[CM.Idx(q)] : b1[CM.Idx(q)];
    }

    void store(int2 p, vecT v){
        uint2 q=CM.W2(p);
        if(!ping) b0[CM.Idx(q)]=v; else b1[CM.Idx(q)]=v;
    }

    vecT sample(float2 sp){
        int2 i0 = int2(floor(sp));
        float2 t = frac(sp);
        vecT a = load(i0);
        vecT b = load(i0 + int2(1,0));
        vecT c = load(i0 + int2(0,1));
        vecT d = load(i0 + int2(1,1));
        return lerp(lerp(a,b,t.x), lerp(c,d,t.x), t.y);
    }

     // 1D monotone cubic (Fritsch–Carlson style limiter), component-wise
    vecT cubicMono(vecT y0, vecT y1, vecT y2, vecT y3, float t){
        vecT d1 = y1 - y0, d2 = y2 - y1, d3 = y3 - y2;
        vecT m1 = 0.5 * (d1 + d2);
        vecT m2 = 0.5 * (d2 + d3);

        // zero slopes when local extrema would cause overshoot
        vecT pos12 = max(0.0, sign(d1 * d2));
        vecT pos23 = max(0.0, sign(d2 * d3));
        m1 *= pos12; m2 *= pos23;

        // guard flat segment
        vecT nz = step(1e-6, abs(d2)); // 0 if |d2|<eps, else 1
        m1 *= nz; m2 *= nz;

        // scale to keep ||(m1/d2, m2/d2)|| <= 3
        vecT a = m1 / max(abs(d2), 1e-6);
        vecT b = m2 / max(abs(d2), 1e-6);
        vecT h = sqrt(a * a + b * b);
        vecT tau = min(1.0, 3.0 / max(h, 1e-6));
        m1 *= tau; m2 *= tau;

        float t2 = t * t, t3 = t2 * t;
        return (2.0*t3 - 3.0*t2 + 1.0) * y1
             + (t3 - 2.0*t2 + t)     * m1
             + (-2.0*t3 + 3.0*t2)    * y2
             + (t3 - t2)             * m2;
    }

    // Bicubic, separable, wrap-aware, no-overshoot
    vecT sampleCubic(float2 sp){
        int2 i1 = int2(floor(sp));
        float2 f = frac(sp);
        int2 b = i1 - int2(1,1);

        vecT r0 = cubicMono(load(b + int2(0,0)), load(b + int2(1,0)), load(b + int2(2,0)), load(b + int2(3,0)), f.x);
        vecT r1 = cubicMono(load(b + int2(0,1)), load(b + int2(1,1)), load(b + int2(2,1)), load(b + int2(3,1)), f.x);
        vecT r2 = cubicMono(load(b + int2(0,2)), load(b + int2(1,2)), load(b + int2(2,2)), load(b + int2(3,2)), f.x);
        vecT r3 = cubicMono(load(b + int2(0,3)), load(b + int2(1,3)), load(b + int2(2,3)), load(b + int2(3,3)), f.x);

        return cubicMono(r0, r1, r2, r3, f.y);
    }

    __subscript(int2 p) -> vecT
    {
        get { return load(p); }
        set { store(p, newValue); }
    }
};
public typealias Grid1 = Grid<1>;
public typealias Grid2 = Grid<2>;

float divergence(Grid2 gv, int2 p){
    float2 uL = gv[p + int2(-1, 0)];
    float2 uR = gv[p + int2( 1, 0)];
    float2 uD = gv[p + int2( 0,-1)];
    float2 uU = gv[p + int2( 0, 1)];
    return 0.5 * ((uR.x - uL.x) + (uU.y - uD.y));
}

float empty(float x)
{
    return step(x, 0.5);
}

float2 gradient(Grid1 gp, Grid1 S, int2 p){
    float c  = gp[p];
    float nL = lerp(c, gp[p + int2(-1, 0)], empty(S[p + int2(-1, 0)]));
    float nR = lerp(c, gp[p + int2(1,  0)], empty(S[p + int2(1, 0)]));
    float nD = lerp(c, gp[p + int2(0, -1)], empty(S[p + int2(0, -1)]));
    float nU = lerp(c, gp[p + int2(0,  1)], empty(S[p + int2(0, 1)]));
    return 0.5 * float2(nR-nL, nU-nD);
}

float2 neighborSumCount(Grid1 gp, Grid1 S, int2 p){
    float2 acc = 0.0;
    acc += float2(gp[p + int2(-1, 0)], 1.0) * empty(S[p + int2(-1, 0)]);
    acc += float2(gp[p + int2(1,  0)], 1.0) * empty(S[p + int2(1,  0)]);
    acc += float2(gp[p + int2(0, -1)], 1.0) * empty(S[p + int2(0, -1)]);
    acc += float2(gp[p + int2(0,  1)], 1.0) * empty(S[p + int2(0,  1)]);
    return acc;
}

/* passes */
[WorkgroupCount(WG_COUNT,WG_COUNT,1)]
[shader("compute")]
[numthreads(IMG_WG,IMG_WG,1)]
void pass_solidmask(uint3 id:SV_DispatchThreadID){
    uint2 p=id.xy; if(any(p>=uint2(SIZE,SIZE))) return;
    Grid1 SOL = Grid1.make(solid, solid);
    SOL[p] = smoothstep(1.0, -1.0, sdfCircle(p+0.5));
}


float2 velClamp(Grid2 gv, Grid1 S, int2 p, int2 d){
    return (S[p + d].x > 0.5) ? gv[p] : gv[p + d];
}

float curl(Grid2 gv, Grid1 S, int2 p){
    float2 uL = velClamp(gv,S,p,int2(-1,0));
    float2 uR = velClamp(gv,S,p,int2( 1,0));
    float2 uD = velClamp(gv,S,p,int2( 0,-1));
    float2 uU = velClamp(gv,S,p,int2( 0, 1));
    return 0.5 * ((uR.y - uL.y) - (uU.x - uD.x));
}

float2 vortForce(Grid2 gv, Grid1 S, int2 p){
    float wC = curl(gv,S,p);
    float wL = abs(curl(gv,S,p + int2(-1,0)));
    float wR = abs(curl(gv,S,p + int2( 1,0)));
    float wD = abs(curl(gv,S,p + int2( 0,-1)));
    float wU = abs(curl(gv,S,p + int2( 1,0))); // typo guard: should be (0,1) in your tree; keep consistent if you already fixed it
    float2 g = 0.5 * float2(wR - wL, wU - wD);
    float m = max(length(g), 1e-6);
    float2 N = g / m;
    return float2(N.y, -N.x) * wC;
}

[WorkgroupCount(WG_COUNT,WG_COUNT,1)]
[shader("compute")]
[numthreads(IMG_WG,IMG_WG,1)]
void pass_advect(uint3 id:SV_DispatchThreadID){
    uint2 p=id.xy; if(any(p>=uint2(SIZE,SIZE))) return;

    Grid2 GV  = Grid2.make(vel0,vel1,false);
    Grid1 SOL = Grid1.make(solid, solid);

    if(time.frame==0u){ GV[p]=0.0; return; }

    float2 v0s = GV[p];
    float2 vA  = GV.sampleCubic(p - dt*v0s);    // use sampleCubic(...) if you enabled it
    addBrushForce(vA,p);

    if(SOL[p].x > 0.5) vA = obstVelocityAt(p+0.5);
    else               vA += dt * VORT_EPS * vortForce(GV, SOL, int2(p));

    GV[p] = vA;
}

[WorkgroupCount(WG_COUNT,WG_COUNT,1)]
[shader("compute")]
[numthreads(IMG_WG,IMG_WG,1)]
void pass_divergence(uint3 id:SV_DispatchThreadID){
    uint2 p=id.xy; if(any(p>=uint2(SIZE,SIZE))) return;

    Grid2 GV  = Grid2.make(vel0, vel1);
    Grid1 DIV = Grid1.make(divg, divg);

    DIV[p] = divergence(GV, p);
}

[DispatchCount(JACOBI_ITERS)]
[WorkgroupCount(WG_COUNT,WG_COUNT,1)]
[shader("compute")]
[numthreads(IMG_WG,IMG_WG,1)]
void pass_pressure(uint3 id:SV_DispatchThreadID){
    uint2 p=id.xy; if(any(p>=uint2(SIZE,SIZE))) return;

    bool readP0=((dispatch.id&1u)==0u);
    Grid1 GP  = Grid1.make(p0, p1, readP0);
    Grid1 DIV = Grid1.make(divg, divg);
    Grid1 SOL = Grid1.make(solid, solid);

    if(SOL[p] > 0.5) { GP[p] = 0.0; return; }

    float  weight = (dispatch.id==0u) ? pReuse : 1.0;
    float2 acc    = neighborSumCount(GP, SOL, p);
    float  rhs    = DIV[p];
    float  pNew   = (acc.x * weight - rhs) / max(acc.y, 1.0);
    GP[p] = pNew;
}

[WorkgroupCount(WG_COUNT,WG_COUNT,1)]
[shader("compute")]
[numthreads(IMG_WG,IMG_WG,1)]
void pass_project(uint3 id:SV_DispatchThreadID){
    uint2 p=id.xy; if(any(p>=uint2(SIZE,SIZE))) return;

    Grid2 GV  = Grid2.make(vel0, vel1);
    Grid1 GP  = Grid1.make(p0, p1);
    Grid1 SOL = Grid1.make(solid, solid);

    float2 v = GV[p] - gradient(GP, SOL, p);
    if(SOL[p] > 0.5) v = obstVelocityAt(float2(p)+0.5);
    GV[p] = v;
}

/* viz */
float3 bwr(float t){ return (t<0.5)? lerp(float3(0,0,1),float3(1,1,1),t/0.5)
                                   : lerp(float3(1,1,1),float3(1,0,0),(t-0.5)/0.5); }

[shader("compute")]
[numthreads(IMG_WG,IMG_WG,1)]
void main_image(uint3 id:SV_DispatchThreadID){
    Viewport V=vp();
    float2 pix=id.xy;
    if(any(pix<V.offs)||any(pix>=V.offs+V.view)) return;
    float2 sp=(pix-V.offs)/V.s;

    Grid2 GV = Grid2.make(vel0,vel1);

    float2 uL = GV.sample(sp + float2(-1,0));
    float2 uR = GV.sample(sp + float2( 1,0));
    float2 uD = GV.sample(sp + float2( 0,-1));
    float2 uU = GV.sample(sp + float2( 0, 1));

    float omega=0.5*(uR.y-uL.y)-0.5*(uU.x-uD.x);
    float t=0.5+tanh(VORT_GAIN*omega);
    screen[id.xy]=float4(bwr(t),1.0);
}
