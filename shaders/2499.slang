import std;

// --- grid ---
static const int SIZE = 128;
static const int VOXELS = SIZE*SIZE*SIZE;

// --- sim params ---
static const int STEPS = 2;
static const float c2   = 1.0;
static const float dt   = 0.20;
static const float damp = 0.02;
static const float normK= 0.5;

// --- init (periodic random) ---
static const int   MODES = 12;
static const int   KMIN  = 1;
static const int   KMAX  = 8;
static const float initAmp = 0.3;
static const float initNormAlpha = 0.85;

// --- workgroup ---
static const int WGX = 8, WGY = 8, WGZ = 4;
static const int GCX = SIZE/WGX, GCY = SIZE/WGY, GCZ = SIZE/WGZ;

// --- volume render ---
static const float CAM_RAD = 1.5 * float(SIZE);
static const float FOVY = 1.04719755; // 60 deg
static const int   MAX_STEPS = 768;
static const float STEP_LEN  = 1.0;
static const float COL_GAIN  = 30.0;
static const float ALPHA_GAIN= 0.025;
static const float GAMMA     = 0.8;

[StorageBuffer(VOXELS)] RWStructuredBuffer<float3> u0_u;
[StorageBuffer(VOXELS)] RWStructuredBuffer<float3> u0_v;
[StorageBuffer(VOXELS)] RWStructuredBuffer<float3> u1_u;
[StorageBuffer(VOXELS)] RWStructuredBuffer<float3> u1_v;

uint  lin3(uint3 p){ return p.x + SIZE*(p.y + SIZE*p.z); }
uint  wrap1(int x){ return uint(x & (SIZE-1)); }
uint3 wrap3(int3 q){ return uint3(wrap1(q.x), wrap1(q.y), wrap1(q.z)); }

float3 loadU(uint3 p, bool from0){ uint i=lin3(p); return from0? u0_u[i]:u1_u[i]; }
float3 loadV(uint3 p, bool from0){ uint i=lin3(p); return from0? u0_v[i]:u1_v[i]; }
void   storeU(uint3 p, float3 v, bool to0){ uint i=lin3(p); if(to0) u0_u[i]=v; else u1_u[i]=v; }
void   storeV(uint3 p, float3 v, bool to0){ uint i=lin3(p); if(to0) u0_v[i]=v; else u1_v[i]=v; }

float3 sampleU(int3 q, bool from0){ return loadU(wrap3(q), from0); }

float3 lapU3(int3 q, bool from0){
    float3 c = sampleU(q,               from0);
    float3 L = sampleU(q+int3(-1,0,0),  from0);
    float3 R = sampleU(q+int3( 1,0,0),  from0);
    float3 D = sampleU(q+int3(0,-1,0),  from0);
    float3 U = sampleU(q+int3(0, 1,0),  from0);
    float3 B = sampleU(q+int3(0,0,-1),  from0);
    float3 F = sampleU(q+int3(0,0, 1),  from0);
    return (L+R+U+D+B+F - 6.0*c);
}

float rnd(uint s){ return frac(sin(float(s)*12.9898 + 78.233)*43758.5453); }
float3 softNorm3(float3 u, float a){
    float l2 = max(dot(u,u), 1e-12);
    return lerp(u, u*rsqrt(l2), a);
}

float3 initU(uint3 p){
    float3 acc = float3(0,0,0);
    float3 x = (float3(p) + 0.5) / float(SIZE);
    for(uint i=0u;i<MODES;++i){
        int kx = KMIN + int(floor(rnd(7919u*i+1u)*float(KMAX-KMIN+1)));
        int ky = KMIN + int(floor(rnd(104729u*i+7u)*float(KMAX-KMIN+1)));
        int kz = KMIN + int(floor(rnd(13007u*i+9u)*float(KMAX-KMIN+1)));
        if (rnd(1223u*i+3u) < 0.5) kx = -kx;
        if (rnd(1229u*i+5u) < 0.5) ky = -ky;
        if (rnd(1231u*i+6u) < 0.5) kz = -kz;
        if (kx==0 && ky==0 && kz==0) kz = 1;
        float phase = 6.2831853 * rnd(15485863u*i+11u);
        float s = sin(6.2831853 * (float(kx)*x.x + float(ky)*x.y + float(kz)*x.z) + phase);
        float a = 6.2831853*rnd(9001u*i+13u);
        float b = acos(2.0*rnd(9013u*i+17u)-1.0);
        float3 q = float3(cos(a)*sin(b), sin(a)*sin(b), cos(b));
        float w = initAmp / (1.0 + float(kx*kx+ky*ky+kz*kz));
        acc += w * s * q;
    }
    return softNorm3(acc, initNormAlpha);
}

[DispatchCount(STEPS)]
[WorkgroupCount(GCX, GCY, GCZ)]
[shader("compute")]
[numthreads(WGX, WGY, WGZ)]
void step(uint3 tid : SV_DispatchThreadID){
    uint3 p = tid.xyz;
    if (any(p >= uint3(SIZE,SIZE,SIZE))) return;

    if (time.frame%512==0u && dispatch.id==0u){
        float3 u = initU(p);
        u0_u[lin3(p)] = u; u1_u[lin3(p)] = u;
        u0_v[lin3(p)] = float3(0,0,0); u1_v[lin3(p)] = float3(0,0,0);
        return;
    }

    uint t  = time.frame*uint(STEPS) + dispatch.id;
    bool r0 = (t & 1u)==0u, w0 = !r0;

    float3 u = loadU(p, r0);
    float3 v = loadV(p, r0);

    float3 lap = c2 * lapU3(int3(p), r0);
    float n2 = dot(u,u);
    float3 normForce = normK * (1.0 - n2) * u;

    v += dt * (lap + normForce);     // dv/dt
    u += dt * (v - damp * u);        // du/dt

    storeU(p, u, w0);
    storeV(p, v, w0);
}

// --- energy density for volume rendering ---
float energyAt(int3 q, bool from0){
    uint3 pu = wrap3(q);
    float3 u = loadU(pu, from0);
    float3 v = loadV(pu, from0);
    float3 ux = 0.5*( sampleU(q+int3(1,0,0),from0) - sampleU(q+int3(-1,0,0),from0) );
    float3 uy = 0.5*( sampleU(q+int3(0,1,0),from0) - sampleU(q+int3(0,-1,0),from0) );
    float3 uz = 0.5*( sampleU(q+int3(0,0,1),from0) - sampleU(q+int3(0,0,-1),from0) );
    float grad2 = dot(ux,ux)+dot(uy,uy)+dot(uz,uz);
    float pot   = (1.0 - dot(u,u)); pot *= pot;
    return 0.5*dot(v,v) + 0.5*c2*grad2 + 0.5*normK*pot;
}

float3 heat(float x){
    x = clamp(x,0.0,1.0);
    float3 c1=float3(0.0,0.0,0.0), c2=float3(0.2,0.1,0.5), c3=float3(0.9,0.4,0.1), c4=float3(1.0,0.95,0.8);
    if (x<0.33) return lerp(c1,c2,x/0.33);
    if (x<0.66) return lerp(c2,c3,(x-0.33)/0.33);
    return lerp(c3,c4,(x-0.66)/0.34);
}

bool rayBox(float3 ro, float3 rd, out float t0, out float t1){
    float3 bmin=float3(0,0,0), bmax=float3(float(SIZE),float(SIZE),float(SIZE));
    float3 inv = 1.0 / rd;
    float3 tA = (bmin - ro) * inv;
    float3 tB = (bmax - ro) * inv;
    float3 tmin3 = min(tA,tB), tmax3 = max(tA,tB);
    t0 = max(max(tmin3.x, tmin3.y), tmin3.z);
    t1 = min(min(tmax3.x, tmax3.y), tmax3.z);
    return t1 > max(t0, 0.0);
}

[shader("compute")]
[numthreads(16,16,1)]
void main_image(uint3 id : SV_DispatchThreadID){
    uint W,H; screen.GetDimensions(W,H);
    float2 res=float2(W,H);
    float2 uv = (float2(id.xy)+0.5)/res*2.0 - 1.0;
    uv.x *= res.x/res.y;

    float yaw   = 6.2831853 * (float(mouse.pos.x)/float(W));
    float pitch = 3.14159265 * (float(mouse.pos.y)/float(H) - 0.5) * 0.9;
    float cy=cos(yaw), sy=sin(yaw), cp=cos(pitch), sp=sin(pitch);

    float3 center = float3(0.5*SIZE, 0.5*SIZE, 0.5*SIZE);
    float3 dirToCam = float3(cp*cy, sp, cp*sy);
    float3 camPos = center + CAM_RAD * dirToCam;
    float3 f = normalize(center - camPos);
    float3 up0 = float3(0,1,0);
    if (abs(dot(f,up0)) > 0.98) up0 = float3(0,0,1);
    float3 r = normalize(cross(f, up0));
    float3 u = cross(r, f);

    float tanH = tan(0.5*FOVY);
    float3 rd = normalize(f + r*(uv.x*tanH) + u*(uv.y*tanH));
    float3 ro = camPos;

    uint last = time.frame*uint(STEPS) + (uint(STEPS)-1u);
    bool latest0 = !((last & 1u)==0u);

    float t0,t1;
    float4 acc = float4(0,0,0,0);
    if (rayBox(ro, rd, t0, t1)){
        float t = max(t0, 0.0);
        [loop]
        for (int i=0;i<MAX_STEPS && t<t1 && acc.a<0.995;i++){
            float3 pos = ro + rd*t;
            int3 q = int3(round(pos - 0.5)); // nearest neighbor to cell center
            float e = energyAt(q, latest0);
            float a = 1.0 - exp(-ALPHA_GAIN * e * STEP_LEN);
            float val = pow(clamp( e, 0.0, 1.0), GAMMA);
            float3 col = COL_GAIN *heat(val);
            acc.rgb += (1.0 - acc.a) * a * col;
            acc.a   += (1.0 - acc.a) * a;
            t += STEP_LEN;
        }
    }
    screen[id.xy] = float4(acc.rgb, 1.0);
}
