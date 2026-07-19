import std;

/* ================================================================
   Oriented-particle membranes -- compute.toys (Slang)
   ----------------------------------------------------------------
   Inertial membrane model, identical physics to the WebGL original:
     gated pair potential  Phi = uR + (1-phi)  /  uA*phi
     phi = 1 + mu*(A-1),  A = ni.nj - aij*aji + s*(aji-aij) - s^2
     symplectic kick-drift leapfrog, exact Rodrigues director
     rotation, pairwise momentum-conserving DPD friction.

   !! CODEGEN WARNING: never write -(a*x + b*y) with vector x,y.
   The Slang->WGSL backend mis-lowers it as (-a*x + b*y), which
   breaks Newton's third law here. Use binary subtraction only
   (see gPhiR). Trig arguments must stay within ~[0, 2*pi]
   (WGSL accuracy domain): azimuths use frac(), camera wraps.

   Structure: two lanes. Step sub-dispatches read lane R only and
   write their own index to lane W (force + integrate atomic on a
   consistent snapshot); sync sub-dispatches copy W -> R. Correct
   under any sub-dispatch ordering. Seeding via SEED_TAG flag in
   pR.w; rewind (time.frame < 2) always reseeds; invalid particles
   self-heal.
   ================================================================ */

/* ---------------- parameters (edit me) ---------------- */
static const int   NUM      = 4096;   // particles (power of two, id <= 16 bits)
static const int   PRESET   = 1;      // 0: shells bounce / 1: counter-spin / 2: discs fuse
static const int   SUBSTEPS = 3;      // leapfrog steps per frame

static const float DT     = 0.025;
static const float ZETA   = 4.0;
static const float RCUT   = 2.6;
static const float IMPACT = 0.45;
static const float MASS = 1.0, INER = 0.4;

static const float SEED_TAG = 8.5;    // bump when changing buffer layout
static const float PI   = 3.14159265359;
static const float PI2  = 1.57079632679;
static const float TAU2 = 6.28318530718;
static const float GAF  = 0.38196601; // golden-angle fraction = (3 - sqrt(5)) / 2
static const uint  NPH  = uint(NUM / 2);

float shellR()      { return sqrt(float(NPH) * 0.866 / (4.0 * PI)); }
float discR()       { return sqrt(float(NPH) * 0.9 / PI); }
float presetMu()    { return PRESET == 2 ? 2.2  : 3.0; }
float presetTheta() { return PRESET == 2 ? 0.05 : min(1.0 / shellR(), 0.3); }
float presetGamma() { return PRESET == 2 ? 0.05 : 0.02; }
float wallR()       { return PRESET == 2 ? 3.5 * discR() + 12.0 : 4.5 * shellR() + 10.0; }
float camDist()     { return PRESET == 2 ? 4.2 * discR() : 5.2 * shellR(); }
float goldenAngle(uint k) { return frac(float(k) * GAF) * TAU2; }

static const int RESX = 1280, RESY = 720;
static const float FOVF = 2.0702;                 // 1 / tan(0.9 rad / 2)
static const int PWG = 256, PWG_N = NUM / PWG;
static const int CLR_WG = (RESX * RESY) / 256;

/* ---------------- buffers: R lane (read), W lane (write) ---------------- */
[StorageBuffer(NUM)] RWStructuredBuffer<float4> pR;   // .w == SEED_TAG when seeded
[StorageBuffer(NUM)] RWStructuredBuffer<float4> dR;
[StorageBuffer(NUM)] RWStructuredBuffer<float4> vR;
[StorageBuffer(NUM)] RWStructuredBuffer<float4> wR;
[StorageBuffer(NUM)] RWStructuredBuffer<float4> pW;
[StorageBuffer(NUM)] RWStructuredBuffer<float4> dW;
[StorageBuffer(NUM)] RWStructuredBuffer<float4> vW;
[StorageBuffer(NUM)] RWStructuredBuffer<float4> wW;
[StorageBuffer(RESX*RESY)] RWStructuredBuffer<Atomic<uint>> splatB;
[StorageBuffer(4)] RWStructuredBuffer<float4> camB;   // [0]=th,ph,dist ; [1]=lastMouse,click

/* ================= state ================= */
struct Particle {
    float3 p; float3 n; float3 v; float3 w;
    bool valid() {                       // false for NaN too
        float nn = dot(n, n);
        return dot(p, p) < 1e12 && dot(v, v) < 1e8 && nn > 0.25 && nn < 4.0;
    }
};

/* lane access; methods index globals directly (WGSL cannot pass
   storage buffers by value, so no buffer-typed struct members) */
struct Sim {
    static bool  seeded(uint j) { return pR[j].w == SEED_TAG; }
    static float3 pos(uint j) { return pR[j].xyz; }
    static float3 dir(uint j) { return dR[j].xyz; }
    static float3 vel(uint j) { return vR[j].xyz; }
    static float3 omg(uint j) { return wR[j].xyz; }

    static Particle read(uint j) {
        Particle q;
        q.p = pR[j].xyz; q.n = dR[j].xyz; q.v = vR[j].xyz; q.w = wR[j].xyz;
        return q;
    }
    static void write(uint j, Particle q) {          // step: W lane only
        pW[j] = float4(q.p, SEED_TAG); dW[j] = float4(q.n, 0);
        vW[j] = float4(q.v, 0);        wW[j] = float4(q.w, 0);
    }
    static void seed(uint j, Particle q) {           // both lanes
        pR[j] = float4(q.p, SEED_TAG); dR[j] = float4(q.n, 0);
        vR[j] = float4(q.v, 0);        wR[j] = float4(q.w, 0);
        pW[j] = float4(q.p, SEED_TAG); dW[j] = float4(q.n, 0);
        vW[j] = float4(q.v, 0);        wW[j] = float4(q.w, 0);
    }
    static void sync(uint j) {                       // W -> R
        pR[j] = pW[j]; dR[j] = dW[j]; vR[j] = vW[j]; wR[j] = wW[j];
    }
};

/* ================= presets ================= */
Particle initParticle(uint i) {
    Particle o;
    bool second = i >= NPH;
    uint k = second ? i - NPH : i;
    float v = IMPACT;

    if (PRESET == 2) {                       // two coplanar Vogel discs, rims approaching
        float rr  = discR() * sqrt((float(k) + 0.5) / float(NPH));
        float a   = goldenAngle(k);
        float gap = discR() + 1.2;
        float sgn = second ? 1.0 : -1.0;
        o.p = float3(rr * cos(a) + sgn * gap, rr * sin(a), 0.0);
        o.n = float3(0, 0, 1);
        o.v = float3(0.0 - sgn * v, 0, 0);
        o.w = float3(0, 0, 0);
        return o;
    }

    /* Fibonacci shell, directors outward */
    float R = shellR();
    float z = 1.0 - 2.0 * (float(k) + 0.5) / float(NPH);
    float r = sqrt(max(0.0, 1.0 - z * z));
    float a = goldenAngle(k);
    float3 n = float3(r * cos(a), r * sin(a), z);

    float3 c; float3 vv; float3 Om = float3(0, 0, 0);
    if (PRESET == 1) {                       // counter-rotating gears
        float gap = R + 2.0, OmZ = 0.35 / R;
        c  = float3(second ?  gap : 0.0 - gap, 0, 0);
        vv = float3(second ? 0.0 - 0.5 * v : 0.5 * v, 0, 0);
        Om = float3(0, 0, second ? 0.0 - OmZ : OmZ);
    } else {                                 // head-on with vertical offset
        float gap = R + 2.5;
        c  = second ? float3(gap, 0.35 * R, 0) : float3(0.0 - gap, 0, 0);
        vv = float3(second ? 0.0 - v : v, 0, 0);
    }
    float3 rel = n * R;
    o.p = c + rel;
    o.n = n;
    o.v = vv + cross(Om, rel);               // rigid-body surface velocity
    o.w = Om - n * dot(Om, n);               // directors co-rotate; no spin about n
    return o;
}

/* ================= one leapfrog step: R -> W (own index) ================= */
void stepParticle(uint i) {
    if (!Sim.seeded(i)) return;
    Particle a = Sim.read(i);

    float3 pi = a.p;
    float3 ni = normalize(a.n);
    float3 vi = a.v;
    float3 wi = a.w;

    float mu  = presetMu();
    float sn  = sin(presetTheta());
    float gam = presetGamma();

    float3 F  = float3(0, 0, 0);
    float3 Gn = float3(0, 0, 0);             // sum of w * grad_ni(phi); tau = Tf - n x Gn
    float3 Tf = float3(0, 0, 0);
    float rc2 = RCUT * RCUT;

    for (uint j = 0; j < uint(NUM); j++) {
        if (j == i) continue;
        float4 pj = pR[j];
        if (pj.w != SEED_TAG) continue;
        float3 d = pj.xyz - pi;
        float r2 = dot(d, d);
        if (!(r2 < rc2)) continue;           // NaN-safe cutoff

        float r   = sqrt(max(r2, 1e-8));
        float3 rh = d / r;
        float3 nj = Sim.dir(j);

        float c   = dot(ni, nj);
        float aij = dot(ni, rh);
        float aji = dot(nj, rh);
        float A   = c - aij * aji + sn * (aji - aij) - sn * sn;
        float phi = 1.0 + mu * (A - 1.0);

        float Frad, w;
        if (r < 1.0) {
            float rr = max(r, 0.55);         // capped r^-8 core
            float q  = 1.0 / rr;
            float q4 = q * q * q * q;
            Frad = 8.0 * (q4 - q4 * q4) * q;
            w    = -1.0;
        } else {
            float t   = (r - 1.0) / (RCUT - 1.0);
            float ct  = clamp(cos(PI2 * t), 1e-6, 1.0);
            float uA  = 0.0 - pow(ct, 2.0 * ZETA);
            float uAp = 2.0 * ZETA * pow(ct, 2.0 * ZETA - 1.0)
                      * sin(PI2 * t) * PI2 / (RCUT - 1.0);
            Frad = uAp * phi;
            w    = uA;
        }

        /* binary subtraction only -- see codegen warning in header */
        float3 gPhiR = mu * (sn * (nj - ni) - aji * ni - aij * nj);
        float3 tang  = gPhiR - rh * dot(rh, gPhiR);
        F  += Frad * rh + (w / r) * tang;
        Gn += w * mu * (nj - (aji + sn) * rh);

        if (gam > 0.0) {                     // pairwise momentum-conserving friction
            float Wr = 1.0 - r / RCUT;
            float3 slip = (vi - Sim.vel(j)) + cross(0.5 * (wi + Sim.omg(j)), d);
            float fk = gam * Wr;
            float3 Fp = float3(0, 0, 0) - fk * slip;   // zero for rigid drift or spin
            F  += Fp;
            Tf += cross(0.5 * d, Fp);
        }
    }

    float3 tau = Tf - cross(ni, Gn);

    float rad = length(pi);                  // soft distant confinement
    if (rad > wallR()) F -= (1.5 * (rad - wallR())) * (pi / rad);

    /* kick -> project -> drift */
    vi += (DT / MASS) * F;
    wi += (DT / INER) * tau;
    wi -= ni * dot(ni, wi);                  // drop unphysical spin about n

    float sv = length(vi); if (sv > 5.0) vi *= 5.0 / sv;
    float sw = length(wi); if (sw > 8.0) wi *= 8.0 / sw;

    pi += DT * vi;

    float wm = length(wi);
    if (wm > 1e-9) {                         // exact Rodrigues rotation of n
        float3 ax = wi / wm;
        float ang = wm * DT, ca = cos(ang), sa = sin(ang);
        ni = normalize(ni * ca + cross(ax, ni) * sa + ax * dot(ax, ni) * (1.0 - ca));
    }

    a.p = pi; a.n = ni; a.v = vi; a.w = wi;
    Sim.write(i, a);
}

/* ================= camera ================= */
struct Camera {
    float3 s; float3 u; float3 f; float3 eye;
    static Camera get() {
        float4 c = camB[0];
        Camera C;
        C.eye = c.z * float3(cos(c.y) * sin(c.x), sin(c.y), cos(c.y) * cos(c.x));
        C.f   = normalize(float3(0, 0, 0) - C.eye);
        C.s   = normalize(cross(C.f, float3(0, 1, 0)));
        C.u   = cross(C.s, C.f);
        return C;
    }
    float4 project(float3 p) {
        float3 v = p - eye;
        float depth = dot(f, v);
        if (depth < 0.5) return float4(-1, -1, -1, 0);
        float asp = float(RESX) / float(RESY);
        float2 ndc = float2(FOVF / asp * dot(s, v), FOVF * dot(u, v)) / depth;
        float2 pix = float2((ndc.x * 0.5 + 0.5) * float(RESX), (0.5 - 0.5 * ndc.y) * float(RESY));
        float size = clamp(float(RESY) * 0.42 / max(depth, 0.5), 1.5, 26.0);
        return float4(pix, depth, 0.5 * size);
    }
    float3 toView(float3 n) { return float3(dot(s, n), dot(u, n), 0.0 - dot(f, n)); }
};

/* ================= passes ================= */
[WorkgroupCount(1,1,1)]
[shader("compute")]
[numthreads(1,1,1)]
void a_camera(uint3 tid : SV_DispatchThreadID) {
    float4 c0 = camB[0], c1 = camB[1];
    float2 m = float2(mouse.pos);
    if (c0.z < 1.0 || time.frame < 2u) {
        c0 = float4(0.5, 0.32, camDist(), 0.0); c1 = float4(m, 0.0, 0.0);
    }
    bool down = mouse.click != 0;
    if (down && c1.z > 0.5) {
        c0.x += (m.x - c1.x) * 0.006;
        c0.y  = clamp(c0.y + (m.y - c1.y) * 0.006, -1.5, 1.5);
    } else if (!down) {
        c0.x += 0.0015;                      // idle auto-orbit
    }
    c0.x = c0.x - TAU2 * floor(c0.x / TAU2); // keep trig argument in domain
    camB[0] = c0;
    camB[1] = float4(m, down ? 1.0 : 0.0, 0.0);
}

[WorkgroupCount(PWG_N,1,1)]
[shader("compute")]
[numthreads(PWG,1,1)]
void b_seed(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= uint(NUM)) return;
    if (time.frame < 2u)  { Sim.seed(i, initParticle(i)); return; }  // rewind = clean start
    if (!Sim.seeded(i))   { Sim.seed(i, initParticle(i)); return; }  // fresh/stale storage
    if (!Sim.read(i).valid()) Sim.seed(i, initParticle(i));          // self-heal
}

/* multistep: even sub-dispatches step (R -> W), odd ones sync (W -> R) */
[DispatchCount(SUBSTEPS * 2)]
[WorkgroupCount(PWG_N,1,1)]
[shader("compute")]
[numthreads(PWG,1,1)]
void c_sim(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= uint(NUM)) return;
    if ((dispatch.id & 1u) == 0u) stepParticle(i);
    else                          Sim.sync(i);
}

/* ================= rasterization: clear -> splat -> resolve ================= */
[WorkgroupCount(CLR_WG,1,1)]
[shader("compute")]
[numthreads(256,1,1)]
void d_clear(uint3 tid : SV_DispatchThreadID) {
    splatB[tid.x].store(0xFFFFFFFFu);
}

[WorkgroupCount(PWG_N,1,1)]
[shader("compute")]
[numthreads(PWG,1,1)]
void e_splat(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= uint(NUM)) return;
    if (!Sim.seeded(i)) return;

    Camera C = Camera.get();
    float4 pr = C.project(Sim.pos(i));
    if (pr.z <= 0.0) return;

    int r = int(ceil(pr.w));
    int2 cpix = int2(floor(pr.xy));
    float dmax = 4.0 * camDist();

    for (int dy = -r; dy <= r; dy++)
    for (int dx = -r; dx <= r; dx++) {
        int2 q = cpix + int2(dx, dy);
        if (q.x < 0 || q.y < 0 || q.x >= RESX || q.y >= RESY) continue;
        float2 c2 = (float2(q) + 0.5 - pr.xy) / pr.w;
        float d2 = dot(c2, c2);
        if (d2 > 1.0) continue;
        float depth = pr.z - 0.5 * sqrt(1.0 - d2);   // sphere bulge for z-order
        uint d16 = uint(clamp(depth / dmax, 0.0, 1.0) * 65535.0);
        splatB[q.x + q.y * RESX].min((d16 << 16) | (i & 0xFFFFu));
    }
}

[shader("compute")]
[numthreads(16,16,1)]
void main_image(uint3 tid : SV_DispatchThreadID) {
    uint W, H; screen.GetDimensions(W, H);
    float sc = min(float(W) / float(RESX), float(H) / float(RESY));
    float2 offs = 0.5 * (float2(W, H) - sc * float2(RESX, RESY));
    float2 pix = float2(tid.xy);

    if (any(pix < offs) || any(pix >= offs + sc * float2(RESX, RESY))) {
        screen[tid.xy] = float4(0.010, 0.013, 0.020, 1.0);
        return;
    }
    float2 sp = (pix - offs) / sc;
    int2 q = int2(clamp(sp, float2(0, 0), float2(RESX - 1, RESY - 1)));

    /* background vignette */
    float2 uv = sp / float2(RESX, RESY) - 0.5;
    float vgn = 1.0 - 0.85 * dot(uv, uv) * 2.2;
    float3 col = lerp(float3(0.016, 0.020, 0.030),
                      float3(0.043, 0.055, 0.082), clamp(vgn, 0.0, 1.0));

    uint key = splatB[q.x + q.y * RESX].load();
    if (key != 0xFFFFFFFFu) {
        uint i = key & 0xFFFFu;
        Camera C = Camera.get();

        float3 nw = normalize(Sim.dir(i));
        float4 pr = C.project(Sim.pos(i));
        float2 c2 = (sp - pr.xy) / max(pr.w, 1e-3);
        float d2 = min(dot(c2, c2), 1.0);

        float3 nv = C.toView(nw);
        float3 base = lerp(float3(0.42, 0.55, 0.72), 0.5 + 0.5 * nw, 0.85);
        float3 Ldir = normalize(float3(0.35, 0.65, 0.67));
        float lam = abs(dot(nv, Ldir));
        float sph = sqrt(max(1.0 - d2, 0.0));
        col = base * (0.28 + 0.62 * lam) * (0.55 + 0.45 * sph)
            + float3(0.10, 0.10, 0.10) * pow(lam, 12.0);
    }
    screen[tid.xy] = float4(col, 1.0);
}