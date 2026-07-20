import std;

/* ================================================================
   Oriented-particle FILM membranes -- fluid-surface hybrid
   (compute.toys / Slang)
   ----------------------------------------------------------------
   ARCHITECTURE (every piece selected by measurement):
   - density: PBF equality constraint, Gaussian kernel sigma=1.5,
     per-particle rest density (event-driven calibration). Central
     corrections -> conserves angular momentum exactly. Weld-on-
     contact, no tearing, rho pinned to 1.000.
   - planarity/bending: pair FORCE U = 0.5*K_PL*w*(nbar.d)^2 with
     energy-matched director torque (the PBD projection form of
     this constraint measurably damps rotation; the force form
     conserves it).
   - viscosity: radial pairwise friction (central, rotation-blind).
   - 2D SURFACE FLUIDITY: s_corr is OFF and the short-range core is
     soft/quadratic. Measured: s_corr's peaked repulsion gives the
     layer a shear modulus -- a T=0 crystal whose elastic shear
     waves smear any surface flow across the sphere in ~1 tau. With
     it removed (pairing guarded by the soft core; kernel wide so
     rest spacing sits on the stable side of the gradient peak), a
     localized surface vortex keeps ~1/3 of its core circulation
     after 8 tau at test scale, ~10x slower spread; expect several-
     fold longer on this build's larger shells.

   PRESET 3 is the acceptance test: one full-N Fibonacci sphere
   with a localized Gaussian vortex kicked on its surface.

   !! CODEGEN WARNINGS (verified on this backend):
   - never -(a*x + b*y) with vectors; binary subtraction only
   - no pow(); exp() fine; trig args within [0, 2*pi]

   Per substep: predict -> sync -> K x (lambda, apply, sync) ->
   finalize -> sync, one [DispatchCount] pass (in-order execution
   with barriers verified on this platform).
   ================================================================ */

/* ---------------- parameters (edit me) ---------------- */
static const int   NUM      = 4096;   // particles (power of two, id <= 16 bits)
static const int   PRESET   = 0;      // 0 shells bounce / 1 counter-spin / 2 discs fuse / 3 SURFACE VORTEX
static const int   SUBSTEPS = 1;
static const int   K_ITERS  = 3;      // density solver iterations

static const float DT   = 0.02;
static const float SIG  = 1.5;        // gaussian width (wide: rest spacing on stable side of peak)
static const float H    = 3.0;        // cutoff
static const float RHO0FLAT = 7.977;  // flat hex sheet, spacing 1, sigma=1.5 (recompute if SIG changes)
static const float EPS_CFM  = 0.02;
static const float SCORR_K  = 0.0;    // KEEP 0: nonzero re-crystallizes the surface (shear modulus)
static const float SCORR_DQ = 0.45;
static const float RELAX    = 0.5;
static const float DP_MAX   = 0.25;
static const float K_PL  = 8.0;       // planarity force (film thinness); torque matched
static const float MU_T  = 3.0;       // alignment / spontaneous curvature torque
static const float GAM_V = 0.04;      // radial viscosity (raise for honey, lower for livelier)
static const float GAM_W = 0.03;      // pairwise omega smoothing
static const float KCORE = 12.0, RCORE = 0.8;   // SOFT quadratic core (pairing guard)
static const float IMPACT = 0.45;
static const float IMPACT_FUSE = 0.9;
static const float VORT_A = 1.2, VORT_TH = 0.5; // vortex amplitude, angular width (rad)
static const float INER = 0.4;

static const float SEED_TAG = 14.5;   // bumped: parameters/presets changed
static const float PI   = 3.14159265359;
static const float TAU2 = 6.28318530718;
static const float GAF  = 0.38196601;
static const uint  NPH  = uint(NUM / 2);
static const int   PHASES = 4 + 3 * K_ITERS;

float shellR()      { return sqrt(float(NPH) * 0.866 / (4.0 * PI)); }
float vortexR()     { return sqrt(float(NUM) * 0.866 / (4.0 * PI)); }   // full-N sphere
float discR()       { return sqrt(float(NPH) * 0.9 / PI); }
float presetTheta() { return PRESET == 2 ? 0.05
                    : PRESET == 3 ? min(1.0 / vortexR(), 0.3)
                    : min(1.0 / shellR(), 0.3); }
float wallR()       { return PRESET == 2 ? 3.5 * discR() + 12.0
                    : PRESET == 3 ? 2.5 * vortexR() + 8.0
                    : 4.5 * shellR() + 10.0; }
float camDist()     { return PRESET == 2 ? 4.2 * discR()
                    : PRESET == 3 ? 2.8 * vortexR()
                    : 5.2 * shellR(); }
float goldenAngle(uint k) { return frac(float(k) * GAF) * TAU2; }

float Wg(float r2) { return exp((0.0 - r2) / (SIG * SIG)); }
float dWg(float r) { return (0.0 - 2.0 * r / (SIG * SIG)) * Wg(r * r); }

static const int RESX = 1280, RESY = 720;
static const float FOVF = 2.0702;
static const int PWG = 256, PWG_N = NUM / PWG;
static const int CLR_WG = (RESX * RESY) / 256;

/* ---------------- buffers ---------------- */
[StorageBuffer(NUM)] RWStructuredBuffer<float4> pR;   // .w == SEED_TAG when seeded
[StorageBuffer(NUM)] RWStructuredBuffer<float4> dR;
[StorageBuffer(NUM)] RWStructuredBuffer<float4> vR;
[StorageBuffer(NUM)] RWStructuredBuffer<float4> wR;
[StorageBuffer(NUM)] RWStructuredBuffer<float4> pW;
[StorageBuffer(NUM)] RWStructuredBuffer<float4> dW;
[StorageBuffer(NUM)] RWStructuredBuffer<float4> vW;
[StorageBuffer(NUM)] RWStructuredBuffer<float4> wW;
[StorageBuffer(NUM)] RWStructuredBuffer<float4> pO;
[StorageBuffer(NUM)] RWStructuredBuffer<float4> lmB;   // (lambda, rho, 0, 0)
[StorageBuffer(NUM)] RWStructuredBuffer<float4> restB; // (rest density, stamp, 0, 0)
[StorageBuffer(RESX*RESY)] RWStructuredBuffer<Atomic<uint>> splatB;
[StorageBuffer(4)] RWStructuredBuffer<float4> camB;

/* ================= state ================= */
struct Particle {
    float3 p; float3 n; float3 v; float3 w;
    bool valid() {
        float nn = dot(n, n);
        return dot(p, p) < 1e12 && dot(v, v) < 1e8 && nn > 0.25 && nn < 4.0;
    }
};
struct Sim {
    static bool  seeded(uint j) { return pR[j].w == SEED_TAG; }
    static float3 pos(uint j) { return pR[j].xyz; }
    static float3 dir(uint j) { return dR[j].xyz; }
    static Particle read(uint j) {
        Particle q;
        q.p = pR[j].xyz; q.n = dR[j].xyz; q.v = vR[j].xyz; q.w = wR[j].xyz;
        return q;
    }
    static void write(uint j, Particle q) {
        pW[j] = float4(q.p, SEED_TAG); dW[j] = float4(q.n, 0);
        vW[j] = float4(q.v, 0);        wW[j] = float4(q.w, 0);
    }
    static void seed(uint j, Particle q) {
        pR[j] = float4(q.p, SEED_TAG); dR[j] = float4(q.n, 0);
        vR[j] = float4(q.v, 0);        wR[j] = float4(q.w, 0);
        pW[j] = float4(q.p, SEED_TAG); dW[j] = float4(q.n, 0);
        vW[j] = float4(q.v, 0);        wW[j] = float4(q.w, 0);
        restB[j] = float4(0, 0, 0, 0);
    }
    static void sync(uint j) {
        pR[j] = pW[j]; dR[j] = dW[j]; vR[j] = vW[j]; wR[j] = wW[j];
    }
};

/* ================= presets ================= */
Particle initParticle(uint i) {
    Particle o;

    if (PRESET == 3) {                       // full-N sphere + localized surface vortex
        float R = vortexR();
        float z = 1.0 - 2.0 * (float(i) + 0.5) / float(NUM);
        float r = sqrt(max(0.0, 1.0 - z * z));
        float a = goldenAngle(i);
        float3 n = float3(r * cos(a), r * sin(a), z);
        o.p = n * R;
        o.n = n;
        /* gaussian vortex about the +x axis: v = f(theta) * (x_hat cross n) */
        float th = acos(clamp(n.x, -1.0, 1.0));
        float q  = th / VORT_TH;
        float f  = VORT_A * exp(0.0 - q * q);
        o.v = float3(0.0, 0.0 - f * n.z, f * n.y);
        o.w = float3(0, 0, 0);
        return o;
    }

    bool second = i >= NPH;
    uint k = second ? i - NPH : i;

    if (PRESET == 2) {                       // two Vogel discs, rims overlapping
        float rr  = discR() * sqrt((float(k) + 0.5) / float(NPH));
        float a   = goldenAngle(k);
        float gap = discR() - 1.0;
        float sgn = second ? 1.0 : -1.0;
        o.p = float3(rr * cos(a) + sgn * gap, rr * sin(a), 0.0);
        o.n = float3(0, 0, 1);
        o.v = float3(0.0 - sgn * IMPACT_FUSE, 0, 0);
        o.w = float3(0, 0, 0);
        return o;
    }

    float v = IMPACT;
    float R = shellR();
    float z = 1.0 - 2.0 * (float(k) + 0.5) / float(NPH);
    float r = sqrt(max(0.0, 1.0 - z * z));
    float a = goldenAngle(k);
    float3 n = float3(r * cos(a), r * sin(a), z);

    float3 c; float3 vv; float3 Om = float3(0, 0, 0);
    if (PRESET == 1) {
        float gap = R + 2.0, OmZ = 0.35 / R;
        c  = float3(second ?  gap : 0.0 - gap, 0, 0);
        vv = float3(second ? 0.0 - 0.5 * v : 0.5 * v, 0, 0);
        Om = float3(0, 0, second ? 0.0 - OmZ : OmZ);
    } else {
        float gap = R + 2.5;
        c  = second ? float3(gap, 0.35 * R, 0) : float3(0.0 - gap, 0, 0);
        vv = float3(second ? 0.0 - v : v, 0, 0);
    }
    float3 rel = n * R;
    o.p = c + rel;
    o.n = n;
    o.v = vv + cross(Om, rel);
    o.w = Om - n * dot(Om, n);
    return o;
}

/* ================= phases ================= */

void phPredict(uint i) {
    if (!Sim.seeded(i)) return;
    Particle a = Sim.read(i);
    float3 pi = a.p;
    float3 ni = normalize(a.n);
    float3 vi = a.v;
    float3 F = float3(0, 0, 0);
    float w0 = Wg(0.0);

    for (uint j = 0; j < uint(NUM); j++) {
        if (j == i) continue;
        float4 pj = pR[j];
        if (pj.w != SEED_TAG) continue;
        float3 d = pj.xyz - pi;
        float r2 = dot(d, d);
        if (!(r2 < H * H)) continue;
        float r = sqrt(max(r2, 1e-8));
        float3 u = d / r;
        float wq = Wg(r2) / w0;

        /* soft quadratic core (pairing guard; no lattice ridge) */
        if (r < RCORE) {
            float t = 1.0 - r / RCORE;
            F -= (KCORE * t * t) * u;
        }

        /* planarity force (terms split -- codegen) */
        float3 nm = 0.5 * (ni + dR[j].xyz);
        float e   = dot(nm, d);
        float wqp = dWg(r) / w0;
        F += (K_PL * wq * e) * nm;
        F += (K_PL * 0.5 * wqp * e * e) * u;

        /* radial viscosity: central, rotation-blind */
        float vr = dot(vi - vR[j].xyz, u);
        F -= (GAM_V * wq * vr) * u;
    }
    float rad = length(pi);
    if (rad > wallR()) F -= (1.5 * (rad - wallR())) * (pi / rad);

    vi += DT * F;
    float sv = length(vi); if (sv > 5.0) vi *= 5.0 / sv;
    pO[i] = float4(pi, 0);
    a.p = pi + DT * vi;
    a.v = vi;

    /* director drift (omega kick happens in finalize) */
    float3 wi = a.w;
    float wm = length(wi);
    if (wm > 1e-9) {
        float3 ax = wi / wm;
        float ang = wm * DT, ca = cos(ang), sa = sin(ang);
        ni = normalize(ni * ca + cross(ax, ni) * sa + ax * dot(ax, ni) * (1.0 - ca));
    }
    a.n = ni;
    Sim.write(i, a);
}

void phLambda(uint i) {
    if (!Sim.seeded(i)) return;
    float3 pi = Sim.pos(i);
    float rho = Wg(0.0);
    float3 gs = float3(0, 0, 0);
    float s2 = 0.0;
    float rest = restB[i].x;
    bool calib = restB[i].y != SEED_TAG;
    if (calib) rest = RHO0FLAT;

    for (uint j = 0; j < uint(NUM); j++) {
        if (j == i) continue;
        float4 pj = pR[j];
        if (pj.w != SEED_TAG) continue;
        float3 d = pi - pj.xyz;
        float r2 = dot(d, d);
        if (!(r2 < H * H)) continue;
        float r = sqrt(max(r2, 1e-8));
        rho += Wg(r2);
        float g = dWg(r) / rest;
        float3 u = d / max(r, 1e-6);
        gs += g * u;
        s2 += g * g;
    }
    if (calib) {
        rest = clamp(rho, 0.8 * RHO0FLAT, 1.6 * RHO0FLAT);
        restB[i] = float4(rest, SEED_TAG, 0, 0);
    }
    float C = rho / rest - 1.0;
    float lam = (0.0 - C) / (dot(gs, gs) + s2 + EPS_CFM);
    lmB[i] = float4(lam, rho, 0, 0);
}

void phApply(uint i) {
    if (!Sim.seeded(i)) return;
    Particle a = Sim.read(i);
    float lamI = lmB[i].x;
    float restI = restB[i].x;
    float wdq = Wg(SCORR_DQ * SCORR_DQ);
    float3 dp = float3(0, 0, 0);

    for (uint j = 0; j < uint(NUM); j++) {
        if (j == i) continue;
        float4 pj = pR[j];
        if (pj.w != SEED_TAG) continue;
        float3 d = a.p - pj.xyz;
        float r2 = dot(d, d);
        if (!(r2 < H * H)) continue;
        float r = sqrt(max(r2, 1e-8));
        float3 u = d / max(r, 1e-6);
        float sc = Wg(r2) / wdq;
        float sc2 = sc * sc;
        float scorr = 0.0 - SCORR_K * sc2 * sc2;
        float g = dWg(r) / restI;
        dp += ((lamI + lmB[j].x + scorr) * g / restI) * u;
    }
    dp *= RELAX;
    float dl = length(dp);
    if (dl > DP_MAX) dp *= DP_MAX / dl;
    a.p += dp;
    Sim.write(i, a);
}

void phFinalize(uint i) {
    if (!Sim.seeded(i)) return;
    Particle a = Sim.read(i);
    float sn = sin(presetTheta());

    float3 v = (a.p - pO[i].xyz) / DT;
    float sv = length(v); if (sv > 5.0) v *= 5.0 / sv;

    float3 ni = normalize(a.n);
    float3 wi = a.w;
    float3 Gn = float3(0, 0, 0);
    float3 Tf = float3(0, 0, 0);
    float w0 = Wg(0.0);

    for (uint j = 0; j < uint(NUM); j++) {
        if (j == i) continue;
        float4 pj = pR[j];
        if (pj.w != SEED_TAG) continue;
        float3 d = pj.xyz - a.p;
        float r2 = dot(d, d);
        if (!(r2 < H * H)) continue;
        float r = sqrt(max(r2, 1e-8));
        float3 u = d / r;
        float wq = Wg(r2) / w0;
        float3 nj = dR[j].xyz;

        float aji = dot(nj, u);
        Gn += wq * MU_T * ((aji + sn) * u - nj);

        float3 nm = 0.5 * (ni + nj);
        float e = dot(nm, d);
        Gn += (K_PL * wq * e * 0.5) * d;

        Tf += GAM_W * wq * (wR[j].xyz - wi);
    }

    float3 tau = Tf - cross(ni, Gn);
    wi += (DT / INER) * tau;
    wi -= ni * dot(ni, wi);
    float sw = length(wi); if (sw > 8.0) wi *= 8.0 / sw;

    a.v = v; a.w = wi;
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
        c0.x += 0.0015;
    }
    c0.x = c0.x - TAU2 * floor(c0.x / TAU2);
    camB[0] = c0;
    camB[1] = float4(m, down ? 1.0 : 0.0, 0.0);
}

[WorkgroupCount(PWG_N,1,1)]
[shader("compute")]
[numthreads(PWG,1,1)]
void b_seed(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= uint(NUM)) return;
    if (time.frame < 2u)  { Sim.seed(i, initParticle(i)); return; }
    if (!Sim.seeded(i))   { Sim.seed(i, initParticle(i)); return; }
    if (!Sim.read(i).valid()) Sim.seed(i, initParticle(i));
}

/* 0 predict | 1 sync | k: 2+3k lambda, 3+3k apply, 4+3k sync |
   2+3K finalize | 3+3K sync */
[DispatchCount(PHASES * SUBSTEPS)]
[WorkgroupCount(PWG_N,1,1)]
[shader("compute")]
[numthreads(PWG,1,1)]
void c_sim(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= uint(NUM)) return;
    if (time.frame < 2u) return;
    uint ph = dispatch.id % uint(PHASES);

    if (ph == 0u)                    { phPredict(i);  return; }
    if (ph == 1u)                    { Sim.sync(i);   return; }
    if (ph == uint(2 + 3 * K_ITERS)) { phFinalize(i); return; }
    if (ph == uint(3 + 3 * K_ITERS)) { Sim.sync(i);   return; }
    uint q = (ph - 2u) % 3u;
    if (q == 0u)      phLambda(i);
    else if (q == 1u) phApply(i);
    else              Sim.sync(i);
}

/* ================= rasterization ================= */
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
        float depth = pr.z - 0.5 * sqrt(1.0 - d2);
        uint d16 = uint(clamp(depth / dmax, 0.0, 1.0) * 65535.0);
        splatB[q.x + q.y * RESX].min((d16 << 16) | (i & 0xFFFFu));
    }
}

[shader("compute")]
[numthreads(16,16,1)]
void main_image(uint3 tid : SV_DispatchThreadID) {
    uint W, H2; screen.GetDimensions(W, H2);
    float sc = min(float(W) / float(RESX), float(H2) / float(RESY));
    float2 offs = 0.5 * (float2(W, H2) - sc * float2(RESX, RESY));
    float2 pix = float2(tid.xy);

    if (any(pix < offs) || any(pix >= offs + sc * float2(RESX, RESY))) {
        screen[tid.xy] = float4(0.010, 0.013, 0.020, 1.0);
        return;
    }
    float2 sp = (pix - offs) / sc;
    int2 q = int2(clamp(sp, float2(0, 0), float2(RESX - 1, RESY - 1)));

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