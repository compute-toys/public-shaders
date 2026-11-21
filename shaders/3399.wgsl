#storage S array<vec4f, 4227072>
// layout (vec4f units): [0, 2097152) field A, [2097152, 4194304) field B, [4194304, ...) ants

#workgroup_count Step 32 1 1
#workgroup_count Diffuse 64 64 1
#workgroup_count Copy 64 64 1

const NU: u32 = 1024u;
const NI: i32 = 1024;
const N: f32 = 1024.0;
const U: f32 = 1.0;            // N / 256, scales spatial constants
const G2: u32 = 2097152u;      // 2 * NU * NU
const AB: u32 = 4194304u;      // 4 * NU * NU
const NANTS: u32 = 8192*4;
const STEPS: i32 = 256;
const LEN: f32 = 3.618;          // ant step length in cells
const SENSE: f32 = 37.0;
const REACH: f32 = 32.0 * U * 0.75;

const A_SELF: f32 = 11.28;
const A_REL: f32 = 17.14;
const DEP: f32 = 0.016;
const ROT: f32 = .01;
const ABSB: f32 = 1.0;
const DIFF: f32 = 0.9;
const DECAY: f32 = 0.1;
const GAIN: f32 = 6.0;
const GLOW: f32 = 1.0;
const R3: f32 = 0.57735027;

struct Ch { lo: vec4f, hi: vec4f }

fn cmul(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
fn ccj(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}

fn charge(a: vec2f, b: vec2f, c: vec2f, s: f32) -> Ch {
    let zab = ccj(a, b);
    let zac = ccj(a, c);
    let zbc = ccj(b, c);
    let aa = dot(a, a);
    let bb = dot(b, b);
    let cc = dot(c, c);
    var o: Ch;
    o.lo = s * vec4f(zab.x, zab.y, 0.5 * (aa - bb), zac.x);
    o.hi = s * vec4f(zac.y, zbc.x, zbc.y, 0.5 * (aa + bb - 2.0 * cc) * R3);
    return o;
}

fn energy(q: Ch, f: Ch) -> f32 {
    return dot(q.lo, f.lo) + dot(q.hi, f.hi);
}

fn cellidx(p: vec2f) -> u32 {
    let q = p - floor(p / N) * N;
    return min(u32(q.y), NU - 1u) * NU + min(u32(q.x), NU - 1u);
}

fn field(i: u32) -> Ch {
    var o: Ch;
    o.lo = S[2u * i];
    o.hi = S[2u * i + 1u];
    return o;
}

@compute @workgroup_size(256)
fn Step(@builtin(global_invocation_id) id: vec3u) {
    let n = id.x;
    if (n >= NANTS) { return; }

    var st = S[AB + 4u * n];
    var a = S[AB + 4u * n + 1u].xy;
    var b = S[AB + 4u * n + 1u].zw;
    var c = S[AB + 4u * n + 2u].xy;
    var s = S[AB + 4u * n + 2u].z;

    if (st.w == 0.0) {
        let g = n / 3u;
        let ang = f32(g) * 2.399963;
        let rad = 45.0 * U * sqrt(f32(g) / f32(NANTS / 3u));
        let jit = 2.0 * U * vec2f(cos(f32(n) * 1.7), sin(f32(n) * 1.7));
        let start = vec2f(N * 0.5, N * 0.5) + rad * vec2f(cos(ang), sin(ang)) + jit;
        st = vec4f(start.x, start.y, f32(n) * 0.7, 1.0);
        let k0 = n % 3u;
        a = vec2f(0.0, 0.0);
        b = vec2f(0.0, 0.0);
        c = vec2f(0.0, 0.0);
        if (k0 == 0u) { a.x = 1.0; }
        if (k0 == 1u) { b.x = 1.0; }
        if (k0 == 2u) { c.x = 1.0; }
        let fi = f32(n);
        a = a + 0.3 * vec2f(cos(fi * 1.3), sin(fi * 1.3));
        b = b + 0.3 * vec2f(cos(fi * 1.3 + 2.1), sin(fi * 1.3 + 2.1));
        c = c + 0.3 * vec2f(cos(fi * 1.3 + 4.2), sin(fi * 1.3 + 4.2));
        let nr0 = sqrt(dot(a, a) + dot(b, b) + dot(c, c));
        a = a / nr0;
        b = b / nr0;
        c = c / nr0;
        s = 1.0;
        if (g % 2u == 1u) { s = -1.0; }
    }

    var p = st.xy;
    var th = st.z;

    for (var i: i32 = 0; i < STEPS; i = i + 1) {
        let ci = cellidx(p);
        let fc = field(ci);
        let fl = field(cellidx(p + REACH * vec2f(cos(th + SENSE), sin(th + SENSE))));
        let fr = field(cellidx(p + REACH * vec2f(cos(th - SENSE), sin(th - SENSE))));

        let qb = charge(a, b, c, s);
        th = th + A_SELF * energy(qb, fc) + A_REL * (energy(qb, fr) - energy(qb, fl));

        let p1 = fc.lo.x;
        let p2 = fc.lo.y;
        let p3 = fc.lo.z;
        let p4 = fc.lo.w;
        let p5 = fc.hi.x;
        let p6 = fc.hi.y;
        let p7 = fc.hi.z;
        let p8 = fc.hi.w;
        let m00 = 0.5 * (p3 + p8 * R3);
        let m11 = 0.5 * (-p3 + p8 * R3);
        let m22 = -p8 * R3;
        let m01 = 0.5 * vec2f(p1, -p2);
        let m10 = 0.5 * vec2f(p1, p2);
        let m02 = 0.5 * vec2f(p4, -p5);
        let m20 = 0.5 * vec2f(p4, p5);
        let m12 = 0.5 * vec2f(p6, -p7);
        let m21 = 0.5 * vec2f(p6, p7);
        let ma = m00 * a + cmul(m01, b) + cmul(m02, c);
        let mb = cmul(m10, a) + m11 * b + cmul(m12, c);
        let mc = cmul(m20, a) + cmul(m21, b) + m22 * c;
        let kr = ROT * s;
        a = a + kr * vec2f(ma.y, -ma.x);
        b = b + kr * vec2f(mb.y, -mb.x);
        c = c + kr * vec2f(mc.y, -mc.x);
        let nrm = sqrt(dot(a, a) + dot(b, b) + dot(c, c));
        a = a / nrm;
        b = b / nrm;
        c = c / nrm;

        let qa = charge(a, b, c, s);
        let dlo = DEP * qb.lo + ABSB * (qb.lo - qa.lo);
        let dhi = DEP * qb.hi + ABSB * (qb.hi - qa.hi);
        S[2u * ci] = clamp(fc.lo + dlo, vec4f(-2.0, -2.0, -2.0, -2.0), vec4f(2.0, 2.0, 2.0, 2.0));
        S[2u * ci + 1u] = clamp(fc.hi + dhi, vec4f(-2.0, -2.0, -2.0, -2.0), vec4f(2.0, 2.0, 2.0, 2.0));

        p = p + LEN * vec2f(cos(th), sin(th));
        p = p - floor(p / N) * N;
    }

    S[AB + 4u * n] = vec4f(p.x, p.y, th, 1.0);
    S[AB + 4u * n + 1u] = vec4f(a.x, a.y, b.x, b.y);
    S[AB + 4u * n + 2u] = vec4f(c.x, c.y, s, 0.0);
}

fn wrapi(x: i32, y: i32) -> u32 {
    return u32(y & (NI - 1)) * NU + u32(x & (NI - 1));
}

@compute @workgroup_size(16, 16)
fn Diffuse(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= NU || id.y >= NU) { return; }
    let x = i32(id.x);
    let y = i32(id.y);
    let i = id.y * NU + id.x;
    let n0 = wrapi(x + 1, y);
    let n1 = wrapi(x - 1, y);
    let n2 = wrapi(x, y + 1);
    let n3 = wrapi(x, y - 1);
    let alo = 0.25 * (S[2u * n0] + S[2u * n1] + S[2u * n2] + S[2u * n3]);
    let ahi = 0.25 * (S[2u * n0 + 1u] + S[2u * n1 + 1u] + S[2u * n2 + 1u] + S[2u * n3 + 1u]);
    S[G2 + 2u * i] = mix(S[2u * i], alo, DIFF) * (1.0 - DECAY);
    S[G2 + 2u * i + 1u] = mix(S[2u * i + 1u], ahi, DIFF) * (1.0 - DECAY);
}

@compute @workgroup_size(16, 16)
fn Copy(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= NU || id.y >= NU) { return; }
    let i = id.y * NU + id.x;
    S[2u * i] = S[G2 + 2u * i];
    S[2u * i + 1u] = S[G2 + 2u * i + 1u];
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let dim = textureDimensions(screen);
    if (id.x >= dim.x || id.y >= dim.y) { return; }

    // bilinear sample of the field at this pixel
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dim) * N - 0.5;
    let f = floor(uv);
    let t = uv - f;
    let x0 = i32(f.x);
    let y0 = i32(f.y);
    let i00 = wrapi(x0, y0);
    let i10 = wrapi(x0 + 1, y0);
    let i01 = wrapi(x0, y0 + 1);
    let i11 = wrapi(x0 + 1, y0 + 1);
    let lo = mix(mix(S[2u * i00], S[2u * i10], t.x), mix(S[2u * i01], S[2u * i11], t.x), t.y);
    let hi = mix(mix(S[2u * i00 + 1u], S[2u * i10 + 1u], t.x), mix(S[2u * i01 + 1u], S[2u * i11 + 1u], t.x), t.y);

    let diag = vec3f(0.5 * (lo.z + hi.w * R3), 0.5 * (-lo.z + hi.w * R3), -hi.w * R3);
    let off = sqrt(lo.x * lo.x + lo.y * lo.y + lo.w * lo.w + hi.x * hi.x + hi.y * hi.y + hi.z * hi.z);
    let col = max(diag, vec3f(0.0, 0.0, 0.0)) * GAIN + vec3f(off, off, off) * GLOW;
    textureStore(screen, id.xy, vec4f(col.x, col.y, col.z, 1.0));
}