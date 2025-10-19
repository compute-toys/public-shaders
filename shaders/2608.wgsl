// KuKo Day 175 — WGSL compute (compute.toys style)
// Renders a glowing “flame column” formed by a pumpkin SDF difference.
// Uses only `screen` and `time.elapsed` like the example you shared.
// Chatgpt struggling to translate my shadertoy code to compute.toys
// https://www.shadertoy.com/view/w32yzh

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coords (origin at center, +y up)
    let frag = vec2f(f32(id.x), f32(screen_size.y) - f32(id.y)) - 0.5 * vec2f(screen_size);
    let uv   = frag / f32(screen_size.y);

    // Camera setup similar to your Shadertoy code
    let ta = vec3f(0.0, 4.0, 0.0);
    let ro = ta + vec3f(5.0 * cos(-1.7), 2.0, 5.0 * sin(-1.5));
    let rd = build_ray(uv, ro, ta, 1.2); // FOV ~69°

    // March scene (flame glow only, like your original)
    let m  = ray_march(ro, rd);

    // Gentle compression so it doesn’t blow out
    let col = tanh_vec(m.glow * 0.5);

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.0));
}

// -------------------- math helpers --------------------

fn S(a: f32, b: f32, c: f32) -> f32 { return smoothstep(a, b, c); }

fn R2(a: f32) -> mat2x2<f32> {
    let s = sin(a); let c = cos(a);
    return mat2x2<f32>(c, -s, s, c);
}

// “soft max” as in your GLSL
fn smax(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0);
    return max(a, b) + 0.25 * h * h / k;
}

fn sd_ellipsoid(p: vec3f, r: vec3f) -> f32 {
    let k1 = length(p / r);
    return (k1 - 1.0) * min(r.x, min(r.y, r.z));
}

fn wrap_center(x: f32, period: f32, half_span: f32) -> f32 {
    return (x - period * floor(x / period)) - half_span;
}

// small fbm-ish like your original
fn fbm(p0: vec3f) -> f32 {
    var p   = p0;
    var amp = 1.0;
    var fre = 1.3;
    var n   = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        n   = n + abs(dot(cos(p * fre), vec3f(0.1, 0.52, 0.3))) * amp;
        amp = amp * 0.59;
        fre = fre * 1.3;
        let rot = R2(p.y * 0.51 + time.elapsed * 0.1);
        let rxy = rot * p.xz;
        // no swizzle writes:
        p = vec3f(rxy.x, p.y - time.elapsed, rxy.y);
    }
    return n;
}

fn tanh_vec(x: vec3f) -> vec3f {
    let e = exp(2.0 * x);
    return (e - vec3f(1.0)) / (e + vec3f(1.0));
}

// -------------------- pumpkin SDF --------------------

const STEM: f32 = 5.0;

fn sdf_pumpkin(pin: vec3f) -> vec2f {
    let scale = 2.0;
    let pos   = pin * scale;

    let proxy = length(pos - vec3f(0.0, 0.2, 0.0));
    if (proxy > 4.0) { return vec2f(proxy - 3.0, 0.0); }

    let angle   = atan2(pos.x, pos.z);
    let section = smax(0.05, abs(sin(angle * 4.0)), 0.05) * 0.1;
    let longlen = length(pos.xz);
    let pinch   = S(1.4, -0.2, longlen);

    var pumpkin = sd_ellipsoid(pos, vec3f(1.7, 1.5, 1.7)) + pinch * 0.6;

    let displ = ((sin(angle * 25.0) + sin(angle * 43.0)) * 0.0015 - section) * S(0.2, 1.3, longlen);
    pumpkin = pumpkin + displ;

    var stem = longlen - 0.29 + S(1.1, 1.5, pos.y) * 0.15 + sin(angle * 4.0) * 0.01;
    let sdisp = sin(angle * 10.0);
    stem = stem + sdisp * 0.005;
    stem = stem - (pos.y - 0.2) * 0.1;
    stem = stem * 0.8;

    let stem_cut = pos.y - 2.0 + pos.x * 0.3;
    stem = smax(stem, stem_cut, 0.05);
    stem = max(stem, 1.0 - pos.y);

    let pumpkin_id = clamp(displ * 4.0 + 0.5, 0.0, 0.999);
    let stem_id    = STEM + (0.5 + sdisp * 0.2) * S(0.1, -0.6, stem_cut);

    pumpkin = abs(pumpkin) - 0.05;

    // face carving
    var face = length(pos.xy - vec2f(0.0, 0.3)) - 1.1;
    face = max(face, -(length(pos.xy - vec2f(0.0, 1.8)) - 2.0));

    var teeth  = abs(pos.x - 0.4)  - 0.16;
    teeth  = smax(teeth,  -0.45 - pos.y + pos.x * 0.1, 0.07);

    var teeth2 = abs(pos.x + 0.40) - 0.16;
    teeth2 = smax(teeth2,  0.5 + pos.y + pos.x * 0.05, 0.07);

    face = smax(face, -min(teeth, teeth2), 0.07);

    var sym = pos.xy; sym.x = abs(sym.x);

    var nose = -pos.y + 0.1;
    nose = max(nose, sym.x - 0.25 + sym.y * 0.5);

    var eyes = -pos.y + 0.48 - sym.x * 0.17;
    eyes = max(eyes,  sym.x - 1.0 + sym.y * 0.5);
    eyes = max(eyes, -sym.x - 0.05 + sym.y * 0.5);

    face   = min(face, nose);
    face   = min(face, eyes);
    face   = max(face, pos.z);

    pumpkin = smax(pumpkin, -face, 0.03);

    var res = vec2f(pumpkin, pumpkin_id);
    let candidate = vec2f(stem, stem_id);
    if (res.x >= candidate.x) {
        res = candidate;
    }
    res.x   = res.x / scale;
    return res;
}

// -------------------- flame (distance + color) --------------------
struct FlameOut { d: f32, col: vec3f, }

fn sdf_flame(pin: vec3f) -> FlameOut {
    let p = pin;

    // animate & tile a copy of p
    var q = p;
    q.z   = q.z + time.elapsed;
    q = vec3f(
        wrap_center(q.x + 0.5, 5.0, 2.5),
        wrap_center(q.y + 0.5, 5.0, 2.5),
        wrap_center(q.z + 0.5, 5.0, 2.5)
    );

    // twist
    let rxy = R2(q.y * 0.3 - time.elapsed * 1.35) * q.xz;
    q = vec3f(rxy.x, q.y, rxy.y);

    let d0 = sdf_pumpkin(q).x;
    let d1 = sdf_pumpkin(q - vec3f(0.1, 0.21, 0.1)).x;
    var d  = max(d0, -d1);

    // turbulence + thin glowing shell
    d = d + fbm(p * 1.6) * 0.454;
    d = abs(d) * 0.2315 + 0.001;

    // color up the column
    let h  = 3.1;
    let k  = clamp(S(2.191, h, p.y), 0.0, 1.0);
    let warm = mix(vec3f(1.0, 0.25, 0.06), vec3f(0.0, 0.98, 0.1), k);               // red->yellow
    let  c   = mix(warm, vec3f(1.0), smoothstep(0.6, 1.0, k));                      // hint of white

    let col = pow(vec3f(1.1) / vec3f(d), vec3f(2.2)) * c;
    return FlameOut(d, col);
}

// -------------------- scene / marcher --------------------

fn map_scene(p: vec3f) -> f32 {
    var u = p;
    u.z   = u.z + time.elapsed;
    u = vec3f(
        wrap_center(u.x + 0.5, 5.0, 2.5),
        wrap_center(u.y + 0.5, 5.0, 2.5),
        wrap_center(u.z + 0.5, 5.0, 2.5)
    );
    let rxy = R2(-time.elapsed * 1.5) * u.xz;
    u = vec3f(rxy.x, u.y, rxy.y);
    return sdf_pumpkin(u).x;
}

struct MarchOut { t: f32, glow: vec3f, }

fn ray_march(ro: vec3f, rd: vec3f) -> MarchOut {
    var t    = 0.0;
    var glow = vec3f(0.0);

    // keep this modest to avoid timeouts; bump if you want more fidelity
    for (var i = 0; i < 84; i = i + 1) {
        let p   = ro + rd * t;
        let f   = sdf_flame(p);
        glow    = glow + f.col * 0.00003; // visible gain
        let d   = min(map_scene(p), f.d);
        t       = t + max(d, 0.001);    // never stall
        if (t > 30.0) { break; }
    }
    return MarchOut(t, glow);
}

// -------------------- camera --------------------

fn build_ray(uv: vec2f, ro: vec3f, ta: vec3f, fovy: f32) -> vec3f {
    let cw = normalize(ta - ro);
    let cp = vec3f(0.0, 2.0, 0.0);                // simple up proxy
    let cu = normalize(cross(cw, cp));
    let cv = cross(cu, cw);
    let tanH = tan(0.5 * fovy);
    return normalize(cw * 2.0 + cu * (uv.x * tanH) + cv * (uv.y * tanH));
}
