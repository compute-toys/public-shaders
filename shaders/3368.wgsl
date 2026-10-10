// ============================================================
//  Cornell Box + 金属球 + 降噪（几何一致邻域钳位版）
// ============================================================

#workgroup_count path_trace_pass      120 68 1
#workgroup_count temporal_accum_pass  120 68 1
#workgroup_count atrous_pass_1        120 68 1
#workgroup_count atrous_pass_2        120 68 1
#workgroup_count atrous_pass_3        120 68 1
#workgroup_count tonemap_pass         120 68 1

#storage cam_state array<f32, 8>

const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;
const PATH_SPP: u32 = 1u;
const MAX_BOUNCES: u32 = 6u;
const MAX_HISTORY: f32 = 64.0;

// ------------------------------------------------------------
// RNG
// ------------------------------------------------------------
struct Rng { state: u32 }
fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn rng_new(seed: u32) -> Rng {
    var r: Rng; r.state = pcg_hash(seed | 1u); return r;
}
fn rng_f32(r: ptr<function, Rng>) -> f32 {
    (*r).state = pcg_hash((*r).state);
    return f32((*r).state) * 2.3283064365386963e-10;
}

// ------------------------------------------------------------
// 材质 / 场景
// ------------------------------------------------------------
const MAT_WHITE: u32 = 0u;
const MAT_RED:   u32 = 1u;
const MAT_GREEN: u32 = 2u;
const MAT_LIGHT: u32 = 3u;
const MAT_METAL: u32 = 4u;

fn material_albedo(m: u32) -> vec3f {
    if (m == MAT_RED)   { return vec3f(0.63, 0.065, 0.05); }
    if (m == MAT_GREEN) { return vec3f(0.14, 0.45, 0.091); }
    if (m == MAT_LIGHT) { return vec3f(0.0); }
    if (m == MAT_METAL) { return vec3f(1.0, 0.766, 0.336); }
    return vec3f(0.73);
}

const METAL_ALPHA: f32 = 0.15 * 0.15;
const METAL_F0: vec3f = vec3f(1.0, 0.766, 0.336);

struct Hit { t: f32, n: vec3f, mat: u32, hit: bool, }

fn hit_quad(ro: vec3f, rd: vec3f, axis: u32, plane: f32, lo: vec2f, hi: vec2f) -> f32 {
    let inv = 1.0 / rd[axis];
    let t = (plane - ro[axis]) * inv;
    if (t < 1e-4) { return -1.0; }
    let p = ro + rd * t;
    var u: f32; var v: f32;
    if (axis == 0u)      { u = p.y; v = p.z; }
    else if (axis == 1u) { u = p.x; v = p.z; }
    else                 { u = p.x; v = p.y; }
    if (u < lo.x || u > hi.x || v < lo.y || v > hi.y) { return -1.0; }
    return t;
}
fn box_slab(ro: vec3f, rd: vec3f, h: vec3f) -> vec2f {
    let inv = 1.0 / rd;
    let t0 = (-h - ro) * inv;
    let t1 = ( h - ro) * inv;
    let ts = min(t0, t1);
    let tb = max(t0, t1);
    return vec2f(max(max(ts.x, ts.y), ts.z), min(min(tb.x, tb.y), tb.z));
}
fn rot_y(p: vec3f, a: f32) -> vec3f {
    let c = cos(a); let s = sin(a);
    return vec3f(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}
fn box_normal_local(p: vec3f, h: vec3f) -> vec3f {
    let d = p / h;
    let a = abs(d);
    if (a.x > a.y && a.x > a.z) { return vec3f(sign(d.x), 0.0, 0.0); }
    if (a.y > a.z)              { return vec3f(0.0, sign(d.y), 0.0); }
    return vec3f(0.0, 0.0, sign(d.z));
}

const LIGHT_Y: f32 = 1.999;
const LIGHT_EMIT: vec3f = vec3f(15.0);
const LIGHT_N: vec3f = vec3f(0.0, -1.0, 0.0);
const LIGHT_MIN = vec2f(0.6, 0.6);
const LIGHT_MAX = vec2f(1.4, 1.4);
const LIGHT_AREA: f32 = 0.64;
const SPHERE_C = vec3f(0.5, 0.3, 0.6);
const SPHERE_R = 0.30;

fn intersect(ro: vec3f, rd: vec3f) -> Hit {
    var best: Hit;
    best.t = 1e30; best.n = vec3f(0.0); best.mat = MAT_WHITE; best.hit = false;
    var t: f32;

    t = hit_quad(ro, rd, 1u, 0.0, vec2f(0.0, 0.0), vec2f(2.0, 2.0));
    if (t > 0.0 && t < best.t) {
        best.t = t; best.n = vec3f(0.0, 1.0, 0.0); best.mat = MAT_WHITE; best.hit = true;
    }
    t = hit_quad(ro, rd, 1u, 2.0, vec2f(0.0, 0.0), vec2f(2.0, 2.0));
    if (t > 0.0 && t < best.t) {
        best.t = t; best.n = vec3f(0.0, -1.0, 0.0); best.mat = MAT_WHITE; best.hit = true;
        let hp = ro + rd * t;
        if (hp.x > LIGHT_MIN.x && hp.x < LIGHT_MAX.x &&
            hp.z > LIGHT_MIN.y && hp.z < LIGHT_MAX.y) { best.mat = MAT_LIGHT; }
    }
    t = hit_quad(ro, rd, 2u, 2.0, vec2f(0.0, 0.0), vec2f(2.0, 2.0));
    if (t > 0.0 && t < best.t) {
        best.t = t; best.n = vec3f(0.0, 0.0, -1.0); best.mat = MAT_WHITE; best.hit = true;
    }
    t = hit_quad(ro, rd, 0u, 0.0, vec2f(0.0, 0.0), vec2f(2.0, 2.0));
    if (t > 0.0 && t < best.t) {
        best.t = t; best.n = vec3f(1.0, 0.0, 0.0); best.mat = MAT_RED; best.hit = true;
    }
    t = hit_quad(ro, rd, 0u, 2.0, vec2f(0.0, 0.0), vec2f(2.0, 2.0));
    if (t > 0.0 && t < best.t) {
        best.t = t; best.n = vec3f(-1.0, 0.0, 0.0); best.mat = MAT_GREEN; best.hit = true;
    }
    {
        let c  = vec3f(0.62, 0.70, 1.25);
        let he = vec3f(0.30, 0.70, 0.30);
        let an = 0.35;
        let lo = rot_y(ro - c, -an); let ld = rot_y(rd, -an);
        let ts = box_slab(lo, ld, he);
        if (ts.x > 1e-4 && ts.y > ts.x && ts.x < best.t) {
            let pl = lo + ld * ts.x;
            best.t = ts.x;
            best.n = rot_y(box_normal_local(pl, he), an);
            best.mat = MAT_WHITE; best.hit = true;
        }
    }
    {
        let c  = vec3f(1.45, 0.30, 0.55);
        let he = vec3f(0.30, 0.30, 0.30);
        let an = -0.30;
        let lo = rot_y(ro - c, -an); let ld = rot_y(rd, -an);
        let ts = box_slab(lo, ld, he);
        if (ts.x > 1e-4 && ts.y > ts.x && ts.x < best.t) {
            let pl = lo + ld * ts.x;
            best.t = ts.x;
            best.n = rot_y(box_normal_local(pl, he), an);
            best.mat = MAT_WHITE; best.hit = true;
        }
    }
    {
        let oc = ro - SPHERE_C;
        let a  = dot(rd, rd);
        let hb = dot(oc, rd);
        let c  = dot(oc, oc) - SPHERE_R * SPHERE_R;
        let disc = hb * hb - a * c;
        if (disc > 0.0) {
            let sq = sqrt(disc);
            var ts = (-hb - sq) / a;
            if (ts < 1e-4) { ts = (-hb + sq) / a; }
            if (ts > 1e-4 && ts < best.t) {
                let pl = ro + rd * ts;
                best.t = ts;
                best.n = normalize(pl - SPHERE_C);
                best.mat = MAT_METAL; best.hit = true;
            }
        }
    }
    return best;
}

// ------------------------------------------------------------
// 采样 / GGX
// ------------------------------------------------------------
fn cosine_hemisphere(n: vec3f, rng: ptr<function, Rng>) -> vec3f {
    let r1 = rng_f32(rng); let r2 = rng_f32(rng);
    let phi = 2.0 * PI * r1;
    let r = sqrt(r2);
    let x = r * cos(phi); let y = r * sin(phi);
    let z = sqrt(max(0.0, 1.0 - r2));
    let t = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let u = normalize(cross(t, n));
    let v = cross(n, u);
    return normalize(u * x + v * y + n * z);
}
fn power_heuristic(a: f32, b: f32) -> f32 {
    let a2 = min(a, 1e18); let b2 = min(b, 1e18);
    let sa = a2 * a2; let sb = b2 * b2;
    let d = sa + sb;
    if (d <= 0.0) { return 0.0; }
    return sa / d;
}
fn pdf_light_solid(p: vec3f, lp: vec3f) -> f32 {
    let to_l = lp - p;
    let dist2 = dot(to_l, to_l);
    if (dist2 < 1e-10) { return -1.0; }
    let dist = sqrt(dist2);
    let dir = to_l / dist;
    let cos_l = abs(dot(LIGHT_N, dir));
    if (cos_l < 1e-6) { return -1.0; }
    return dist2 / (LIGHT_AREA * cos_l);
}
fn ggx_D(NoH: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}
fn smith_G1(NoV: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = NoV + sqrt(a2 + (1.0 - a2) * NoV * NoV);
    return 2.0 * NoV / max(denom, 1e-8);
}
fn fresnel_schlick(f0: vec3f, VoH: f32) -> vec3f {
    let c = clamp(1.0 - VoH, 0.0, 1.0);
    let c5 = c * c * c * c * c;
    return f0 + (vec3f(1.0) - f0) * c5;
}
fn ggx_sample_h(n: vec3f, alpha: f32, rng: ptr<function, Rng>) -> vec3f {
    let u1 = rng_f32(rng); let u2 = rng_f32(rng);
    let phi = 2.0 * PI * u1;
    let a2 = alpha * alpha;
    let cosTheta2 = (1.0 - u2) / (1.0 + (a2 - 1.0) * u2);
    let cosTheta = sqrt(clamp(cosTheta2, 0.0, 1.0));
    let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    let hLocal = vec3f(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    let t = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let b1 = normalize(cross(t, n));
    let b2 = cross(n, b1);
    return normalize(b1 * hLocal.x + b2 * hLocal.y + n * hLocal.z);
}
fn ggx_pdf_wi(wo: vec3f, wi: vec3f, n: vec3f, alpha: f32) -> f32 {
    let sum = wo + wi;
    let sl = dot(sum, sum);
    if (sl < 1e-12) { return 0.0; }
    let hh = sum * (1.0 / sqrt(sl));
    let NoH = dot(n, hh);
    let VoH = dot(wo, hh);
    if (NoH <= 0.0 || VoH <= 0.0) { return 0.0; }
    return ggx_D(NoH, alpha) * NoH / (4.0 * VoH);
}
fn ggx_brdf_over_pdf(wo: vec3f, wi: vec3f, n: vec3f, alpha: f32, f0: vec3f) -> vec3f {
    let NoV = max(dot(n, wo), 1e-6);
    let NoL = max(dot(n, wi), 1e-6);
    let sum = wo + wi;
    let sl = dot(sum, sum);
    if (sl < 1e-12) { return vec3f(0.0); }
    let hh = sum * (1.0 / sqrt(sl));
    let NoH = max(dot(n, hh), 0.0);
    let VoH = max(dot(wo, hh), 0.0);
    if (NoH <= 0.0 || VoH <= 0.0) { return vec3f(0.0); }
    let D = ggx_D(NoH, alpha);
    let G = smith_G1(NoV, alpha) * smith_G1(NoL, alpha);
    let F = fresnel_schlick(f0, VoH);
    return F * G * VoH / (NoV * NoH);
}

// ------------------------------------------------------------
// 路径追踪
// ------------------------------------------------------------
fn trace(ro_in: vec3f, rd_in: vec3f, rng: ptr<function, Rng>) -> vec3f {
    var ro = ro_in; var rd = rd_in;
    var radiance   = vec3f(0.0);
    var throughput = vec3f(1.0);
    var prev_pdf_bsdf = 1.0;
    var prev_p = vec3f(0.0);

    for (var b = 0u; b < MAX_BOUNCES; b++) {
        let h = intersect(ro, rd);
        if (!h.hit) { break; }
        let p = ro + rd * h.t;

        if (h.mat == MAT_LIGHT) {
            if (b == 0u) {
                radiance += throughput * LIGHT_EMIT;
            } else {
                var w = 1.0;
                if (prev_pdf_bsdf < 1e29) {
                    let pdf_l = pdf_light_solid(prev_p, p);
                    w = select(0.0, power_heuristic(prev_pdf_bsdf, pdf_l), pdf_l > 0.0);
                }
                radiance += throughput * LIGHT_EMIT * w;
            }
            break;
        }

        let n = h.n;

        if (h.mat == MAT_METAL) {
            let f0 = METAL_F0; let alpha = METAL_ALPHA;
            let wo = -rd;
            {
                let r1 = rng_f32(rng); let r2 = rng_f32(rng);
                let lp = vec3f(
                    LIGHT_MIN.x + r1 * (LIGHT_MAX.x - LIGHT_MIN.x),
                    LIGHT_Y,
                    LIGHT_MIN.y + r2 * (LIGHT_MAX.y - LIGHT_MIN.y)
                );
                let to_l = lp - p;
                let dist2 = dot(to_l, to_l);
                let dist = sqrt(dist2);
                let wi_l = to_l / dist;
                if (dot(n, wi_l) > 0.0) {
                    let pdf_l = pdf_light_solid(p, lp);
                    if (pdf_l > 0.0) {
                        let pdf_b = ggx_pdf_wi(wo, wi_l, n, alpha);
                        if (pdf_b > 0.0) {
                            let w_mis = power_heuristic(pdf_l, pdf_b);
                            let sh = intersect(p + n * 1e-3, wi_l);
                            if (!sh.hit || sh.t > dist - 1e-3) {
                                let f_cos = ggx_brdf_over_pdf(wo, wi_l, n, alpha, f0) * pdf_b;
                                radiance += throughput * f_cos * LIGHT_EMIT * w_mis / pdf_l;
                            }
                        }
                    }
                }
            }
            let hh = ggx_sample_h(n, alpha, rng);
            let wi = reflect(rd, hh);
            if (dot(n, wi) <= 0.0) { break; }
            throughput *= ggx_brdf_over_pdf(wo, wi, n, alpha, f0);
            prev_pdf_bsdf = ggx_pdf_wi(wo, wi, n, alpha);
            prev_p = p;
            if (b >= 2u) {
                let q = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.95);
                if (rng_f32(rng) > q) { break; }
                throughput /= q;
            }
            ro = p + n * 1e-3; rd = wi;
            continue;
        }

        let albedo = material_albedo(h.mat);
        {
            let r1 = rng_f32(rng); let r2 = rng_f32(rng);
            let lp = vec3f(
                LIGHT_MIN.x + r1 * (LIGHT_MAX.x - LIGHT_MIN.x),
                LIGHT_Y,
                LIGHT_MIN.y + r2 * (LIGHT_MAX.y - LIGHT_MIN.y)
            );
            let to_l = lp - p;
            let dist2 = dot(to_l, to_l);
            let dist = sqrt(dist2);
            let wi = to_l / dist;
            let cos_s = dot(n, wi);
            if (cos_s > 0.0) {
                let pdf_l = pdf_light_solid(p, lp);
                if (pdf_l > 0.0) {
                    let pdf_b = cos_s * INV_PI;
                    let w_mis = power_heuristic(pdf_l, pdf_b);
                    let sh = intersect(p + n * 1e-3, wi);
                    if (!sh.hit || sh.t > dist - 1e-3) {
                        let contrib = albedo * INV_PI * LIGHT_EMIT * cos_s * w_mis / pdf_l;
                        radiance += throughput * contrib;
                    }
                }
            }
        }
        let wi = cosine_hemisphere(n, rng);
        let cos_s = dot(n, wi);
        if (cos_s <= 0.0) { break; }
        let pdf_bsdf = cos_s * INV_PI;
        throughput *= albedo;
        prev_pdf_bsdf = pdf_bsdf;
        prev_p = p;
        if (b >= 2u) {
            let q = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.95);
            if (rng_f32(rng) > q) { break; }
            throughput /= q;
        }
        ro = p + n * 1e-3; rd = wi;
    }
    return radiance;
}

// ------------------------------------------------------------
// 工具
// ------------------------------------------------------------
fn aces_tonemap(x: vec3f) -> vec3f {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}
fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}
fn is_bad(x: f32) -> bool {
    let bits = bitcast<u32>(x);
    return (bits & 0x7F800000u) == 0x7F800000u;
}
fn any_bad(v: vec3f) -> bool {
    return is_bad(v.x) || is_bad(v.y) || is_bad(v.z);
}
struct CamBasis { pos: vec3f, dir: vec3f, right: vec3f, up: vec3f, hw: f32, hh: f32, }
fn make_camera(yaw: f32, pitch: f32, zoom: f32, aspect: f32) -> CamBasis {
    let center = vec3f(1.0, 1.0, 1.0);
    let radius = 3.4 / zoom;
    let cosP = cos(pitch);
    let cam_pos = center + vec3f(
        radius * cosP * sin(yaw),
        radius * sin(pitch),
        -radius * cosP * cos(yaw)
    );
    let w = normalize(center - cam_pos);
    let u = normalize(cross(vec3f(0.0, 1.0, 0.0), w));
    let v = cross(w, u);
    let max_tan = 0.99 / 3.4;
    var hw = max_tan; var hh = max_tan;
    if (aspect > 1.0) { hh = max_tan / aspect; } else { hw = max_tan * aspect; }
    var cb: CamBasis;
    cb.pos = cam_pos; cb.dir = w; cb.right = u; cb.up = v;
    cb.hw = hw; cb.hh = hh;
    return cb;
}

// ------------------------------------------------------------
// PASS 1: 路径追踪
// ------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn path_trace_pass(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let dims = vec2f(f32(screen_size.x), f32(screen_size.y));
    let isFirst = (id.x == 0u && id.y == 0u);

    var yaw      = cam_state[0];
    var pitch    = cam_state[1];
    let zoom_prv = cam_state[2];
    let mx_prv   = cam_state[3];
    let my_prv   = cam_state[4];
    let click_prv = cam_state[5];

    let mx_now = f32(mouse.pos.x);
    let my_now = f32(mouse.pos.y);
    let click_now = f32(mouse.click);
    let cur_zoom = max(mouse.zoom, 0.15);

    let just_pressed = (click_now == 1.0) && (click_prv != 1.0);
    let ddx = mx_now - mx_prv;
    let ddy = my_now - my_prv;
    let mouse_moved = (ddx * ddx + ddy * ddy) > 0.25;
    let rotating = (click_now == 1.0) && (!just_pressed) && mouse_moved;

    if (rotating) {
        yaw   -= ddx * 0.005;
        pitch  = clamp(pitch + ddy * 0.005, -1.4, 1.4);
    }

    let cam_moved = rotating || (abs(cur_zoom - zoom_prv) > 1e-3);

    if (isFirst) {
        cam_state[0] = yaw;
        cam_state[1] = pitch;
        cam_state[2] = cur_zoom;
        cam_state[3] = mx_now;
        cam_state[4] = my_now;
        cam_state[5] = click_now;
        cam_state[6] = select(0.0, 1.0, cam_moved);
    }

    let aspect = dims.x / dims.y;
    let cb = make_camera(yaw, pitch, cur_zoom, aspect);
    let fragCoord = vec2f(f32(id.x) + 0.5, f32(screen_size.y - id.y) - 0.5);
    let uv = fragCoord / dims;

    var rng = rng_new(
        (id.x * 1973u) ^ (id.y * 9277u) ^ ((time.frame + 1u) * 26699u)
    );

    var col = vec3f(0.0);
    var first_hit_depth = 1e30;
    var first_hit_normal = vec3f(0.0, 1.0, 0.0);

    for (var s = 0u; s < PATH_SPP; s++) {
        let jitter = vec2f(rng_f32(&rng), rng_f32(&rng)) - 0.5;
        let px = ((uv.x + jitter.x / dims.x) * 2.0 - 1.0) * cb.hw;
        let py = ((uv.y + jitter.y / dims.y) * 2.0 - 1.0) * cb.hh;
        let rd = normalize(cb.dir + cb.right * px + cb.up * py);
        if (s == 0u) {
            let gh = intersect(cb.pos, rd);
            if (gh.hit) { first_hit_depth = gh.t; first_hit_normal = gh.n; }
        }
        col += trace(cb.pos, rd, &rng);
    }
    col /= f32(PATH_SPP);
    if (any_bad(col)) { col = vec3f(0.0); }

    let c = vec2i(i32(id.x), i32(id.y));
    textureStore(pass_out, c, 0, vec4f(col, 0.0));
    textureStore(pass_out, c, 1, vec4f(first_hit_normal, first_hit_depth));
}

// ------------------------------------------------------------
// PASS 2: 时间累积 + 几何一致邻域钳位 → L2
// ------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn temporal_accum_pass(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let c = vec2i(i32(id.x), i32(id.y));

    let raw = passLoad(0, c, 0).xyz;

    // ---- 几何一致邻域钳位 ----
    // 只统计与中心像素"几何相近"的邻居：
    //   法线点积 > 0.9（约 25° 以内）
    //   深度相对差 < 5%
    let W = i32(screen_size.x);
    let H = i32(screen_size.y);

    let center_n = passLoad(1, c, 0).xyz;
    let center_z = passLoad(1, c, 0).w;

    var cmin = raw;
    var cmax = raw;
    var count = 1.0;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let nc = vec2i(clamp(c.x + dx, 0, W - 1), clamp(c.y + dy, 0, H - 1));
            let s_n = passLoad(1, nc, 0).xyz;
            let s_z = passLoad(1, nc, 0).w;

            // 几何一致性检查
            let n_dot = dot(center_n, s_n);
            let z_diff = abs(center_z - s_z) / max(center_z, 1e-3);

            // 法线朝向一致 且 深度相近 才纳入
            if (n_dot > 0.9 && z_diff < 0.05) {
                let s = passLoad(0, nc, 0).xyz;
                cmin = min(cmin, s);
                cmax = max(cmax, s);
                count += 1.0;
            }
        }
    }

    // 邻居太少（孤立像素）就放宽：退化为全局 3x3
    if (count < 3.0) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let nc = vec2i(clamp(c.x + dx, 0, W - 1), clamp(c.y + dy, 0, H - 1));
                let s = passLoad(0, nc, 0).xyz;
                cmin = min(cmin, s);
                cmax = max(cmax, s);
            }
        }
    }

    let clamped_raw = clamp(raw, cmin, cmax);

    // 历史
    let hist_color = passLoad(2, c, 0).xyz;
    let hist_count = passLoad(2, c, 0).w;
    let cam_moved = cam_state[6] > 0.5;

    var new_color = clamped_raw;
    var new_count = 1.0;
    if (!cam_moved && hist_count > 0.5) {
        let n = min(hist_count, MAX_HISTORY - 1.0);
        new_color = (hist_color * n + clamped_raw) / (n + 1.0);
        new_count = min(n + 1.0, MAX_HISTORY);
    }

    if (any_bad(new_color)) { new_color = clamped_raw; new_count = 1.0; }

    textureStore(pass_out, c, 2, vec4f(new_color, new_count));
}

// ------------------------------------------------------------
// à-trous 核
// ------------------------------------------------------------
fn atrous_filter(c: vec2i, step: i32, screen_size: vec2u, src_layer: i32) -> vec3f {
    let center_col = passLoad(src_layer, c, 0).xyz;
    let center_n   = passLoad(1, c, 0).xyz;
    let center_z   = passLoad(1, c, 0).w;
    let center_lum = luminance(center_col);

    let SIGMA_N = 32.0;
    let SIGMA_Z = 0.4;
    let SIGMA_L = 2.5;

    var sum = vec3f(0.0);
    var wsum = 0.0;
    let W = i32(screen_size.x);
    let H = i32(screen_size.y);

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nc = vec2i(
                clamp(c.x + dx * step, 0, W - 1),
                clamp(c.y + dy * step, 0, H - 1)
            );
            let s_col = passLoad(src_layer, nc, 0).xyz;
            let s_n   = passLoad(1, nc, 0).xyz;
            let s_z   = passLoad(1, nc, 0).w;
            let s_lum = luminance(s_col);

            let wn = pow(max(dot(center_n, s_n), 0.0), SIGMA_N);
            let wz = exp(-abs(center_z - s_z) / SIGMA_Z);
            let wl = exp(-abs(center_lum - s_lum) / SIGMA_L);

            let w = wn * wz * wl;
            sum  += s_col * w;
            wsum += w;
        }
    }

    var result = select(center_col, sum / max(wsum, 1e-6), wsum > 1e-6);
    if (any_bad(result)) { result = center_col; }
    return result;
}

@compute @workgroup_size(16, 16)
fn atrous_pass_1(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let c = vec2i(i32(id.x), i32(id.y));
    let r = atrous_filter(c, 1, screen_size, 2);
    textureStore(pass_out, c, 3, vec4f(r, 1.0));
}

@compute @workgroup_size(16, 16)
fn atrous_pass_2(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let c = vec2i(i32(id.x), i32(id.y));
    let r = atrous_filter(c, 2, screen_size, 3);
    textureStore(pass_out, c, 0, vec4f(r, 0.0));
}

@compute @workgroup_size(16, 16)
fn atrous_pass_3(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let c = vec2i(i32(id.x), i32(id.y));
    let r = atrous_filter(c, 4, screen_size, 0);
    textureStore(pass_out, c, 3, vec4f(r, 1.0));
}

// ------------------------------------------------------------
// PASS 4: 色调映射
// ------------------------------------------------------------
@compute @workgroup_size(16, 16)
fn tonemap_pass(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let c = vec2i(i32(id.x), i32(id.y));
    var col = passLoad(3, c, 0).xyz;
    if (any_bad(col)) { col = vec3f(0.0); }
    col = aces_tonemap(col * 1.2);
    textureStore(screen, id.xy, vec4f(col, 1.0));
}