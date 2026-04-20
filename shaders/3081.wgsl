// ==========================================
// STRUCTS
// ==========================================

struct Transform2D {
    pos: vec2f,
    angle: f32,
    scale: vec2f,
    anchor: vec2f,
};

struct TangramPiece {
    type_id: u32,
    color: vec3f,
    transform: Transform2D,
}

struct SDFResult {
    dist: f32,
    color: vec3f,
}

// ==========================================
// CONSTANTS
// ==========================================

const PI: f32 = 3.14159265359;
const bg_col = vec3f(0.9, 0.8, 0.7);

const square_col = vec3<f32>(0.773, 0.561, 0.702);
const bigtri1_col = vec3<f32>(0.502, 0.749, 0.239);
const bigtri2_col = vec3<f32>(0.494, 0.325, 0.545);
const midtri_col = vec3<f32>(0.439, 0.573, 0.235);
const smalltri1_col = vec3<f32>(0.604, 0.137, 0.443);
const smalltri2_col = vec3<f32>(0.012, 0.522, 0.298);
const parallelogram_col = vec3<f32>(0.133, 0.655, 0.420);

const pieces: array<TangramPiece, 7> = array(
    TangramPiece(0u, square_col, Transform2D(vec2f(0.0), 0.0, vec2f(1.0), vec2f(0.0))),
    TangramPiece(1u, bigtri1_col, Transform2D(vec2f(0.0), 0.0, vec2f(1.0), vec2f(0.0))),
    TangramPiece(2u, bigtri2_col, Transform2D(vec2f(0.0), 0.0, vec2f(1.0), vec2f(0.0))),
    TangramPiece(3u, midtri_col, Transform2D(vec2f(0.0), 0.0, vec2f(1.0), vec2f(0.0))),
    TangramPiece(4u, smalltri1_col, Transform2D(vec2f(0.0), 0.0, vec2f(1.0), vec2f(0.0))),
    TangramPiece(5u, smalltri2_col, Transform2D(vec2f(0.0), 0.0, vec2f(1.0), vec2f(0.0))),
    TangramPiece(6u, parallelogram_col, Transform2D(vec2f(0.0), 0.0, vec2f(1.0), vec2f(0.0))),
);

// ==========================================
// ANIMATION STATES
// ==========================================

const state_closed: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(0.0, 0.0), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.0, 0.0), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.0, 0.0), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.0, 0.0), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.0, 0.0), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.0, 0.0), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.0, 0.0), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

const state_opened: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(-0.25, 0.0), -PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.0, 0.8), -0.18, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.8, 0.3), -0.18, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.6, -0.6), 0.33, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.5, 0.2), 0.1, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.83, -0.2), -0.22, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.6, -0.5), 0.15, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

const state_cat1: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(0.7, 0.79), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.5, 0.0), -PI * 0.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.5, -1.41), PI * 1.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.21, 0.29), PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(1.7, 1.79), PI, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(1.20, 1.29), PI * 0.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.0, -0.91), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

const state_cat2: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(0.9, -0.21), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.8, -0.5), PI, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.9, -0.21), PI * 0.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.095, 0.205), PI * 1.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.9, -0.21), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(1.40, 0.29), PI * 1.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.8, -0.5), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

const state_cat3: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(-0.1, 0.91), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.51, -0.5), -PI * 0.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.9, -0.5), PI * 1.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.9, -1.9), PI * 1.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.9, 1.91), PI, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.4, 1.41), PI * 0.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.19, -0.5), PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

const state_cat4: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(-1.02, 0.5), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.515, 0.0), PI, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.9, 0.0), PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.19, -0.71), PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.02, 0.5), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.52, 1.0), PI * 1.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(1.61, -1.42), PI * 0.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

const state_cat5: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(-1.0, -0.25), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.91, -0.75), PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(1.61, -1.458), -PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.2, -0.04), PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.0, -0.25), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.5, 0.25), PI * 1.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.2, -0.46), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

const state_cat6: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(1.3, -0.86), PI * 0.666, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.91, -0.75), PI * 0.666, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.675, -1.085), -PI * 1.085, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.515, -0.65), PI * 1.416, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.61, -0.67), PI * 0.16, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.8, 0.005), PI * 1.666, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-2.49, 0.05), PI * 0.45, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

const state_heart: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(-0.5, -1.00), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.5, -1.00), 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.5, 1.0), PI * 0.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.5, -1.0), PI * 0.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.0, 1.5), PI * 1.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.0, -0.5), PI * 1.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(1.0, -0.5), PI, vec2<f32>(-1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

const state_letter_c: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(0.3, 2.0), PI * 1.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.402, 1.0), PI * 1.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.4, -0.41), PI * 1.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.69, 0.3), PI * 1.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(1.3, 2.0), PI * 1.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.015, -1.11), PI * 1.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.015, 0.3), PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

const state_letter_a: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(0.29, -0.22), PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(1.0, -0.2), PI * 0.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.10, -0.5), PI * 0.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.4, -0.22), PI * 0.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.6, 0.0), PI * 1.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.4, 0.495), PI * 1.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.41, 0.485), PI * 0.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

const state_letter_m: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(-0.70, 0.0), PI * 1.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(1.41, 0.0), PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.41, 0.0), PI * 1.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.0, 0.71), PI * 1.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-2.11, -0.70), PI * 1.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(1.4, -0.0), PI * 1.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.695, -1.41), PI * 0.75, vec2<f32>(-1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

const state_letter_r: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(0.0, 0.0), PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.0, 1.66), PI * 1.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.0, 0.25), PI * 1.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.0, 0.25), PI * 1.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(1.71, 1.35), PI * 1.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(1.1, 1.66), PI * 1.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.1, -0.7), PI * 0.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

const state_letter_d: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(-0.193, -0.4), PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.6, 1.7), PI * 1.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.6, -1.1), PI * 1.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.19, 0.31), PI * 1.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.195, -0.4), PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.19, 1.0), PI * 1.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.19, 0.29), PI * 0.75, vec2<f32>(-1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

const state_letter_s: array<Transform2D, 7> = array(
    Transform2D(vec2<f32>(-0.4, -0.3), PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(1.3, -0.59), PI * 0.5, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.4, 0.7), PI * 1.0, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.3, 0.29), PI * 0.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(0.3, -0.175), PI * 1.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-0.4, 1.7), PI * 0.25, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
    Transform2D(vec2<f32>(-1.1, -1.589), PI * 0.75, vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0)),
);

// ==========================================
// MATH & PRIMITIVES
// ==========================================

fn transform_to_local(uv: vec2f, xform: Transform2D) -> vec2f {
    var p = uv - xform.pos;
    let c = cos(xform.angle);
    let s = sin(xform.angle);
    p = vec2f(c * p.x + s * p.y, -s * p.x + c * p.y);
    p -= xform.anchor;
    let sg = select(vec2f(-1.0), vec2f(1.0), xform.scale >= vec2f(0.0));
    let safe_scale = sg * max(abs(xform.scale), vec2f(0.001));
    p /= safe_scale;
    return p;
}

fn scale_sdf_distance(dist: f32, xform: Transform2D) -> f32 {
    let s = abs(xform.scale);
    if (abs(s.x - s.y) < 0.001) {
        return dist * s.x;
    }
    let ratio = max(s.x, s.y) / min(s.x, s.y);
    if (ratio < 2.0) {
        return dist * (2.0 / (1.0 / s.x + 1.0 / s.y));
    }
    return dist * min(s.x, s.y);
}

fn box(p: vec2f, b: vec2f) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}

fn tri(p: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> f32 {
    let e0 = p1 - p0; let e1 = p2 - p1; let e2 = p0 - p2;
    let v0 = p - p0;  let v1 = p - p1;  let v2 = p - p2;
    let pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
    let pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
    let pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
    let s = sign(e0.x * e2.y - e0.y * e2.x);
    let d0 = vec2<f32>(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x));
    let d1 = vec2<f32>(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x));
    let d2 = vec2<f32>(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x));
    let d = min(min(d0, d1), d2);
    return -sqrt(d.x) * sign(d.y);
}

fn parallelogram(p_in: vec2<f32>, wi: f32, he: f32, sk: f32) -> f32 {
    let e = vec2<f32>(sk, he);
    var p = p_in;
    if (p.y < 0.0) { p = -p; }
    var w = p - e; w.x = w.x - clamp(w.x, -wi, wi);
    var d = vec2<f32>(dot(w, w), -w.y);
    let s = p.x * e.y - p.y * e.x;
    if (s < 0.0) { p = -p; }
    var v = p - vec2<f32>(wi, 0.0);
    v = v - e * clamp(dot(v, e) / dot(e, e), -1.0, 1.0);
    d = min(d, vec2<f32>(dot(v, v), wi * he - abs(s)));
    return sqrt(d.x) * sign(-d.y);
}

// ==========================================
// TANGRAM PIECE SDF (unified, scale-safe)
// ==========================================

fn tangramPieceSDF(p: vec2f, piece: TangramPiece, transform: Transform2D) -> f32 {
    let q = transform_to_local(p, transform);

    switch piece.type_id {
        case 0u: {
            var t: Transform2D;
            t.pos = vec2f(0.5, 0.0);
            t.angle = PI * 0.25;
            t.scale = vec2f(1.0);
            t.anchor = vec2f(0.0);
            let raw = box(transform_to_local(q, t), vec2f(0.3535));
            return scale_sdf_distance(raw, t) * abs(transform.scale.x);
        }
        case 1u: { return scale_sdf_distance(tri(q, vec2(-1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 1.0)), transform); }
        case 2u: { return scale_sdf_distance(tri(q, vec2(-1.0, 1.0), vec2(0.0, 0.0), vec2(-1.0, -1.0)), transform); }
        case 3u: { return scale_sdf_distance(tri(q, vec2(1.0, -1.0), vec2(1.0, 0.0), vec2(0.0, -1.0)), transform); }
        case 4u: { return scale_sdf_distance(tri(q, vec2(1.0, 1.0), vec2(1.0, 0.0), vec2(0.5, 0.5)), transform); }
        case 5u: { return scale_sdf_distance(tri(q, vec2(0.0, 0.0), vec2(0.5, -0.5), vec2(-0.5, -0.5)), transform); }
        case 6u: {
            var t: Transform2D;
            t.pos = vec2f(-0.25, -0.75);
            t.angle = 0.0;
            t.scale = vec2f(1.0);
            t.anchor = vec2f(0.0);
            let raw = parallelogram(transform_to_local(q, t), 0.5, 0.25, 0.25);
            return scale_sdf_distance(raw, t) * abs(transform.scale.x);
        }
        default: { return 1e10; }
    }
}

// ==========================================
// ANIMATION SYSTEM (BPM-driven)
// ==========================================

fn mixTransform(a: Transform2D, b: Transform2D, t: f32) -> Transform2D {
    var result: Transform2D;
    result.pos = mix(a.pos, b.pos, t);
    result.scale = mix(a.scale, b.scale, t);
    result.anchor = mix(a.anchor, b.anchor, t);
    result.angle = mix(a.angle, b.angle, t);
    return result;
}

fn get_state(index: u32, piece_index: u32) -> Transform2D {
    switch index {
        case 0u:  { return state_closed[piece_index]; }
        case 1u:  { return state_opened[piece_index]; }
        case 2u:  { return state_cat1[piece_index]; }
        case 3u:  { return state_cat2[piece_index]; }
        case 4u:  { return state_cat3[piece_index]; }
        case 5u:  { return state_cat4[piece_index]; }
        case 6u:  { return state_cat5[piece_index]; }
        case 7u:  { return state_cat6[piece_index]; }
        case 8u:  { return state_heart[piece_index]; }
        case 9u:  { return state_letter_c[piece_index]; }
        case 10u: { return state_letter_a[piece_index]; }
        case 11u: { return state_letter_m[piece_index]; }
        case 12u: { return state_letter_r[piece_index]; }
        case 13u: { return state_letter_d[piece_index]; }
        case 14u: { return state_letter_s[piece_index]; }
        default:  { return state_closed[piece_index]; }
    }
}

fn get_animated_transform(piece_index: u32, time_val: f32) -> Transform2D {
    const NUM_PHASES = 9u;
    const BPM = 170.0;
    const BEATS_PER_HOLD = 4.0;
    const BEATS_PER_TRANSITION = 2.0;
    const BEATS_PER_PHASE = 6.0;

    var seq = array<u32, 9>(
        9u,  // c
        10u, // a
        11u, // m
        10u, // a
        12u, // r
        10u, // a
        13u, // d
        10u, // a
        14u  // s
    );

    let beat = time_val * (BPM / 60.0);
    let total_beats = BEATS_PER_PHASE * f32(NUM_PHASES);
    let beat_mod = beat % total_beats;

    let phase = u32(floor(beat_mod / BEATS_PER_PHASE)) % NUM_PHASES;
    let beat_in_phase = beat_mod - f32(phase) * BEATS_PER_PHASE;
    let next_phase = (phase + 1u) % NUM_PHASES;

    var f: f32;
    if (beat_in_phase < BEATS_PER_HOLD) {
        f = 0.0;
    } else {
        let t = (beat_in_phase - BEATS_PER_HOLD) / BEATS_PER_TRANSITION;
        f = smoothstep(0.0, 1.0, t);
    }

    return mixTransform(
        get_state(seq[phase], piece_index),
        get_state(seq[next_phase], piece_index),
        f
    );
}

// ==========================================
// RENDERING (clean, original style)
// ==========================================

fn renderTangram(uv: vec2f, transform: Transform2D, screen_size: vec2u) -> vec3f {
    let q = transform_to_local(uv, transform);
    var result = SDFResult(1e10, vec3f(0.0));

    for (var i = 0u; i < 7u; i++) {
        let anim_transform = get_animated_transform(i, time.elapsed);
        let piece_dist = tangramPieceSDF(q, pieces[i], anim_transform);
        let d = scale_sdf_distance(piece_dist, transform);

        if (d < result.dist) {
            result.dist = d;
            result.color = pieces[i].color;
        }
    }

    let aa_width = 1.5 / f32(screen_size.y);
    let t = smoothstep(-aa_width, aa_width, result.dist);
    return mix(result.color, bg_col, t);
}

// ==========================================
// MAIN
// ==========================================

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = (fragCoord * 2.0 - vec2<f32>(screen_size)) / f32(screen_size.y);

    const tangramScale = 0.333;
    const transformTangram = Transform2D(vec2f(0.0), 0.0, vec2f(tangramScale), vec2f(0.0));

    var col = renderTangram(uv, transformTangram, screen_size);

    col = pow(col, vec3f(2.2));

    textureStore(screen, id.xy, vec4f(col, 1.0));
}

