const PI: f32 = 3.1415926;
const CSIZE: f32 = 256.0;
const GSIZE: f32 = 64.0;
const EPSILON: vec2f = vec2f(0.001, 0.000);

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy) - 0.5 * vec2f(screen_size);
    let local_coord = to_local(frag_coord);

    let duvdx = 0.5 * (to_local(frag_coord + EPSILON) - to_local(frag_coord - EPSILON)) / EPSILON.x;
    let duvdy = 0.5 * (to_local(frag_coord + EPSILON.yx) - to_local(frag_coord - EPSILON.yx)) / EPSILON.x;

    // First LF
    let sdf_cb = sdf_checkerboard(local_coord, GSIZE);
    let nabla_cb = nabla_checkerboard(local_coord, GSIZE);
    let nabla = vec2f(dot(nabla_cb, duvdx), dot(nabla_cb, duvdy));
    var signal = linearstep(-length(nabla), length(nabla), sdf_cb);
    // Second LF
    let fsize = length(to_local(vec2f(1)) - to_local(vec2f(0.01)));
    let lod = linearstep(0.0, GSIZE, fsize);
    signal = mix(signal, 0.5, lod);

    var col = vec3f(signal);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn to_local(frag_coord: vec2f) -> vec2f {
    let s = vec2f(custom.scale_x, custom.scale_y);
    let t = custom.theta * PI;
    let m = mat2x2f(cos(t), sin(t), -sin(t), cos(t));

    let coord = s * (m * frag_coord);

    let cell_coord = modulo2(coord, vec2f(CSIZE)) - 0.5 * CSIZE;

    let R = 0.4 * CSIZE;
    let dist = sdf_circle(cell_coord, R); 
    // var r = max(-dist, 0.0) / R;     // no aliasing
    var r = op_smax(-dist, 0.0, 2) / R;
    r = sqrt(abs(2.0 * r - r * r));

    let strength = 2.0 * (0.5 * sin(time.elapsed) + 0.5);

    return coord + strength * (1 - r) * cell_coord;
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_checkerboard(p: vec2f, gsize: f32) -> f32 {
    let sdf_v = op_onion(p.x, gsize, 0.5 * gsize);
    let sdf_h = op_onion(p.y, gsize, 0.5 * gsize);

    return op_xor(sdf_v, sdf_h);
}

fn nabla_checkerboard(p: vec2f, gsize: f32) -> vec2f {
    const NV: vec2f = vec2f(1.0, 0.0);
    const NH: vec2f = vec2f(0.0, 1.0);
    
    let sdf_v = op_onion(p.x, gsize, 0.5 * gsize);
    let sdf_h = op_onion(p.y, gsize, 0.5 * gsize);

    return nabla_xor(sdf_v, sdf_h, NV, NH);
}

fn op_onion(d: f32, l: f32, t: f32) -> f32 {
    return abs(modulo(d, 2.0 * l) - l) - t;
}

fn op_xor(d1: f32, d2: f32) -> f32 {
    return max(min(d1, d2), -max(d1, d2));
}

fn op_smin(d1: f32, d2: f32, k: f32) -> f32 {
    let k4 = 4.0 * k;
    let h = max(k4 - abs(d1 - d2), 0.0) / k4;
    return min(d1, d2) - 0.25 * h * h * k4;
}

fn op_smax(d1: f32, d2: f32, k: f32) -> f32 {
    return -op_smin(-d1, -d2, k);
}

fn nabla_union(a: f32, b: f32, na: vec2f, nb: vec2f) -> vec2f {
    return select(nb, na, a < b);
}

fn nabla_intersection(a: f32, b: f32, na: vec2f, nb: vec2f) -> vec2f {
    return select(nb, na, a > b);
}

fn nabla_substraction(a: f32, b: f32, na: vec2f, nb: vec2f) -> vec2f {
    return nabla_intersection(a, -b, na, -nb);
}

fn nabla_xor(a: f32, b: f32, na: vec2f, nb: vec2f) -> vec2f {
    let sdf = select(vec2f(b, a), vec2f(a, b), a < b);
    let nabla = select(vec4f(nb, na), vec4f(na, nb), a < b);

    return nabla_intersection(sdf.x, -sdf.y, nabla.xy, -nabla.zw);
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return clamp((x - e0) / (e1 - e0), 0.0, 1.0);
}

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn modulo2(a: vec2f, b: vec2f) -> vec2f {
    return a - b * floor(a / b);
}