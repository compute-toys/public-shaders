const R: f32 = 32.0;
const D: f32 = 40.0;
const PI: f32 = 3.1415926;
const GSIZE: f32 = 1.14 * 2.0 * (R + D);
const LIGHT: vec3f = normalize(vec3f(0.75, 0.5, 1.));

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy);

    let value = pattern(frag_coord);

    let tt = vec2f(0.5, 0.0);

    let dpdx_ = pattern(frag_coord + tt) - pattern(frag_coord - tt);
    let dpdy_ = pattern(frag_coord + tt.yx) - pattern(frag_coord -tt.yx);
    let fwidt = length(vec2f(dpdx_, dpdy_));

    let normal = normalize(vec3f(-dpdx_, -dpdy_, 1.));

    let mask = linearstep(-fwidt, fwidt, value);

    let grid = grid(frag_coord);

    // var col = vec3f(normal * grid);
    // var col = vec3f(dot(normal, LIGHT) * mask);
    var col = vec3f(value / R * mask * grid);

    // Convert from gamma-encoded to linear colour space
    // col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn op_onion(d: f32, l: f32, t: f32) -> f32 {
    return abs(modulo(d, 2.0 * l) - l) - t;
}

// fn sdf_line(p: vec2f, a: f32, b: f32, c: f32) -> f32 {
//     return (a * p.x + b * p.y + c) / length(vec2f(a, b));
// }

fn op_smin(d1: f32, d2: f32, k: f32) -> f32 {
    let k4 = 4.0 * k;
    let h = max(k4 - abs(d1 - d2), 0.0) / k4;
    return min(d1, d2) - 0.25 * h * h * k4;
}

fn op_smax(d1: f32, d2: f32, k: f32) -> f32 {
    return -op_smin(-d1, -d2, k);
}

fn pattern(p: vec2f) -> f32 {
    let gsize = vec2f(GSIZE);
    let hgsize = 0.5 * gsize;
    
    var cell_coord = modulo2(p - hgsize, gsize) - hgsize;
    cell_coord = cell_coord * rmat(PI / 12);
    var sdf = cell_pattern(cell_coord);

    cell_coord = modulo2(p, gsize) - hgsize;
    cell_coord = cell_coord * rmat(PI / 12);
    sdf = min(sdf, cell_pattern(cell_coord));

    cell_coord = modulo2(p - vec2f(hgsize.x, 0.), gsize) - hgsize;
    cell_coord = cell_coord * rmat(5. / 12 * PI);
    sdf = min(sdf, cell_pattern(cell_coord));

    cell_coord = modulo2(p - vec2f(0., hgsize.y), gsize) - hgsize;
    cell_coord = cell_coord * rmat(5. / 12 * PI);
    sdf = min(sdf, cell_pattern(cell_coord));

    // var ndist = max(-sdf, 0) / R;
    // ndist = o_ease_circ(ndist);

    // return ndist * R;
    return max(-sdf, 0.);
}

fn cell_pattern(p: vec2f) -> f32 {
    var sdf = sdf_circle(p, R);
    sdf = min(sdf, sdf_circle(p - vec2f(D, 0.), R));
    sdf = min(sdf, sdf_circle(p + vec2f(D, 0.), R));

    let T = sqrt(4.0 * R * R - D * D);

    sdf = max(sdf, -sdf_circle(p - vec2f(0., T), R));
    sdf = max(sdf, -sdf_circle(p + vec2f(0., T), R));

    return sdf;
}

fn grid(p: vec2f) -> f32 {
    let vsdf = op_onion(p.x - 0.5 * GSIZE, 0.5 * GSIZE, 1.);
    let hsdf = op_onion(p.y - 0.5 * GSIZE, 0.5 * GSIZE, 1.);

    return linearstep(-1., 1., min(vsdf, hsdf));
}

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn modulo2(a: vec2f, b: vec2f) -> vec2f {
    return a - b * floor(a / b);
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return clamp((x - e0) / (e1 - e0), 0.0, 1.0);
}

fn rmat(t: f32) -> mat2x2f {
    let c = cos(t);
    let s = sin(t);

    return mat2x2f(c, s, -s, c);
}

fn o_ease_circ(x: f32) -> f32 {
    let t = 1 - x;
    return sqrt(1 - t * t);
}