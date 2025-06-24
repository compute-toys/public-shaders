const PI: f32 = 3.1415926;
const SLICE: f32 = 2.0 * PI / 6.;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy);

    let size = vec2f(0.5, 1.0) * vec2f(screen_size);
    let cell_coord = modulo2(frag_coord, size) - 0.5 * size;

    let sdf = select(
        map_iq(cell_coord, SLICE),
        map_mine(cell_coord, SLICE),
        frag_coord.x < size.x,
    );

    var col = select(vec3f(0.65,0.85,1.0), vec3f(0.9,0.6,0.3), sdf > 0.);
    col *= linearstep(-1., 1., op_shell(sdf, 3.));
    col *= clamp(linearstep(-2., 2., op_onion(sdf, 8., 4.)), 0.4, 1.0);
    col += linearstep(.5, -.5, op_shell(sdf, 1.));

    // middle line
    let lsdf = sdf_line(frag_coord, 1.0, 0.0, -size.x);
    let line_mask = linearstep(-1., 1., op_shell(lsdf, 2.));
    col *= line_mask;

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

// fn sdf_circle(p: vec2f, r: f32) -> f32 {
//     return length(p) - r;
// }

fn sdf_line(p: vec2f, a: f32, b: f32, c: f32) -> f32 {
    return (a * p.x + b * p.y + c) / length(vec2f(a, b));
}

fn sdf_box(p: vec2f, b: vec2f) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}

fn sdf(p: vec2f) -> f32 {
    // rotate + translate shape
    var coord = p - vec2f(128., 0.);
    coord = mat2x2(cos(time.elapsed), sin(time.elapsed), -sin(time.elapsed), cos(time.elapsed)) * coord;

    return sdf_box(coord, vec2f(50., 15.));
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

fn op_onion(d: f32, l: f32, t: f32) -> f32 {
    return abs(modulo(d, 2.0 * l) - l) - t;
}

fn op_shell(d: f32, t: f32) -> f32 {
    return abs(d) - t;
}

fn op_angular_repetition(p: vec2f, radian: f32) -> vec2f {
    let theta = atan2(p.y, p.x);
    let rotat = radian * round(theta / radian);

    let c = cos(rotat);
    let s = sin(rotat);

    // p * M is equal to M^T * p
    return p * mat2x2f(c, s, -s, c);
}

fn map_mine(p: vec2f, radian: f32) -> f32 {
    let theta = atan2(p.y, p.x);
    let rotat = radian * round(theta / radian);

    let c = cos(rotat);
    let s = sin(rotat);

    // p * M is equal to M^T * p
    let coord = p * mat2x2f(c, s, -s, c);

    return sdf(coord);
}

fn map_iq(p: vec2f, radian: f32) -> f32 {
    let theta = atan2(p.y, p.x);
    let i = floor(theta / radian);

    let c1 = radian * i;
    let c2 = radian * (i + 1.);

    let p1 = p * mat2x2f(cos(c1), sin(c1), -sin(c1), cos(c1));
    let p2 = p * mat2x2f(cos(c2), sin(c2), -sin(c2), cos(c2));

    return min(sdf(p1), sdf(p2));
}