const PI: f32 = 3.1415926;
const NUM_BLADES: u32 = 3;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy) - 0.5 * vec2f(screen_size) + vec2f(0.5);

    let R = 0.5 * 0.75 * f32(screen_size.y);

    var sdf = sdf_circle(frag_coord, R);
    sdf = op_sub(sdf, sdf + 0.85 * R);
    var mask = linearstep(1.0, -1.0, sdf);

    let theta = atan2(frag_coord.y, frag_coord.x);
    let radiu = length(frag_coord);
    
    let rotation = 2.0 * PI * fract(0.1 * time.elapsed);
    let delta = modulo(theta + rotation - 0.5 * radiu / R, 2.0 * PI / f32(NUM_BLADES));
    let triwv = sin(delta * f32(NUM_BLADES) * 0.5);
    mask *= pow(triwv, 6.);

    mask *= textureSampleLevel(channel0, bilinear_repeat, vec2f(radiu / R, fract(time.elapsed)), 0).r;

    let ctr = linearstep(1.0, -1.0, sdf_circle(frag_coord, 0.1 * R));
    let out = linearstep(-0.5, 0.5, 1. - abs(sdf));

    var col = vec3f(mask + 0.1 * out + 0.8 * ctr);

    // // Convert from gamma-encoded to linear colour space
    // col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn op_sub(a: f32, b: f32) -> f32 {
    return max(a, -b);
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return clamp((x - e0) / (e1 - e0), 0.0, 1.0);
}

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn rot2d_mat(t: f32) -> mat2x2f {
    let s = sin(t);
    let c = cos(t);
    
    return mat2x2f(c, s, -s, c);
}