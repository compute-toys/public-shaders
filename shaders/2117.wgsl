const PI: f32 = 3.1415926;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let size = vec2f(screen_size);
    let frag_coord = vec2f(id.xy) - 0.5 * size;

    let hypr_coord = hyper_space(frag_coord, size);

    let dpdx_ = hyper_space(frag_coord + vec2f(1., 0.), size) - hypr_coord;
    let dpdy_ = hyper_space(frag_coord + vec2f(0., 1.), size) - hypr_coord;

    let fwid_ = vec2f(length(vec2f(dpdx_.x, dpdy_.x)), length(vec2f(dpdx_.w, dpdy_.w)));

    let hwave = 0.5 * vec2f(custom.r_lambda, 1. / floor(custom.t_frequency));
    let afade = linearstep2(vec2f(0.), hwave, fwid_);

    let sdf = op_onion2(hypr_coord.xy, hwave, 0.5 * hwave);
    var bnd = mix(linearstep2(-fwid_, fwid_, sdf), vec2f(0.5), afade);
    let pattern = op_xor(bnd.x, bnd.y);

    let ambient = hypr_coord.z / size.y;
    
    var col = vec3f(pattern * ambient);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn op_onion2(d: vec2f, l: vec2f, t: vec2f) -> vec2f {
    return abs(modulo2(d, 2.0 * l) - l) - t;
}

// fn op_xor(d1: f32, d2: f32) -> f32 {
//     return max(min(d1, d2), -max(d1, d2));
// }

fn op_xor(a: f32, b: f32) -> f32 {
    return a + b - 2.0 * a * b;
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

fn linearstep2(e0: vec2f, e1: vec2f, x: vec2f) -> vec2f {
    return clamp((x - e0) / (e1 - e0), vec2f(0), vec2f(1));
}

fn hyper_space(p: vec2f, size: vec2f) -> vec4f {
    let is_square = custom.square > 0.5;
    
    var r = length(p);
    if is_square {
        let p8 = pow(abs(p), vec2f(8.));
        r = pow(p8.x + p8.y, 1. / 8.);
    }
    let i = size.y * custom.r_lambda / r + 2. * custom.r_lambda * fract(time.elapsed);
    let t = atan2(p.y, p.x) / (2.0 * PI);

    return vec4f(i, t + 0.1 * sin(time.elapsed) * r / size.y - 0.2 * fract(time.elapsed), r, abs(t));
}