const R: f32 = 96.0;
const PI: f32 = 3.1415926;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy) - 0.5 * vec2f(screen_size);
    let mose_coord = vec2f(mouse.pos) - 0.5 * vec2f(screen_size);

    let bc1 = blur_circle(frag_coord, R, custom.sigma);
    let bc2 = blur_circle(frag_coord - mose_coord, R, custom.sigma);

    let ctre_coord = vec2f(cos(time.elapsed), sin(time.elapsed)) * 128.;
    let bc3 = blur_circle(frag_coord - ctre_coord, R, custom.sigma);

    // let u = 1. - (1. - vec3f(bc1, 0., 0.)) * (1. - vec3f(0., bc2, 0.)) * (1. - vec3f(0., 0., bc3));
    let u = 1. - (1. - bc1) * (1. - bc2) * (1. - bc3);
    var col = vec3f(u);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn blur_circle(p: vec2f, r: f32, sigma: f32) -> f32 {
    let l = length(p);

    let g = convolution(l + r, sigma);
    let v = convolution(l - r, sigma);

    return g - v;
}

fn erf(x: f32) -> f32 {
    // return  x / sqrt(1. + x * x);
    // return tanh(x);

    const C0: f32 = 2. / sqrt(PI);
    const C1: f32 = 0.24295;
    const C2: f32 = 0.03395;
    const C3: f32 = 0.01040;

    let x1 = x * C0;
    let x2 = x1 * x1;
    let x3 = x1 + (C1 + (C2 + C3 * x2) * x2) * (x1 * x2);

    return x3 / sqrt(1. + x3 * x3);
}

// The convolution between __step function__ and __gaussian distribution function__
fn convolution(x: f32, sigma: f32) -> f32 {
    const V: f32 = 1. / sqrt(2);

    return 0.5 * (1.0 + erf(x * V / sigma));
}