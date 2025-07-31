const PI: f32 = 3.1415926;
const NMAX: f32 = 16.;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let size = vec2f(0.5, 1.0) * vec2f(screen_size);
    let cell_size = vec2f(id.xy) % size - 0.5 * size;
    let cell_id = floor(f32(id.x) / size.x);

    // var n = 2. + (NMAX - 2.) * exp2(-8. * custom.sigma / custom.r);
    var n = 1. / (1.65 * custom.sigma / custom.r) + 2.;
    n = select(NMAX, min(n, NMAX), cell_id > 0.);

    let sdf = sdf_squircle(cell_size, custom.r, n);
    let gblur = gblur_box(-2. * custom.r, 0., custom.sigma, sdf);

    var col = vec3f(gblur);
    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_squircle(p: vec2f, r: f32, n: f32) -> f32 {
    var v = abs(p);
    v = select(v.yx, v.xy, v.x > v.y);
    let u = v.y / max(v.x, 0.001);

    return v.x * pow(1. + pow((u), n), 1. / n) - r;
}

fn erf7(x: f32) -> f32 {
    const C0: f32 = 2. / sqrt(PI);
    const C1: f32 = 0.24295;
    const C2: f32 = 0.03395;
    const C3: f32 = 0.01040;

    let x1 = x * C0;
    let x2 = x1 * x1;
    let x3 = x1 + (C1 + (C2 + C3 * x2) * x2) * (x1 * x2);

    return x3 / sqrt(1. + x3 * x3);
}

fn gblur_box(e0: f32, e1: f32, sigma: f32, x: f32) -> f32 {
    const V: f32 = inverseSqrt(2);

    let v = V / sigma;

    return 0.5 * (erf7((x - e0) * v) - erf7((x - e1) * v));
}