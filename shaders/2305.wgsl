const PI: f32 = 3.1415926;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy) - 0.5 * vec2f(screen_size);

    var sdf = length(frag_coord);
    sdf = op_onion(sdf, custom.nabla);
    let shape = linearstep(1., -1., sdf);
    let rounded = sin(0.5 * PI * sdf / (0.25 * custom.nabla));
    let blur = 0.5 + 0.5 * erf7(rounded / (9.5 * custom.sigma / custom.nabla));

    var col = vec3f(blur);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn op_onion(d: f32, n: f32) -> f32 {
    return abs(modulo(d - 0.25 * n, n) - 0.5 * n) - 0.25 * n;
}

fn modulo(a: f32, b: f32) -> f32 {
    return fma(-b, floor(a / b), a);
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return saturate((x - e0) / (e1 - e0));
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