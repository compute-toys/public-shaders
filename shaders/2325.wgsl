const R: f32 = 128.0;
const PI: f32 = 3.1415926;

const COLOR_0: vec3f =vec3f(0.9, 0.2, 0.1);
const COLOR_1: vec3f = vec3f(0.92);

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy) - 0.5 * vec2f(screen_size);

    let gggg_coord = vec2f(id.xy / 2);
    
    let noise = modulo(dot(gggg_coord .xyxy, sin(gggg_coord.yxyx)), 2.0);

    let norm = vec2f(sqrt(2.)) / 2. * rmat(1.5 * time.elapsed);
    let leng = dot(norm, frag_coord);
    let rati = tanh(leng / (sqrt(2.) * R)) * 0.5 + 0.5;

    let sigma = rati * custom.sigma + 2.;

    var n = 1. / (1.65 * sigma / R) + 2.;
    n = min(n, 12.);
    let sdf = sdf_squircle(frag_coord, R, n);
    let gblur = gblur_box(-2. * R, 0., sigma, sdf);

    var col = mix(COLOR_1, COLOR_0, vec3f(gblur));
    // grain
    col += (noise - 0.5) * custom.grain;
    
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

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn uniform_to_triangle(v: f32) -> f32 {
    var n = v * 2.0 - 1.0;
    n = sign(n) * (1.0 - sqrt(max(0.0, 1.0 - abs(n)))); // [-1, 1], max prevents NaNs
    // return n + 0.5; // [-0.5, 1.5]
    return n;
}

fn rmat(t: f32) -> mat2x2f {
    let c = cos(t);
    let s = sin(t);

    return mat2x2f(c, s, -s, c);
}