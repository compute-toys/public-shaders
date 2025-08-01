const R: f32 = 128.0;
const PI: f32 = 3.1415926;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy) - 0.5 * vec2f(screen_size) + vec2f(0., 0.5 * custom.offset);
    let uv = vec2f(id.xy) / vec2f(screen_size);

    let gggg_coord = vec2f(id.xy);
    let noise = modulo(dot(gggg_coord .xyxy, sin(gggg_coord.yxyx)), 2.0);

    let t = sin(4. * time.elapsed + sin(uv.x * 10.1451 + cos(time.elapsed * 2.451 + uv.x * 4.)));

    let sigma = mix(4., custom.sigma, uv.y * (0.5 + 0.5 * cos(time.elapsed + 4. * sin(time.elapsed))));
    let gbr = blur_ring(frag_coord + vec2f(t * 8., uv.y * -custom.offset), sigma);
    let gbg = blur_ring(frag_coord + vec2f(0., uv.y * -custom.offset), sigma);
    let gbb = blur_ring(frag_coord + vec2f(-t * 8., uv.y * -custom.offset), sigma);

    var col = 1. - vec3f(gbr, gbg, gbb);
    col = pow(col, vec3f(4.));
    // grain
    col += (noise - 0.5) * custom.grain;
    
    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn blur_ring(p: vec2f, sigma: f32) -> f32 {
    const T: f32 = 0.25 * R;

    let sdf = op_shell(sdf_squircle(p, R, 2.), T);
    let gblur = gblur_box(-2. * T, 0., sigma, sdf);

    return gblur;
}

fn sdf_squircle(p: vec2f, r: f32, n: f32) -> f32 {
    var v = abs(p);
    v = select(v.yx, v.xy, v.x > v.y);
    let u = v.y / max(v.x, 0.001);

    return v.x * pow(1. + pow((u), n), 1. / n) - r;
}

fn op_shell(d: f32, t: f32) -> f32 {
    return abs(d) - t;
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

fn rmat(t: f32) -> mat2x2f {
    let c = cos(t);
    let s = sin(t);

    return mat2x2f(c, s, -s, c);
}