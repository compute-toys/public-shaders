const R: f32 = 196.0;
const PI: f32 = 3.1415926;
const WIDTH: f32 = 64.0;

// const COLOR_0: vec3f =vec3f(0.9, 0.2, 0.1);
const COLOR_0: vec3f = vec3f(21., 121., 235.) / 255.;
const COLOR_1: vec3f =vec3f(237., 66., 66.) / 255.;
const COLOR_2: vec3f = vec3f(0.88);

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy) - 0.5 * vec2f(screen_size);
    let mose_coord = vec2f(mouse.pos) - 0.5 * vec2f(screen_size);

    var t = 2. * abs(modulo(frag_coord.x, WIDTH) - 0.5 * WIDTH) / WIDTH;
    let rgb0 = mix(0.02, 0.96, t);
    let alpha0 = mix(0.35, 1.0, t);
    let blend = vec4f(vec3f(rgb0), alpha0);

    var sdf = sdf_circle(frag_coord - vec2f(-384., -196.), R);
    let gblur1 = saturate(gblur_box(-2.0 * R, 0.0, custom.sigma, sdf));

    let r = 0.75 * R;
    let n = 1. / (1.65 * custom.sigma / r) + 2.;
    t = op_onion(0.25 * time.elapsed, 1., 0.);
    t = io_ease_bounce(t);
    sdf = sdf_squircle((frag_coord - mose_coord) * rmat(2.0 * PI * t), r, n);
    let gblur2 = saturate(gblur_box(-2. * r, 0., custom.sigma, sdf));

    var rgb1 = mix(COLOR_2, COLOR_1, gblur2);
    rgb1 = mix(rgb1, COLOR_0, gblur1);
    let base = vec4f(rgb1, 1.0);

    let gggg_coord = vec2f(id.xy);
    let noise = modulo(dot(gggg_coord .xyxy, sin(gggg_coord.yxyx)), 2.0);

    var col = overlay(base, blend);
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

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn op_onion(d: f32, h: f32, t: f32) -> f32 {
    return abs(modulo(d, 2.0 * h) - h) - t;
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return saturate((x - e0) / (e1 - e0));
}

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
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

fn overlay(base: vec4f, blend: vec4f) -> vec3f {
    let grey = luminance(base.rgb);

    let screen = 1.0 - 2.0 * (1.0 - base.rgb) * (1.0 - blend.rgb);
    let multiply = 2.0 * base.rgb * blend.rgb;

    var rgb = mix(multiply, screen, vec3f(grey));
    rgb = mix(base.rgb, rgb, blend.a);

    return rgb;
}

/// @param rgb - 输入的线性 sRGB 颜色值 (vec3<f32>)
/// @returns a f32 value representing the perceived luminance.
fn luminance(rgb: vec3<f32>) -> f32 {
    // Rec.709 Luma coefficients for sRGB color space
    // Y = 0.2126*R + 0.7152*G + 0.0722*B
    let luma_weights = vec3<f32>(0.2126, 0.7152, 0.0722);
    return dot(rgb, luma_weights);
}

fn rmat(t: f32) -> mat2x2f {
    let c = cos(t);
    let s = sin(t);

    return mat2x2f(c, s, -s, c);
}

fn i_ease_bounce(x: f32) -> f32 {
    return 1 - o_ease_bounce(1 - x);
}

fn o_ease_bounce(x: f32) -> f32 {
    let a = 4.0 / 11.0;
    let b = 8.0 / 11.0;
    let c = 9.0 / 10.0;

    let ca = 4356.0 / 361.0;
    let cb = 35442.0 / 1805.0;
    let cc = 16061.0 / 1805.0;

    let t2 = x * x;

    return select(
      select(
        select(
          10.8 * x * x - 20.52 * x + 10.72,
          ca * t2 - cb * x + cc,
          x < c
        ),
        9.075 * t2 - 9.9 * x + 3.4,
        x < b
      ),
      7.5625 * t2,
      x < a
    );
}

fn io_ease_bounce(x: f32) -> f32 {
  return select(
    0.5 * o_ease_bounce(2 * x - 1) + 0.5,
    0.5 * (1.0 - o_ease_bounce(1 - x * 2)),
    x < 0.5
  );
}