const PI: f32 = radians(180.);

const COLOR_0: vec3f =vec3f(0.9, 0.2, 0.1);
const COLOR_1: vec3f = vec3f(0.30, 0.60, 0.20);

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let size_f = vec2f(screen_size);
    let frag_coord = vec2f(id.xy) - 0.5 * size_f;

    var col = vec3f(0.90);

    var layer0 = draw_persimmon(frag_coord, radius_elastic(time.elapsed), vec2f(0.1, 0.5));
    col = mix(col, layer0.rgb, layer0.a);
    layer0 = draw_persimmon(frag_coord + vec2f(-128., 150.), radius_elastic(time.elapsed + custom.step), vec2f(0.2, -0.3));
    col = mix(col, layer0.rgb, layer0.a);
    layer0 = draw_persimmon(frag_coord + vec2f(310., 80.), radius_elastic(time.elapsed + 2. * custom.step), vec2f(-0.2, -0.1));
    col = mix(col, layer0.rgb, layer0.a);
    layer0 = draw_persimmon(frag_coord + vec2f(-256., -140.), radius_elastic(time.elapsed + 3. * custom.step), vec2f(0.1, 0.4));
    col = mix(col, layer0.rgb, layer0.a);
    layer0 = draw_persimmon(frag_coord + vec2f(140., -150.), radius_elastic(time.elapsed + 4. * custom.step), vec2f(0., 0.3));
    col = mix(col, layer0.rgb, layer0.a);
    layer0 = draw_persimmon(frag_coord + vec2f(-320., 80.), radius_elastic(time.elapsed + 5. * custom.step), vec2f(0.1, 0.));
    col = mix(col, layer0.rgb, layer0.a);

    // grain
    col += (random12(frag_coord) - 0.5) * custom.grain_strength;

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn radius_elastic(elapsed: f32) -> f32 {
    let t = elapsed % custom.gap / custom.gap;
    let n = floor(elapsed / custom.gap);
    var rs = vec2f(custom.radius, 1.5 * custom.radius);
    rs = select(rs, rs.yx, n % 2. == 0.);

    return mix(rs.x, rs.y, o_ease_elastic(t));
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

// Pseudo-SDF: This function does not return a true signed distance.
// It produces a visual approximation of a flower shape for shading/thresholding,
// but the values are not proportional to Euclidean distance from the shape boundary.
fn sdf_flower(p: vec2f, r: f32, n: u32) -> f32 {
    let a = atan2(p.y, p.x);
    return length(p) - (0.5 * abs(cos(0.5 * f32(n) * a)) + 0.5) * r;
}

fn draw_persimmon(p: vec2f, r: f32, o: vec2f) -> vec4f {
    var d1 = sdf_circle(p, r);
    d1 = normalize_range(-0.5 * r, 0., d1); d1 = io_ease_quad(d1);
    var d2 = sdf_circle(p + r * o, 2. * r);
    d2 = normalize_range(-1.8 * r, 0., d2);

    let d = (1. - d1) * (1. - d2);
    var col = mix(vec3f(0.), COLOR_0, d);

    var d3 = sdf_flower(p + r * o, 0.20 * r, 5);
    d3 = smoothstep(-4., 4., d3);
    col = mix(COLOR_1, col, d3);

    return vec4f(col, max(d, 1. - d3));
}

fn normalize_range(a: f32, b: f32, x: f32) -> f32 {
    return clamp((x - a) / (b - a), 0., 1.);
}

fn i_ease_quad(x: f32) -> f32 {
    return x * x;
}

fn o_ease_quad(x: f32) -> f32 {
    let t = 1 - x;
    return 1 - t * t;
}

fn io_ease_quad(x: f32) -> f32 {
    return select(
        1 - 2 * pow(1 - x, 2),
         2 * x * x,
        x < 0.5,
    );
}

const C4: f32 = 0.6667 * PI;
const C5: f32 = 0.4444 * PI;

fn i_ease_elastic(x: f32) -> f32 {
    return  select(
        step(0., x),
        -pow(2, 10 * x - 10) * sin((10 * x - 10.75) * C4),
        x >= 0 && x <= 1,
    );
}

fn o_ease_elastic(x: f32) -> f32 {
    return  select(
        step(0., x),
        pow(2, -10 * x) * sin((10 * x - 0.75) * C4) + 1,
        x >= 0 && x <= 1,
    );
}

fn io_ease_elastic(x: f32) -> f32 {
    return select(
        step(0., x),
        select(
         0.5 * pow(2, -20 * x + 10) * sin((20 * x - 11.125) * C5) + 1,
        -0.5 * pow(2, 20 * x - 10) * sin((20 * x - 11.125) * C5),
        x < 0.5,
        ),
        x >= 0 && x <= 1,
    );
}

const RANDOM_SINLESS: bool = true;
const RANDOM_SCALE: vec4f = vec4f(.1031, .1030, .0973, .1099);

fn random12(st: vec2f) -> f32 {
    if (RANDOM_SINLESS) {
        var p3  = fract(vec3(st.xyx) * RANDOM_SCALE.xyz);
        p3 += dot(p3, p3.yzx + 33.33);
        return fract((p3.x + p3.y) * p3.z);
    } else {
        return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453);
    }
}