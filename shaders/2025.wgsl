const PI: f32 = radians(180.);

const COLOR_0: vec3f =vec3f(0.9, 0.2, 0.1);
const COLOR_1: vec3f = vec3f(0.92);
const COLOR_2: vec3f = vec3f(0.30, 0.60, 0.20);

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let size_f = vec2f(screen_size);
    let frag_coord = vec2f(id.xy) - 0.5 * size_f;

    var d = sdf_circle(frag_coord, custom.radius);
    d = normalize_range(-custom.range, 0., d); d = io_ease_quad(d);
    // d = smoothstep(-custom.range, 0., d);
    var col = mix(COLOR_0, COLOR_1, d);

    d = sdf_flower(frag_coord, 0.20 * custom.radius, 5);
    d = smoothstep(-8., 8., d);
    col = mix(COLOR_2, col, d);

    // grain
    col += (random12(frag_coord) - 0.5) * custom.grain_strength;

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
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