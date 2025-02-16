
const IDT: mat3x3f = mat3x3f(
    0.61313242, 0.07012438, 0.02058766,
    0.33953802, 0.91639401, 0.10957457,
    0.04741670, 0.01345152, 0.86978540
);

const ODT: mat3x3f = mat3x3f(
    1.70485868, -0.13007682, -0.02396407,
    -0.62171602, 1.14073577, -0.12897551,
    -0.08329937, -0.01055980, 1.15301402
);

const SHADES: f32 = 256.;
const SCATTER: f32 = 1.;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(id.xy) - 0.5 * vec2f(screen_size);

    let uv = fragCoord / f32(screen_size.y);

    let centre = vec2f(0.5 * sin(2. * time.elapsed), 0.25 * cos(time.elapsed));

    // distance to outside of bulb
    let d = max(sdf_circle(uv - centre, custom.bulb_rad), 0.);
    let b = custom.brightness_adj / (custom.max_brightness_inv + d);

    // Color in gamma space
    var col = vec3f(custom.r, custom.g, custom.b);
    // To linear space
    col = pow(col, vec3f(2.2));
    // To ACEScg space
    col = IDT * col;
    // Tonemapping
    col = tonemap_aces(col * b);
    // To linear space
    col = ODT * col;

    // Sample blue noise
    let tex_size = vec2f(textureDimensions(channel0).xy);
    let nuv = modulo(fragCoord, tex_size) / tex_size;
    var bn = textureSampleLevel(channel0, nearest, nuv, 0).rgb;
    bn = uniform_to_triangle(bn);

    let shades = SHADES * custom.shades;
    let scatter = SCATTER / clamp(d, 0.02, 1.) * custom.scatter;

    // col = floor(col * shades + bn * scatter) / shades;
    col = saturate(col + bn / shades * scatter);

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_square(p: vec2f, hs: f32) -> f32 {
    let d = abs(p) - vec2f(hs);

    return length(max(d, vec2f(0))) + min(max(d.x, d.y), 0);
}

fn tonemap_aces(x: vec3f) -> vec3f {
    let a = x * (x + 0.0245786) - 0.000090537;
    let b = x * (0.983729 * x + 0.4329510) + 0.238081;

    return clamp(a / b, vec3f(0.), vec3f(1.));
}

fn modulo(x: vec2f, y: vec2f) -> vec2f {
    return x - y * floor(x / y);
}

fn uniform_to_triangle(v: vec3f) -> vec3f {
    var n = v * 2.0 - 1.0;
    n = sign(n) * (1.0 - sqrt(max(vec3f(0.), 1. - abs(n)))); // [-1, 1], max prevents NaNs
    return n + 0.5; // [-0.5, 1.5]
}