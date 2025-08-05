const C0: vec3f = vec3f(0.0, 125., 255.) / 255.;
const C1: vec3f = vec3f(255., 125., 0.) / 255.;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(f32(id.x), f32(screen_size.y) - f32(id.y)) - 0.5 * vec2f(screen_size);
    var uv = frag_coord / f32(screen_size.y);

    let f = flame(uv);

    var col = tanh(f * 8.);
    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn flame(uv: vec2f) -> vec3f {
    // scale uv
    let uv0 = uv * vec2f(10., 1.3);
    // scale and offset
    let uv1 = uv0 * 0.02 - time.elapsed * vec2f(0.02, 0.05);
    // sample texture
    let random = textureSampleLevel(channel0, bilinear_repeat, uv1, 0).r;
    // y gradient
    let y = smoothstep(-0.4, 0.4, uv.y);

    let uv2 = uv0 + random * y * vec2f(0.7, 1.3);
    // out flame(circle)
    let c = mix(C0, C1, smoothstep(-0.6, 0.15, uv.y));
    var f = smoothstep(-0.2, 0.0, 0.4 - length(uv2)) * c;
    // inner flame(circle)
    f *= smoothstep(0.1, 1.0, length(uv2 * vec2f(1.0, 0.6) + vec2f(0.0, 0.35)));

    return f;
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