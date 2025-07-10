const BAND_SIZE: f32 = 32.0;
const PI: f32 = 3.1415926;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let s = vec2f(custom.scale_x, custom.scale_y);
    let t = PI * custom.theta;
    let rmat = rmat(t);

    let frag_coord = vec2f(id.xy) - 0.5 * vec2f(screen_size);
    let local_coord = rmat * (s * frag_coord);

    let nb = nabla_band();

    let sdf = band(local_coord.x);
    // First LF: for AA
    let nabla = vec2f(
        dot(nb, s.x * vec2f(cos(t), sin(t))),
        dot(nb, s.y * vec2f(-sin(t), cos(t))),
    );
    var signal = linearstep(-length(nabla), length(nabla), sdf);
    // Second LF: anti moiré pattern
    //
    // ## How it works
    //
    // In local coordinates, the length of the pixel footprint is simply `s`.
    // (In the general case, you would calculate this using dpdx and dpdy).
    // When the pixel footprint length is greater than the band's wavelength, the average color is 0.5.
    //
    // What about more complex procedural textures?
    // 1. Guess an average color.
    // 2. Guess a maximum pixel step.
    // 3. Iterate until it looks good.
    let foot = rmat * (s * vec2f(1, 1));
    let fsize = abs(dot(nb, foot));   // footprint size
    let lod = linearstep(0.0, BAND_SIZE, fsize);
    signal = mix(signal, 0.5, lod);

    var col = vec3f(signal);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn band(d: f32) -> f32 {
    return abs(modulo(d, 2.0 * BAND_SIZE) - BAND_SIZE) - 0.5 * BAND_SIZE;
}

fn nabla_band() -> vec2f {
    return vec2f(1.0, 0.0);
}

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return clamp((x - e0) / (e1 - e0), 0.0, 1.0);
}

fn rmat(t: f32) -> mat2x2f {
    let c = cos(t);
    let s = sin(t);

    return mat2x2f(c, s, -s, c);
}