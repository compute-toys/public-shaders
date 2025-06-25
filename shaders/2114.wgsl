@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = 2. * vec2f(f32(id.x), f32(screen_size.y - id.y)) - vec2f(screen_size);
    let uv = frag_coord / f32(screen_size.y) / 0.1;

    let a = min(uv.y, 3.33 * uv.y);
    let b = cos(uv.x + cos(uv.x * 0.6 - time.elapsed) + vec4f(0., .3, .6, 1.));
    let c = sin(uv.x * 0.4 - time.elapsed);

    var col = vec3f(tanh(.4 / abs(a / b.xyz / c)));

    let tex_size = vec2f(textureDimensions(channel0));
    let tuv = frag_coord / tex_size;
    
    var bn = textureSampleLevel(channel0, nearest_repeat, tuv, 0.).r;
    bn = pow(bn, 1.0 / 2.2);
    bn = uniform_to_triangle(bn);

    col = floor(col * custom.noise_scale + bn) / custom.noise_scale;

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn uniform_to_triangle(v: f32) -> f32 {
    var n = v * 2.0 - 1.0;
    n = sign(n) * (1.0 - sqrt(max(0.0, 1.0 - abs(n)))); // [-1, 1], max prevents NaNs
    // return n + 0.5; // [-0.5, 1.5]
    return n;
}