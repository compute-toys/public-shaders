@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = 2.0 * vec2f(f32(id.x), f32(screen_size.y - id.y))
        - vec2f(screen_size);

    var uv = frag_coord / f32(screen_size.y);
    uv /= 0.9;

    let circ = length(uv) - 1.0;
    let glow = 0.1 / max(circ / 0.1, -circ);

    let distortion = max(1.0, -circ / 0.1);
    let phase = circ + uv.y * distortion + time.elapsed + vec3f(0.0, 1.0, 2.0);

    var pattern = glow - sin(phase);
    pattern = tanh(pattern);

    var col = 0.5 + 0.5 * pattern;

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
