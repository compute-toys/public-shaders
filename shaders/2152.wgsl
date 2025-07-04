@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy);
    let uv = (2.0 * frag_coord - vec2f(screen_size)) / f32(screen_size.y);

    let sdf = 1.8 - length(uv - vec2f(1.0, -1.0));
    let glow = vec3f(1.0, 0.4, 0.2) / max(sdf, -sdf * 10.0);

    let noise = modulo(dot(frag_coord.xyxy, sin(frag_coord.yxyx)) + time.elapsed, 2.0);
    let twave = sin(time.elapsed + sin(time.elapsed / 0.6 + uv.y));
    let pattern = exp(noise + twave);

    var col = tanh(glow / pattern);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}