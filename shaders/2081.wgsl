@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let S = u32(custom.scale);
    let frag_coord = vec2f(id.xy / S) + vec2f(0.5);

    let a = sin(custom.alpha);
    let b = cos(custom.alpha);

    var col = vec3f(step(0., sip_line(frag_coord, a, b, 0.)));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sip_line(p: vec2f, a: f32, b: f32, c: f32) -> f32 {
    return a * p.x + b * p.y + c;
}