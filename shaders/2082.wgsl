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

    var col = vec3f(cov_line(frag_coord, a, b, 0.));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn int_line(p: vec2f, a: f32, b: f32, c: f32) -> f32 {
    let x1 = a * p.x + c;
    let y1 = b * p.y + c;
    let u = a * b;

    let v = select(0., 0.5 * (x1 * x1 + y1 * y1 - c * c) / u, abs(u) > 0.01) + p.x * p.y;

    let t = a * p.x + b * p.y + c;

    return select(0., v, t > 0.);
}

fn cov_line(p: vec2f, a: f32, b: f32, c: f32) -> f32 {
    return int_line(p + vec2f(.5, .5), a, b, c)
         + int_line(p - vec2f(.5, .5), a, b, c)
         - int_line(p + vec2f(-.5, .5), a, b, c)
         - int_line(p + vec2f(.5, -.5), a, b, c);
}