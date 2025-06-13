@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    
    let size = vec2u(screen_size.x / 2, screen_size.y);

    let cell_id = id.xy / size;
    let cell_nm = cell_id.y * 2 + cell_id.x;

    let S = u32(custom.scale);

    let idxy = id.xy % size;
    let ssize = size / S;

    let frag_coord = vec2f(idxy / S) + vec2f(0.5) - vec2f(size / 2 / S);

    let alpha = 3.14 * fract(0.1 * time.elapsed);
    let a = sin(alpha);
    let b = cos(alpha);

    var d = 0.0;
    if cell_nm == 0 {
        d = cov_line(frag_coord, a, b, 0.);
    } else {
        d = sdf_line(frag_coord, a, b, 0.);
        d = linearstep(-0.5, 0.5, d);
    }

    var col = vec3f(d);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_line(p: vec2f, a: f32, b: f32, c: f32) -> f32 {
    return (a * p.x + b * p.y + c) / length(vec2f(a, b));
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

fn linearstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
}