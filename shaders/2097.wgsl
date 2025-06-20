const PI: f32 = 3.1415926;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    var t = pingpong(time.elapsed, 1.2);
    t = smoothstep(0., 1.2, t);
    t = mix(-PI / 6., PI / 6., t);

    let rmat = mat_rotate(t);
    let smat = mat_scale(0.5, 1.);

    // Pixel coordinates (centre of pixel, origin at bottom left)
    var frag_coord = vec2f(id.xy) - 0.5 * vec2f(screen_size);
    frag_coord = rmat * smat * frag_coord;

    let gsize = custom.gsize;
    let hsize = 0.5 * gsize;

    var sdf_v = sdf_line(frag_coord, 1., 0., 0.);
    sdf_v = op_onion(sdf_v, gsize, hsize);
    var sdf_h = sdf_line(frag_coord, 0., 1., 0.);
    sdf_h = op_onion(sdf_h, gsize, hsize);

    let sdf_g = op_xor(sdf_v, sdf_h);
    let mask = linearstep(0.5, -0.5, sdf_g);

    var col = vec3f(clamp(mask, 0.1313, 1.0));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_line(p: vec2f, a: f32, b: f32, c: f32) -> f32 {
    return (a * p.x + b * p.y + c) / length(vec2f(a, b));
}

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return clamp((x - e0) / (e1 - e0), 0.0, 1.0);
}

fn op_onion(d: f32, l: f32, t: f32) -> f32 {
    return abs(modulo(d + l, 2. * l) - l) - t;
}

fn op_xor(d1: f32, d2: f32) -> f32 {
	return max(min(d1, d2), -max(d1, d2));
}

fn op_union(d1: f32, d2: f32) -> f32 {
    return min(d1, d2);
}

fn op_intersection(d1: f32, d2: f32) -> f32 {
    return max(d1, d2);
}

fn op_substraction(d1: f32, d2: f32) -> f32 {
    return max(d1, -d2);
}

fn mat_rotate(t: f32) -> mat2x2f {
    let c = cos(t);
    let s = sin(t);

    return mat2x2f(c, -s, s, c);
}

fn mat_scale(sx: f32, sy: f32) -> mat2x2f {
    return mat2x2f(sx, 0., 0., sy);
}

fn pingpong(x: f32, l: f32) -> f32 {
    return abs(modulo(x, 2. * l) - l);
}