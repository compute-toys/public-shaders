const PI: f32 = 3.1415926;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy);

    let orig_coord = custom.scale * frag_coord;

    let _dpdx = abs(custom.scale * (frag_coord + vec2f(1.0, 0.0)) - orig_coord).x;

    let hwave = 0.5 * custom.lambda;

    let vsdf = op_onion(sdf_line(orig_coord, 1.0, 0.0, 0.0), hwave, 0.5 * hwave);
    let alph = linearstep(0., hwave, _dpdx);

    var band = linearstep(-(_dpdx), _dpdx, vsdf);
    band = select(band, mix(band, 0.5, alph), frag_coord.y > 0.5 * f32(screen_size.y));

    var col = vec3f(band);

    // draw a red line
    let lsdf = op_shell(sdf_line(frag_coord, 0.0, 1.0, -0.5 * f32(screen_size.y)), 2.);
    let line = linearstep(-0.5, 0.5, lsdf);
    col = mix(vec3f(1.0, 0.0, 0.0), col, line);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_line(p: vec2f, a: f32, b: f32, c: f32) -> f32 {
    return (a * p.x + b * p.y + c) / length(vec2f(a, b));
}

fn op_onion(d: f32, l: f32, t: f32) -> f32 {
    return abs(modulo(d, 2.0 * l) - l) - t;
}

fn op_shell(d: f32, t: f32) -> f32 {
    return abs(d) - t;
}

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return clamp((x - e0) / (e1 - e0), 0.0, 1.0);
}