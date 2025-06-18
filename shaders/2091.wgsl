const PI: f32 = 3.1415926;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let frag_coord = vec2f(id.xy);

    // Normalised pixel coordinates (from 0 to 1)
    var uv = (2. * frag_coord - vec2f(screen_size)) / f32(screen_size.y);
    let uv0 = uv;

    var fcol = vec3f(0.0);
    for (var i = 0; i < 4; i++) {
        uv = fract(custom.t * uv) - 0.5;
        
        var d = length(uv) * exp(-length(uv0));

        let col = palette(length(uv0) + f32(i) / 3. + fract(time.elapsed));

        d = sin(8. * d + time.elapsed);
        d = abs(d);
        d = 0.1 / (d + 0.05);

        fcol += pow(d, 2.) * col;
    }

    // Convert from gamma-encoded to linear colour space
    fcol = pow(fcol, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(fcol, 1.));
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return clamp((x - e0) / (e1 - e0), 0.0, 1.0);
}

fn modulo2(a: vec2f, b: vec2f) -> vec2f {
    return a - b * floor(a / b);
}

fn palette(t: f32) -> vec3f {
    return 0.5 + 0.5 * cos(2. * PI * (t + vec3f(0.263, 0.416, 0.557)));
}