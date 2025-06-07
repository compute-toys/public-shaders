const COUNT: vec2i = vec2i(4, 3);

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
    let cell_size = vec2i(screen_size) / COUNT;

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let frag_coord = vec2f(id.xy);

    let cell_id = vec2i(id.xy) / cell_size;
    let cell_coord = vec2i(id.xy) % cell_size;

    var d = 1000.;
    for (var x = 0; x < COUNT.x; x++) {
        let d_ = sdf_line_t(frag_coord, 1., 0., -(f32(x) + 1.) * f32(cell_size.x), 1.);
        d = min(d, d_);
    }
    for (var y = 0; y < COUNT.y; y++) {
        let d_ = sdf_line_t(frag_coord, 0., 1., -(f32(y) + 1.) * f32(cell_size.y), 1.);
        d = min(d, d_);
    }

    let cell_uv = vec2f(cell_coord) / f32(cell_size.y);
    
    var col = vec3f(1.);
    var uv: vec2f;
    if cell_id.y == 0 {
        if cell_id.x == 0 {
            uv = vec2f(cell_uv.x, 0.);
        } else if cell_id.x == 1 {
            uv = vec2f(cell_uv.x, floor(time.elapsed) * 0.025);
        } else if cell_id.x == 2 {
            uv = vec2f(cell_uv.x, fract(time.elapsed) * 0.5);
        } else if cell_id.x == 3 {
            uv = vec2f(cell_uv.x - fract(time.elapsed), 0.);
        }
        col = textureSampleLevel(channel0, bilinear_repeat, uv, 0).rrr; 
    } else if cell_id.y == 1 {
        if cell_id.x == 0 {
            uv = vec2f(cell_uv.x, 0.);
        } else if cell_id.x == 1 {
            uv = vec2f(cell_uv.x, floor(time.elapsed) * 0.025);
        } else if cell_id.x == 2 {
            uv = vec2f(cell_uv.x, fract(time.elapsed) * 0.5);
        } else if cell_id.x == 3 {
            uv = vec2f(cell_uv.x - fract(time.elapsed), 0.);
        }
        col = textureSampleLevel(channel0, bilinear_repeat, uv, 0).rgb; 
    } else if cell_id.y == 2 {
        let cell_uv_t = vec2f(cell_coord - cell_size / 2) / f32(cell_size.y);
        let l = length(cell_uv_t);
        if cell_id.x == 0 {
            uv = vec2f(l, 0.);
        } else if cell_id.x == 1 {
            uv = vec2f(l, floor(time.elapsed) * 0.025);
        } else if cell_id.x == 2 {
            uv = vec2f(l, fract(time.elapsed) * 0.5);
        } else if cell_id.x ==3 {
            uv = vec2f(l - fract(time.elapsed), 0.);
        }
        col = textureSampleLevel(channel0, bilinear_repeat, uv, 0).rgb; 
    }

    col = col * clamp(d, 0., 1.);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_line(p: vec2f, a: f32, b: f32, c: f32) -> f32 {
    return (a * p.x + b * p.y + c) / sqrt(a * a + b * b);
}

fn sdf_line_t(p: vec2f, a: f32, b: f32, c: f32, ht: f32) -> f32 {
    return abs(sdf_line(p, a, b, c)) - ht;
}