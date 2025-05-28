@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let frag_coord = vec2f(id.xy);
    let frag_coord_c = frag_coord - 0.5 * vec2f(screen_size);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = frag_coord / f32(max(screen_size.x, screen_size.y));
   
    var value = sdf_circle(frag_coord_c, vec2f(0.0), custom.radius);
    // after blur
    value = smoothstep(-custom.blur_size, custom.blur_size, value);
    // after noise
    let noise = textureSampleLevel(channel0, bilinear_repeat, uv * custom.uv_scale, 0.).r;

    let hard_mix = hard_mix(value + noise, 1.0);

    var col = vec3f(hard_mix);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_circle(p: vec2f, c: vec2f, r: f32) -> f32 {
    return length(p - c) - r;
}

fn hard_mix(value: f32, thres: f32) -> f32 {
    return step(thres, value);
}