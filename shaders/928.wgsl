
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(id.xy);
    let uv = fragCoord / vec2f(screen_size);

    let tex_siz = vec2f(textureDimensions(channel0));
    let tex_uv = modulo(fragCoord, tex_siz) / tex_siz;

    var bn = textureSampleLevel(channel0, nearest, tex_uv, 0).r;
    
    // fix wrong distribution
    if mouse.click == 1 {
        bn = pow(bn, 1. / 2.2);
    }

    var less_than_half = f32(bn <= 0.5);
    var more_than_half = f32(bn >= 0.5);

    let left_or_right = step(0.5, uv.x);

    // draw noise
    var col =vec3f(mix(less_than_half, more_than_half, left_or_right));
    // draw line
    col -= step(abs(uv.x - 0.5), 0.002);

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}


fn modulo(x: vec2f, y: vec2f) -> vec2f {
    return x - y * floor(x / y);
}