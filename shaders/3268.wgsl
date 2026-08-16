const shr = 1u; // reduction amount to see pixels better

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen) >> vec2u(shr);

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x >> shr) + .5, f32(screen_size.y - (id.y >> shr)) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / vec2f(screen_size);

    var dither : f32;
    if (mouse.click == 0) {
        dither = textureLoad(channel0, (id.xy >> vec2u(shr)) & vec2u(127), 0).x;
    } else {
        dither = textureLoad(channel1, (id.xy >> vec2u(shr)) & vec2u(1023), 0).x;
    }

    dither = pow(dither, 1./2.2); // how to specify unorm texture format to avoid this?

    var col = vec3f(step(uv.x, dither));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
