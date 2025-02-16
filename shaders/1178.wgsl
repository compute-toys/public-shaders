
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    var uv = fragCoord / vec2f(screen_size);
    // Center pixel
    uv -= 0.5;
    // Correct Aspcet
    uv.x *= f32(screen_size.x) / f32(screen_size.y);
    // make a grid uv
    var guv = uv * custom.gridScale + 0.5;
    // Get the fraction of each grid cell
    var fuv = fract(guv);
    // Get the id of each grid cell
    var iduv = floor(guv);

    var col = vec3f(abs(iduv / custom.gridScale), 0.0);
    if (iduv.x == 0 && iduv.y == 0) {
      col = textureSampleLevel(channel0, nearest, fuv, 1).xyz;
      // col = textureSampleLevel(channel0, bilinear, fuv, 1).xyz;
    }

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
