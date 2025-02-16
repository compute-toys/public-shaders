enable f16;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);
    var col = .5 + .5 * cos(time.elapsed + uv.xyx + vec3f(0.,2.,4.));

    // f16 pow function
    col = vec3f(pow(vec3h(col), vec3h(2.2)));

    textureStore(screen, id.xy, vec4f(col, 1.));
}
