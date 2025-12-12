@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    
    let uv = fragCoord / vec2f(screen_size);
    let color = vec4f(uv.x, uv.y, 0, 1);
    
    textureStore(screen, id.xy, color);
}
