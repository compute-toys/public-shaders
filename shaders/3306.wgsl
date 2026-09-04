
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    if (id.x >= screen_size.x || id.y >= screen_size.y) {
        return;
    }

    let fragCoord = vec2f(
        f32(id.x) + 0.5,
        f32(id.y) + 0.5
    );

    let uv = fragCoord / vec2f(screen_size);


    let noise_texture_dim = textureDimensions(channel1);
    let coord = (vec2f(id.xy) + time.elapsed * 1000 ) % vec2f(noise_texture_dim.xy);

    let noise = textureLoad(channel1, vec2i(coord), 0);
    var color = textureSampleLevel(channel0, bilinear, uv, 0.0);

    let rand = u32(round((noise.r * 3.)));

    if rand == 0 {
        color.g = 0.01;
        color.b = 0.0;
    } else if rand == 1 {
        color.r = 0.0;
        color.b = 0.01;
    } else {
        color.r = 0.0;
        color.g = 0.0;
    } 

    textureStore(screen, id.xy, vec4f(color.rgb, 1.0));
}
