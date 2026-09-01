// Helper for per-channel overlay calculation
fn channel_overlay(base: f32, blend: f32) -> f32 {
    if (base < 0.5) {
        return 2.0 * base * blend;
    } else {
        return 1.0 - 2.0 * (1.0 - base) * (1.0 - blend);
    }
}

// Vectorized overlay blend mode function
fn blend_overlay(base: vec3<f32>, blend: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        channel_overlay(base.r, blend.r),
        channel_overlay(base.g, blend.g),
        channel_overlay(base.b, blend.b)
    );
}


fn rand22(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

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
    let coord = vec2i(id.xy % noise_texture_dim.xy);

    let noise = textureLoad(channel1, coord, 0);
    let color = textureSampleLevel(channel0, bilinear, uv, 0.0);

    // The color determines the density.
    // The noise determines which pixels are selected.
    var red   = select(0.0, 1.0, noise.r < color.r);
    var green = select(0.0, 1.0, noise.g < color.g);
    var blue  = select(0.0, 1.0, noise.b < color.b);

    let rand = u32(rand22(uv + vec2f(time.elapsed)) * 3);

    if rand == 0 {
        green = 0.0;
        blue = 0.0;
    } else if rand == 1 {
        red = 0.;
        blue = 0.;
    } else {
        red = 0.;
        green =0.;
    }

    textureStore(screen, id.xy, vec4f(red, green, blue, 1.0));
}
