
#include "Dave_Hoskins/hash"

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(id.xy);
    // for sampling bayer matrix texture
    let bayer_uv = vec2f(id.xy % 8) / 8.;

    let uv = (2.0 * fragCoord - vec2f(screen_size)) / f32(screen_size.x);

    // let t_bias = hash12(fragCoord) / 60.0;
    let t_bias = textureSampleLevel(channel0, nearest, bayer_uv, 0).x / 60.;
    let val = sample_scene(uv, time.elapsed + t_bias);

    textureStore(screen, id.xy, vec4f(vec3f(val), 1.));
}

fn sample_scene(uv: vec2f, t: f32) -> f32 {
    const EDGES: f32 = 0.004;

    let speed = custom.speed * 2.;
    let radiu = custom.radius;
    
    let centre = vec2f(sin(speed * 5.0 * t), sin(speed * 6. * t)) * 0.4;
    let dist = radiu - length(uv - centre);

    return smoothstep(0, EDGES, dist);
}
