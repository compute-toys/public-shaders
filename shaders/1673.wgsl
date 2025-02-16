const TWO_PI: f32 = 6.28318530718;
const COUNT: i32 = 100;


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / vec2f(screen_size);

    var closestDistance = length(vec2(1));
    var index = 0.0;

    for (var i = 0; i < COUNT; i++) {
        let x = rand(vec2(f32(i)) + vec2f(mouse.pos));
        let y = rand(vec2(f32(i)) + 0.5 + vec2f(mouse.pos));
        let pos = vec2(x, y);

        let dist = length(pos - uv);

        if (dist < closestDistance) {
            closestDistance = dist;
            index = f32(i);
        }
    }

    // Time varying pixel colour
    var col = vec3f(rand(vec2(index)));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn rand(co: vec2f) -> f32 {
    return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453123);
}