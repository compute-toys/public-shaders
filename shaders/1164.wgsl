const PI = 3.1423;

fn uvToRayDirection(uv: vec2<f32>) -> vec3<f32> {
    // Convert UV coordinates to spherical coordinates
    let azimuth = uv.x * 2.0 * PI; // Azimuth angle (0 to 2*pi)
    // let elevation = (uv.y - 0.5) * PI; // Elevation angle (-pi/2 to pi/2)
    let elevation = uv.y * PI / 2.0; // Elevation angle (0 to pi/2)

    // Convert spherical coordinates to Cartesian coordinates
    let x = cos(elevation) * cos(azimuth);
    let y = sin(elevation);
    let z = cos(elevation) * sin(azimuth);

    return normalize(vec3(x, y, z));
}

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

    // Time varying pixel colour
    var rayDir = uvToRayDirection(uv);

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4(rayDir,1));
}
