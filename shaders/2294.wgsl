// Here's the main idea:
// - we upload a texture of data.
// - the scene is composed of nested spheres an AABBs
// - we march the scene to compute the id od the object
// - for each pixel, we re-march the scene, but use the identified ID
// first, I'll implement a simple raytracer for 4 spheres.

fn sphere_sdf(point: vec3f, radius: f32) -> f32 {
    return length(point) - radius;
}

// march 10 steps
fn march(start: vec3f, dir: vec3f) -> vec3f {
    var point = start;
    for (var i = 0; i < 32; i++) {
        let d0 = sphere_sdf(point, 1.);
        let d1 = sphere_sdf(point + vec3(1), 1.5);
        let d2 = sphere_sdf(point + vec3(-1), 0.5);
        let d3 = sphere_sdf(point + vec3(0,1,1), 0.5);
        let dist = min(min(d0, d1), min(d2, d3));
        point += dir * dist;
    }
    return point;
}

// coord must by adjusted for aspect ratio
// longest dimension is 1
// generates ray
// assumes dir and up have length of 1
fn gen_ray(screen: vec2f, dir: vec3f, up: vec3f) -> vec3f {
    // cross dir with up to get horizontal
    let h = normalize(cross(dir, up));
    let v = cross(h, dir);
    // create the screen at the origin, move it
    var plane = h * screen.x + v * screen.y;
    plane += dir;
    // return normalized direction vector
    return normalize(plane);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from -1 to 1)
    var aspect = f32(screen_size.y) / f32(screen_size.x);
    var uv = fragCoord / vec2f(screen_size)*2.-1.;
    uv.y *= aspect;

    let camera = vec3(3, cos(time.elapsed), sin(time.elapsed));
    let dir = normalize(-camera);
    let ray = gen_ray(uv, dir, vec3(0, 0, 1));
    let marched = march(camera, ray);

    // Time varying pixel colour
    var col = vec3(marched.xyz);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
