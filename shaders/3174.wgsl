const max_speed = 1000.;
const render_dist = 1500.;
const epsilon = 0.01;
const sun_dir = normalize(vec3f(-1., 2., -0.5));
const sky_col = vec3f(0.1, 0.3, 1.);

fn rampup(t: float) -> float {
    if(t < 1) {
        return 0.5 * t * t;
    } else {
        return t - 0.5;
    }
}

fn cam_depth(time: float) -> float {
    return rampup(time / 10.) * 10. * max_speed;
}

// https://compute.toys/view/15
fn hash33(p: float3) -> float3 {
    var p3 = fract(p * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return fract((p3.xxy + p3.yxx)*p3.zyx);
}

struct RayResult {
    hit: bool,
    pos: vec3f,
    len: float,
    steps: i32,
}

fn pos_to_cell(pos: vec3f) -> vec3f {
    var p = pos;
    p.y = min(p.y, 1000.);
    return vec3f(round(p.x / 50.), round((p.y - 5.) / 100.), round(p.z / 50.));
}

fn sdf(pos: vec3f) -> float {
    var dist = 10.;
    dist = min(dist, pos.y);
    let cell = pos_to_cell(pos);
    let wrap_pos = vec3f(pos.x - cell.x * 50., pos.y - cell.y * 100. - 5., pos.z -  cell.z * 50.);
    //let offset = textureLoad(channel1, vec2u(uint(cell.x), uint(cell.z)) % 256, 0) - 0.5;
    let offset = hash33(cell) - 0.5;
    dist = min(dist, length(wrap_pos - vec3f(offset.x * 10., offset.y * 10., offset.z * 10.)) - 5.);
    return dist;
}

fn sdf_gradient(pos: vec3f) -> vec3f {
    let base = sdf(pos);
    let dx = (sdf(pos + vec3f(epsilon, 0., 0.)) - base) / epsilon;
    let dy = (sdf(pos + vec3f(0., epsilon, 0.)) - base) / epsilon;
    let dz = (sdf(pos + vec3f(0., 0., epsilon)) - base) / epsilon;
    return normalize(vec3f(dx, dy, dz));
}

fn cast_ray(start_pos: vec3f, ray_dir: vec3f) -> RayResult {
    var ray_pos = start_pos;
    var dist = sdf(ray_pos);
    var ray_len = 0.;
    var steps = 0;
    while(dist > epsilon && ray_len < render_dist && steps < 1000) {
        steps++;
        ray_len += dist;
        ray_pos = start_pos + ray_dir * ray_len;
        dist = sdf(ray_pos);
    }
    return RayResult(dist <= epsilon, ray_pos, ray_len, steps);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let noise = pow(textureLoad(channel0, id.xy % 1024, 0).r, 1/2.2) - 1.;

    let pixel = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5) - vec2f(screen_size) / 2.;

    let ray_dir = normalize(vec3f(pixel.xy / float(screen_size.y), 0.7));
    let depth = cam_depth(time.elapsed);
    let prev_depth = cam_depth(time.elapsed - time.delta);
    let cam_pos = vec3f(0., 150., mix(depth, prev_depth, noise));
    let result = cast_ray(cam_pos, ray_dir);
    var col = sky_col;
    if(result.hit) {
        let normal = sdf_gradient(result.pos);
        col = vec3f(100. / float(result.steps + 100));
        if(all(normal != vec3f(0., 1., 0.))) {
            let cell = pos_to_cell(result.pos);
            col *= pow(hash33(cell + vec3f(0., 100., 0.)), vec3f(2.2));
        } else {
            col *= textureLoad(channel1, vec2u(uint(result.pos.x + 512.), uint(result.pos.z)) % 1024, 0).rgb;
        }
        let light = max(0., dot(normal, sun_dir)) * 0.9 + 0.1;
        col *= light;
        let fog = sqrt(1. - result.len / render_dist);
        col = mix(sky_col, col, fog);
    }

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
