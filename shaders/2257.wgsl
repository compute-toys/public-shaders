const R: f32 = 128.0;
const H: f32 = 32.0;
const PI: f32 = 3.1415926;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy);

    let size = vec2f(0.5 * f32(screen_size.x), f32(screen_size.y));
    let cell_coord = frag_coord % size - 0.5 * size;

    var sdf = 0.0;
    var norm = vec3f(0.0);
    var h = 0.;
    if frag_coord.x / size.x < 1. {
        sdf = sdf_circle(cell_coord, R);
        h = button(-sdf, H);

        let dhdx = button(-sdf_circle(cell_coord + vec2f(1., 0.), R), H) - h;
        let dhdy = button(-sdf_circle(cell_coord + vec2f(0., 1.), R), H) - h;
        norm = normalize(vec3f(-dhdx, -dhdy, 1.));
    } else {
        let a = vec2f(-96., 0.);
        let b = -a;
        let t = 64.;
        sdf = sdf_segment(cell_coord, a, b, t);
        h = button(-sdf, H);

        let dhdx = button(-sdf_segment(cell_coord + vec2f(1., 0.), a, b, t), H) - h;
        let dhdy = button(-sdf_segment(cell_coord + vec2f(0., 1.), a, b, t), H) - h;
        norm = normalize(vec3f(-dhdx, -dhdy, 1.));
    }

    let mask = linearstep(1., -1., sdf);
    
    let t = time.elapsed * PI;
    let ldir = normalize(vec3f(cos(t), sin(t), 1.5));

    let diffuse = 0.5 * dot(ldir, norm) + 0.5;

    var col = mix(vec3f(0.5), vec3f(diffuse), mask);
    if mouse.click == 1 {
        col = mix(vec3f(0.5), norm, mask);
    }

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

// give a x, return the height of button
//
// h is the highest
fn button(x: f32, h: f32) -> f32 {
    var v = x * (2. * h - x);
    v = op_smax(v, 0., 32.);
    v = sqrt(v);
    return select(h, v, x < h);
}

// fn button(x: f32, h: f32) -> f32 {
//     let v = sqrt(x * (2. * h - x));
//     return select(0., select(h, v, x < h), x > 0.);
// }

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_segment(p: vec2f, a: vec2f, b: vec2f, t: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;

    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);

    return length(pa - ba * h) - t;
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return clamp((x - e0) / (e1 - e0), 0.0, 1.0);
}

fn op_smin(d1: f32, d2: f32, k: f32) -> f32 {
    let k4 = 4.0 * k;
    let h = max(k4 - abs(d1 - d2), 0.0) / k4;
    return min(d1, d2) - 0.25 * h * h * k4;
}

fn op_smax(d1: f32, d2: f32, k: f32) -> f32 {
    return -op_smin(-d1, -d2, k);
}
