const R: f32 = 128.0;
const H: f32 = 32.0;
const L: f32 = R;
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
    var ndir = vec3f(0.0);
    var h = 0.;
    if frag_coord.x / size.x < 1. {
        sdf = sdf_circle(cell_coord, R);
        h = droplet(-sdf, H, L);

        let dhdx = droplet(-sdf_circle(cell_coord + vec2f(1., 0.), R), H, L) - h;
        let dhdy = droplet(-sdf_circle(cell_coord + vec2f(0., 1.), R), H, L) - h;
        ndir = normalize(vec3f(-dhdx, -dhdy, 1.));
    } else {
        let a = vec2f(-64., 0.);
        let b = -a;
        let t = 96.;
        sdf = sdf_segment(cell_coord, a, b, t);
        h = droplet(-sdf, H, t);

        let dhdx = droplet(-sdf_segment(cell_coord + vec2f(1., 0.), a, b, t), H, t) - h;
        let dhdy = droplet(-sdf_segment(cell_coord + vec2f(0., 1.), a, b, t), H, t) - h;
        ndir = normalize(vec3f(-dhdx, -dhdy, 1.));
    }

    let mask = linearstep(1., -1., sdf);
    
    let t = time.elapsed;
    let ldir = normalize(vec3f(cos(t), sin(t), custom.light_height));
    let vdir = vec3f(0., 0., 1.);
    let hdir = normalize(ldir + vdir);

    let diffuse = 0.5 * dot(ldir, ndir) + 0.5;
    let specular = pow(max(dot(ndir, hdir), 0.), custom.shininess);

    var col = mix(vec3f(0.5), vec3f(diffuse + specular), mask);
    if mouse.click == 1 {
        col = mix(vec3f(0.5), ndir, mask);
    }
    col = tanh(col);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn droplet(x: f32, h: f32, l: f32) -> f32 {
    let t = h / l;
    var v = t * x * (2. * h - t * x);
    v = sqrt(op_smax(v, 0., 8.));
    return select(h, v, x < l);
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_segment(p: vec2f, a: vec2f, b: vec2f, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;

    var h = smoothstep4(0., dot(ba, ba), dot(pa, ba));
    if mouse.click == 2 {
        h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    }

    return length(pa - ba * h) - r;
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

fn smoothstep2(e0: f32, e1: f32, x: f32) -> f32 {
    const K: f32 = 8.;

    let r = (x - e0) / (e1 - e0);

    var v = 1. + exp2(K * r);
    v /= 1. + exp2(K * (r - 1.));
    v = log2(v) / K;

    return v;
}

fn smoothstep3(e0: f32, e1: f32, x: f32) -> f32 {
    const K: f32 = 0.100;

    let r = (x - e0) / (e1 - e0);

    return op_smax(op_smin(r, 1., K), 0., K);
}

fn smoothstep4(e0: f32, e1: f32, x: f32) -> f32 {
    const K: f32 = 0.100;
    
    let r = (x - e0) / (e1 - e0) - 0.5;
    var v = select(-1., 1., r > 0.) * op_smin(abs(r), 0.5, K) + 0.5;
    // NOTE: op_smin(0, 0.5, K) could be a constant!
    v += 2. * op_smin(0, 0.5, K) * step(r, 0.);
        
    return v;
}