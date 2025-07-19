const R: f32 = 128.0;
const H: f32 = 48.0;
const L: f32 = R;
const PI: f32 = 3.1415926;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy) - 0.5 * vec2f(screen_size);

    let t = 0.5 * sin(2.0 * PI * time.elapsed) + 0.5;
    let pa = vec2f(-128., 0.);
    let sdf = sdf_fusion(frag_coord, R, pa, -pa, t);
    let h = droplet(-sdf, H, L);

    let dhdx = droplet(-sdf_fusion(frag_coord + vec2f(1., 0.), R, pa, -pa, t), H, L) - h;
    let dhdy = droplet(-sdf_fusion(frag_coord + vec2f(0., 1.), R, pa, -pa, t), H, L) - h;
    let ndir = normalize(vec3f(-dhdx, -dhdy, 1.));

    let mask = linearstep(1., -1., sdf); 

    let ldir = normalize(vec3f(cos(time.elapsed), sin(time.elapsed), custom.light_height));
    let vdir = vec3f(0., 0., 1.);
    let hdir = normalize(ldir + vdir);

    let diffuse = 0.5 * dot(ldir, ndir) + 0.5;
    let specular = pow(max(dot(ndir, hdir), 0.), custom.shininess);

    var col = mix(vec3f(0.5), vec3f(diffuse + specular), mask);
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

fn sdf_fusion(p: vec2f, r: f32, a: vec2f, b: vec2f, t: f32) -> f32 {
    let rmat = mat2x2f(cos(time.elapsed), sin(time.elapsed), -sin(time.elapsed), cos(time.elapsed));
    let aa = rmat * a;
    let bb = rmat * b;
    
    let sdfa = sdf_circle(p, r);
    let sdfb = sdf_segment(p, aa, bb, 1.2 * r);

    return mix(sdfa, sdfb, t);
    // return sdfb;
}

fn sdf_segment(p: vec2f, a: vec2f, b: vec2f, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;

    // let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    let h = smoothstep2(0., dot(ba, ba), dot(pa, ba));

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