const R_MAX: f32 = 96.0;
const H: f32 = 32.0;
const PI: f32 = 3.1415926;

const PA: vec2f = vec2f(-196., 0.);
const PB: vec2f = -PA;

const vdir: vec3f = vec3f(0., 0., 1.);

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
    let screen_size_f = vec2f(screen_size);

    // var t = i_ease_elastic(0.5 * modulo(time.elapsed, 2.));
    var t = i_ease_elastic(0.5 * op_onion(time.elapsed - 2., 2., 0.));
    let R = mix(R_MAX, 0.70 * R_MAX, t);
    let L = R;

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    t = -PI / 4.;
    let ldir = normalize(vec3f(cos(t), sin(t), custom.light_height));
    let hdir = normalize(ldir + vdir);

    let frag_coord = vec2f(id.xy) - 0.5 * vec2f(screen_size);

    let size = vec2f(0.5 * f32(screen_size.x), f32(screen_size.y));
    let cell_coord = frag_coord % size - 0.5 * size;

    t = io_ease_back(0.25 * op_onion(time.elapsed - 4., 4., 0.));
    // t =  0.25 * op_onion(time.elapsed - 4., 4., 0.);
    let cent_coord = mix(PA, PB, t);
    let orig_coord = frag_coord - cent_coord;

    // t = 0.5 * op_onion(time.elapsed - 2., 2., 0.);
    t = 0.5 * max(op_onion(time.elapsed - 3.75, 4., 2.), 0.);
    t = o_ease_elastic(t);

    let sdf = sdf_mix(orig_coord, R, t);
    let h = droplet(-sdf, H, L);

    let dhdx = droplet(-sdf_mix(orig_coord + vec2f(1., 0.), R, t), H, L) - h;
    let dhdy = droplet(-sdf_mix(orig_coord + vec2f(0., 1.), R, t), H, L) - h;
    let ndir = normalize(vec3f(-dhdx, -dhdy, 1.));

    let mask = linearstep(1., -1., sdf + 1.5);

    var shadow = sdf_mix(orig_coord + custom.shadow_offset * normalize(ldir.xy), R, t);
    shadow = abs(shadow) - 0.33 * custom.shadow_blur;
    shadow = linearstep(-custom.shadow_blur, custom.shadow_blur, shadow);
    shadow = mix(custom.shadow_lightness, 1., shadow);

    let rim = abs(dot(ndir, ldir));
    // let diffuse = 0.5 * dot(ldir, ndir) + 0.5;
    // let specular = pow(max(dot(ndir, hdir), 0.), custom.shininess);

    let idir = vec3f(0., 0., -1.);

    var col = vec3f(0.0);
    for (var i = 0; i < 3; i++) {
        let odir = refract(idir, ndir, custom.base_ior + f32(i) * custom.dispersion);
        var dfc = frag_coord - odir.xy / odir.z * H * custom.distort_scale;
        dfc = mix(frag_coord, dfc, mask);
        col[i] = draw_background(dfc, screen_size_f)[i];
    }
    
    // col = draw_background(dfc);
    col = col * shadow;
    col = mix(col, pow(col, vec3f(custom.contrast)), mask);
    col += custom.lightness * mask;
    col += custom.rim_multi * pow(rim, custom.rim_power) + custom.rim_multi2 * pow(rim, custom.rim_power2);
    col += 0.1 * linearstep(1., -1., abs(sdf) - 1.);
    // col -= 0.2 * diffuse;
    // col += specular;
    // tonemapping
    col = tanh(1.5 * pow(col, vec3f(1.8)));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn draw_background(p: vec2f, s: vec2f) -> vec3f {
    const GSIZE: f32 = 48.0;
    const TSIZE: f32 = 0.05;
    const DGREY: vec3f = vec3f(59., 67., 73.) / 255.;
    const LGREY: vec3f = vec3f(184., 190., 203.) / 255.;

    var sdf = op_onion(p.x, GSIZE, TSIZE);
    sdf = min(sdf, op_onion(p.y, GSIZE, TSIZE));
    let grid = linearstep(-1., 1., sdf); 

    let ra = length(p + vec2f(-0.8, 0.6) * s) / s.x;
    let bg = mix(DGREY, LGREY, clamp(ra, 0., 1.));

    let gradient = draw_gradient(p, 1.);

    var col = bg;
    col *= mix(0.7, 1., grid);
    col = mix(col, gradient.rgb, gradient.a);

    return col;
}

fn draw_gradient(p: vec2f, s: f32) -> vec4f {
    const HUEA: f32 = 261. / 360.;
    const HUEB: f32 = 358. / 360.;
    const HUEC: f32 = (360 + 90.) / 360.;
    
    let sdf = sdf_segment(p, PA, PB, 32.);
    let alpha = linearstep(s, -s, sdf);
    
    let r = clamp((p.x - PA.x) / (PB.x - PA.x), 0.0, 1.0);

    var hue = mix(HUEA, HUEB, linearstep(0.0, 0.4, r));
    hue = mix(hue, HUEC, linearstep(0.4, 1.0, r));

    return vec4f(hsv2rgb(vec3f(hue, 0.60, 0.85)), alpha);
}


fn droplet(x: f32, h: f32, l: f32) -> f32 {
    let t = h / l;
    var v = t * x * (2. * h - t * x);
    v = sqrt(op_smax(v, 0., 8.));
    return select(h, v, x < l);
}

fn sdf_mix(p: vec2f, r: f32, t: f32) -> f32 {
    let sdfa = sdf_circle(p, r);
    let sdfb = sdf_smooth_segment(p, -vec2f(0.7 * r, 0.), vec2f(0.7 * r, 0.), 0.6 * R_MAX);

    return mix(sdfa, sdfb, t);
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_segment(p: vec2f, a: vec2f, b: vec2f, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;

    // let h = smoothstep2(0., dot(ba, ba), dot(pa, ba));
    let h = saturate(dot(pa, ba) / dot(ba, ba));

    return length(pa - ba * h) - r;
}

fn sdf_smooth_segment(p: vec2f, a: vec2f, b: vec2f, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;

    let h = smoothstep2(0., dot(ba, ba), dot(pa, ba));
    // let h = saturate(dot(pa, ba) / dot(ba, ba));

    return length(pa - ba * h) - r;
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return clamp((x - e0) / (e1 - e0), 0.0, 1.0);
}

fn op_onion(d: f32, l: f32, t: f32) -> f32 {
    return abs(modulo(d, 2. * l) - l) - t;
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
    const K: f32 = 6.;

    let r = (x - e0) / (e1 - e0);

    var v = 1. + exp2(K * r);
    v /= 1. + exp2(K * (r - 1.));
    v = log2(v) / K;

    return v;
}

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn hsv2rgb(c: vec3f) -> vec3f {
    const K: vec4f = vec4f(1., 2. / 3., 1. / 3., 3.);

    let p = abs(fract(c.xxx + K.xyz) * 6. - K.www);

    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3f(0.), vec3f(1.)), c.y);
}

const C1: f32 = 1.70158;
const C2: f32 = C1 * 1.525;
const C3: f32 = 1 + C1;
const C4: f32 = 0.6667 * PI;
const C5: f32 = 0.4444 * PI;

fn i_ease_elastic(x: f32) -> f32 {
    return  select(
        step(0., x),
        -pow(2, 10 * x - 10) * sin((10 * x - 10.75) * C4),
        x >= 0 && x <= 1,
    );
}

fn o_ease_elastic(x: f32) -> f32 {
    return  select(
        step(0., x),
        pow(2, -10 * x) * sin((10 * x - 0.75) * C4) + 1,
        x >= 0 && x <= 1,
    );
}

fn io_ease_elastic(x: f32) -> f32 {
    return  select(
        step(0., x),
        select(
         0.5 * pow(2, -20 * x + 10) * sin((20 * x - 11.125) * C5) + 1,
        -0.5 * pow(2, 20 * x - 10) * sin((20 * x - 11.125) * C5),
        x < 0.5,
        ),
        x >= 0 && x <= 1,
    );
}

fn io_ease_back(x: f32) -> f32 {
    let t0 = 2 * x - 2;
    let t1 = 2 * x;
    
    return select(
        0.5 * t0 * t0 * ((C2 + 1) * t0 + C2) + 1,
        0.5 * t1 * t1 * ((C2 + 1) * t1 - C2),
        x < 0.5,
    );
}