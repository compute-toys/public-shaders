#storage global Global
#storage light BallLight

#include "Dave_Hoskins/hash"

const PI: f32 = 3.1415926;

struct Global {
    screen_size: vec2f,

    click: i32,
    pressed: i32,

    colr_changed: i32,
    colr_seed: f32,
}

struct HitInfo {
    is_hit: bool,
    // hit position
    pos: vec2f,
    norm: vec2f,
}

struct BallLight {
    flux: f32,
    colr: vec3f,
    
    pos: vec2f,
    radius: f32,
}

fn to_radiance(light: BallLight) -> vec3f {
    // from gamma to linear
    let colr = pow(light.colr, vec3f(2.2));

    return colr * light.flux / (2. * PI * light.radius);
}

fn to_color(seed: f32) -> vec3f {
    return normalize(hash31(seed * 64.));
} 

#workgroup_count init 1 1 1
@compute @workgroup_size(1)
fn init(@builtin(global_invocation_id) id: uint3) {
    if(any(id != vec3(0u))) { return; }
    if(time.frame != 0u) { return; }

    global.screen_size = vec2f(textureDimensions(screen));

    light = BallLight (
        1024,
        to_color(custom.colr_seed),

        vec2f(0.),
        16,
    );
}

#workgroup_count update 1 1 1
@compute @workgroup_size(1)
fn update(@builtin(global_invocation_id) id: uint3) {
    if(any(id != vec3(0u))) { return; }
    if(time.frame == 0u) { return; }

    global.click = 0;
    global.click = i32(global.pressed == 0 && mouse.click == 1);
    global.pressed = mouse.click;

    let mouseCoord = vec2f(mouse.pos) - 0.5 * global.screen_size;
    light.pos = mouseCoord;

    global.colr_changed = i32(global.colr_seed != custom.colr_seed);
    global.colr_seed = custom.colr_seed;
    light.colr = to_color(global.colr_seed);
}

@compute @workgroup_size(16,16)
fn update_lightmap(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= vec2u(global.screen_size))) { return; }

    let fragCoord = vec2f(id.xy) - 0.5 * global.screen_size;
    let d_light = sdf_light(fragCoord);
    
    let illu = to_radiance(light) * smoothstep(2, 0, d_light);

    let new_iinfo = vec4f(illu, 1.);
    let old_iinfo = passLoad(0, vec2i(id.xy), 0);

    let to_store = select(
        max(new_iinfo, old_iinfo),
        new_iinfo,
        global.click == 1 || global.colr_changed == 1,
    );

    passStore(0, vec2i(id.xy), to_store);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= u32(global.screen_size.x) || id.y >= u32(global.screen_size.y)) { return; }

    let fragPixel = vec2i(id.xy);
    let fragCoord = vec2f(id.xy) - 0.5 * global.screen_size;

    let theta = 2 * PI * hash13(vec3f(fragCoord, time.elapsed));
    let vdir = vec2f(cos(theta), sin(theta));

    let d_scene = sdf_scene(fragCoord);
    let hinfo = ray_marching(fragCoord, vdir, d_scene, 1000.);

    let hitCoord = hinfo.pos + hinfo.norm;
    let hitPixel = vec2i(round(hitCoord + 0.5 * global.screen_size));

    let radi = select(
        vec3f(0.),
        passLoad(0, hitPixel, 0).xyz,
        hinfo.is_hit,
    );

    let iinfo = passLoad(0, fragPixel, 0);
    let count = iinfo.w + 1.;
    let illu = (iinfo.xyz * (count - 1) + radi) / count;

    passStore(0, fragPixel, vec4f(illu, count));

    var col = max(illu, vec3f(smoothstep(2, 0, d_scene)));

    // post-processing
    // 
    // 1. tonemapping
    col = tonemaping_aces(col);
    // 2. anti color-banding
    col = dithering(fragCoord, col);

    // Output to screen (linear colour space)
    textureStore(screen, vec2i(id.xy), vec4f(col, 1.));
}

// ======== ======== ======== ======== ========
// ====== ======== SDF Functions ======== =====
// ======== ======== ======== ======== ========
fn sdf_scene(p: vec2f) -> f32 {
    const CELL_SIZE: f32 = 128.;
    const BALL_R: f32 = 16.;

    let half_size = global.screen_size / 2.;
    
    let cell_coord = modulo(p, vec2f(CELL_SIZE));
    let d_balls = sdf_circle(cell_coord - vec2f(CELL_SIZE / 2.), BALL_R);

    let d_wall = max(
        sdf_rectangle(p, half_size + vec2f(2.)),
        -sdf_rectangle(p, half_size - vec2f(16.))
    );

    let d_light = sdf_light(p);

    return min(min(d_balls, d_wall), d_light);
}

fn sdf_light(p: vec2f) -> f32 {
    // let d_ball = sdf_circle(p - light.pos, light.radius);
    let d_cross = min(
        sdf_rectangle(p - light.pos, light.radius * vec2f(2, 0.25)),
        sdf_rectangle(p - light.pos, light.radius * vec2f(0.25, 2)),
    );

    // return d_ball;
    return d_cross;
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_rectangle(p: vec2f, hs: vec2f) -> f32 {
    let d = abs(p) - hs;

    return length(max(d, vec2f(0))) + min(max(d.x, d.y), 0);
}

fn normal(p: vec2f) -> vec2f {
    const EPS: vec2f = vec2f(0.01, 0.);

    let dx = sdf_scene(p + EPS.xy) - sdf_scene(p - EPS.xy);
    let dy = sdf_scene(p + EPS.yx) - sdf_scene(p - EPS.yx);

    return normalize(vec2f(dx, dy));
}

fn ray_marching(ori: vec2f, dir: vec2f, init_d: f32, maxt: f32) -> HitInfo {
    const MINT: f32 = 1.;
    const MAX_STEP: u32 = 32;

    var info: HitInfo;

    if (init_d > 0) {
        var len = max(init_d, MINT);
        for (var i: u32 = 0; i < MAX_STEP && len < maxt; i++) {
            let p = ori + dir * len;
            let d = sdf_scene(p);
            let n = normal(p);

            if (d < 1 && dot(n, dir) < 0) {
                info.is_hit = true;
                info.pos = p - n * d;
                info.norm = n;
                return info;
            }

            len += d;
        }
    }

    info.is_hit = init_d <= 0.;
    info.pos = ori;
    info.norm = vec2f(0.);
    return info;
}

// ======== ======== ======== ======== ========
// ==== ======== Post-Processing ======== =====
// ======== ======== ======== ======== ========

fn tonemaping_aces(col_in: vec3f) -> vec3f {
    const IDT: mat3x3f = mat3x3f(
    0.61313242, 0.07012438, 0.02058766,
    0.33953802, 0.91639401, 0.10957457,
    0.04741670, 0.01345152, 0.86978540
    );

    const ODT: mat3x3f = mat3x3f(
        1.70485868, -0.13007682, -0.02396407,
        -0.62171602, 1.14073577, -0.12897551,
        -0.08329937, -0.01055980, 1.15301402
    );

    // linear-space to ACEScg-space
    var col = IDT * col_in;
    let a = col * (col + 0.0245786) - 0.000090537;
    let b = col * (0.983729 * col + 0.4329510) + 0.238081;
    col = clamp(a / b, vec3f(0.), vec3f(1.));
    // ACEScg-space to linear-space
    col = ODT * col;

    return col;
}

fn dithering(fragCoord: vec2f, col_in: vec3f) -> vec3f {
    const SHADES: f32 = 256.;
    const SCATTER: f32 = 1.;
    
    let tex_size = vec2f(textureDimensions(channel0).xy);
    let noise_uv = modulo(fragCoord, tex_size) / tex_size;
    var noise = textureSampleLevel(channel0, nearest, noise_uv, 0).rgb;
    // fix color space(from inverse gamma to linear)
    noise = pow(noise, vec3f(1 / 2.2));
    noise = uniform_to_triangle(noise);

    let col = saturate(col_in + noise / SHADES * SCATTER);

    return col;
}

// ======== ======== ======== ======== ========
// ======= ======== Utilities ======== ========
// ======== ======== ======== ======== ========

fn modulo(x: vec2f, y: vec2f) -> vec2f {
    return x - y * floor(x / y);
}

fn uniform_to_triangle(v: vec3f) -> vec3f {
    var n = v * 2.0 - 1.0;
    n = sign(n) * (1.0 - sqrt(max(vec3f(0.), 1. - abs(n)))); // [-1, 1], max prevents NaNs
    return n + 0.5; // [-0.5, 1.5]
}