const PI: f32 = 3.1415926;

const CELL: f32 = 64.;
const BLUR: f32 = 1. / CELL;

const BGCOL: vec3f = vec3f(255., 228., 0.) / 255.;
const DTCOL: vec3f = vec3f(4.) / 255.;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let size = vec2f(screen_size);

    let frag_coord = vec2f(id.xy);

    let cell_part = frag_coord / CELL;
    let cell_nums = trunc(size / CELL);

    let cell_id = trunc(cell_part);
    let cell_uv = fract(cell_part);

    let tex_size = vec2f(textureDimensions(channel0));
    let cuv = cell_id / cell_nums;
    let rdm = textureSampleLevel(channel0, bilinear, cuv, 0).rgb;

    var center = vec2f(0.5);
    center += custom.spread * cos(time.elapsed * custom.speedA + 2. * PI * rdm.rg);

    center -= 0.5;
    center = rotate(center, rdm.b + custom.speedB * time.elapsed);
    center += 0.5;

    let sdf = sdf_circle(cell_uv - center, custom.radius);
    let dot = linearstep(BLUR, -BLUR, sdf);

    var col = mix(BGCOL, DTCOL, dot);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return saturate((x - e0) / (e1 - e0));
}

fn rotate(p: vec2f, t: f32) -> vec2f {
    return vec2f(
        p.x * cos(t) - p.y * sin(t),
        p.x * sin(t) + p.y * cos(t),
    );
}