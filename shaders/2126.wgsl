const GSIZE: f32 = 32.;
const PI: f32 = 3.1415926;
const COL: vec3f = normalize(vec3f(0.8, 0.2, 0.4));
const GCOL_0: vec3f = vec3f(78 , 104, 120) / 255;
const GCOL_1: vec3f = vec3f(244, 235, 222) / 255;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let size = vec2f(screen_size);
    let muv = vec2f(mouse.pos) / size;

    let frag_coord = vec2f(id.xy) - 0.5 * size;

    let rmat = rmat(2.0 * PI * fract(0.05 * time.elapsed));
    let smat = normalize(vec2f(1., 2.)) * muv.y;

    let uvy = f32(screen_size.y - id.y) / size.y;

    let freq_0 = 16.;
    let freq_1 = 4.0;

    var col = vec3f(0.0);
    var d = vec2f(0.05, linearstep(0.5, 1.5, uvy) * size.y / 504.);
    for (var i = 1.0; i < 16.0; i += 1. / i) {
        var orig_coord = rmat * (smat * (frag_coord + i * d));

        d = d * mat2x2f(-73., -67., 67., -73.) / 99.;

        let dpdx_ = rmat * (smat * vec2f(1., 0.));
        let dpdy_ = rmat * (smat * vec2f(0., 1.));

        let board = checkerboard(orig_coord, dpdx_, dpdy_);
        let chkrb = mix(GCOL_0, GCOL_1, board);

        // up direction in origin-space
        var up = vec2f(0.0, 1.0);
        up = normalize(rmat * (smat * up));

        orig_coord -= 0.5 * GSIZE;

        var cell_coord = orig_coord + custom.height * up;
        var r = length(floor(cell_coord / GSIZE));
        var wave = sin(freq_0 * r + freq_1* time.elapsed) * 0.25 * GSIZE;
        cell_coord = modulo2(cell_coord, vec2f(GSIZE)) - 0.5 * GSIZE;
        // to screen-space
        cell_coord = (cell_coord * rmat) / smat;

        let s_sdf = sdf_circle(cell_coord + vec2f(0., wave), 0.25 * GSIZE / muv.y);
        let shape = linearstep(1., -1, s_sdf);

        cell_coord = orig_coord - custom.height * up;
        r = length(floor(cell_coord / GSIZE));
        wave = sin(freq_0 * r + freq_1 * time.elapsed) * 0.25 * GSIZE;
        cell_coord = modulo2(cell_coord, vec2f(GSIZE)) - 0.5 * GSIZE;
        // to screen-space
        cell_coord = (cell_coord * rmat) / smat;

        let r_sdf = sdf_circle(cell_coord - vec2f(0., wave), 0.25 * GSIZE / muv.y);
        let rflct = linearstep(2., -2., r_sdf);

        var t = 0.25 * chkrb;
        // reflection
        t = mix(t, t * set_saturation(COL, mix(0.5, 0.8, board)), rflct);
        // main shape
        t = mix(t, COL, shape);

        col += t;
    }

    // convert color from gamma to linear
    col = pow(col / 128, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn op_onion(d: f32, l: f32, t: f32) -> f32 {
    return abs(modulo(d, 2. * l) - l) - t;
}

fn op_xor(d1: f32, d2: f32) -> f32 {
    return max(min(d1, d2), -max(d1, d2));
}

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn modulo2(a: vec2f, b: vec2f) -> vec2f {
    return a - b * floor(a / b);
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return clamp((x - e0) / (e1 - e0), 0.0, 1.0);
}

fn rmat(t: f32) -> mat2x2f {
    let c = cos(t);
    let s = sin(t);

    return mat2x2f(c, s, -s, c);
}

fn luminance(srgb: vec3f) -> f32 {
    let rgb = pow(srgb, vec3f(2.2));
    return dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
}

fn set_saturation(srgb: vec3f, saturation: f32) -> vec3f {
    var luminance = luminance(srgb);
    luminance = pow(luminance, 1.0 / 2.2);
    return mix(vec3f(luminance), srgb, vec3f(saturation));
}

fn checkerboard(p: vec2f, dpdx_: vec2f, dpdy_: vec2f) -> f32 {
    let hori = op_onion(p.x, GSIZE, 0.5 * GSIZE);
    let vert = op_onion(p.y, GSIZE, 0.5 * GSIZE);

    let fwid = length(max(abs(dpdx_), abs(dpdy_)));
    let fade = linearstep(0., GSIZE, fwid);

    return mix(linearstep(-fwid, fwid, op_xor(hori, vert)), 0.5, fade);
}