@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
    let c_height = screen_size.y / 4;

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let cell_nm = id.y / c_height;
    let frag_coord = vec2f(id.xy);

    var col = vec3f(1.0);

    let D = u32(custom.depth);
    let L = screen_size.x / D;
    
    let level = f32(id.x) / f32(L);
    if cell_nm == 0 {
        let gray = floor(level) / f32(D - 1);
        col = vec3f(gray);
    } else if cell_nm == 1 {
        let tex_size = vec2f(textureDimensions(channel0));
        let tuv = frag_coord / tex_size;

        var bnoise = textureSampleLevel(channel0, nearest_repeat, tuv, 0).r;
        // fix color space(from inverse gamma to linear)
        bnoise = pow(bnoise, 1. / 2.2);
        // Transform distribution to triangle
        bnoise = uniform_to_triangle(bnoise);

        let gray = floor(level + bnoise) / f32(D - 1);
        col = vec3f(gray);
    } else if cell_nm == 2 {
        var wnoise = random12(frag_coord);
        // Transform distribution to triangle
        wnoise = uniform_to_triangle(wnoise);

        let gray = floor(level + wnoise) / f32(D - 1);
        col = vec3f(gray);
    } else {
        let tex_size = vec2f(textureDimensions(channel1));
        let tuv = frag_coord / tex_size;
        var bayer = textureSampleLevel(channel1, nearest_repeat, tuv, 0).r;
        // fix color space(from inverse gamma to linear)
        bayer = pow(bayer, 1. / 2.2);
        // Transform distribution to triangle
        bayer = uniform_to_triangle(bayer);

        let gray = floor(level + bayer) / f32(D - 1);
        col= vec3f(gray);
    }

    // draw grid
    col *= draw_hori_lines(frag_coord, 3, c_height);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_line(p: vec2f, a: f32, b: f32, c: f32, t: f32) -> f32 {
    let sdf = (a * p.x + b * p.y + c) / length(vec2f(a, b)); 
    
    return abs(sdf) - t;
}

fn draw_hori_lines(frag_coord: vec2f, count: u32, dist: u32) -> f32 {
    var sdf = 99.;
    for (var y: u32 = 1; y <= count; y++) {
        let d = sdf_line(frag_coord, 0., 1., -f32(y * dist), 1.);
        sdf = min(sdf, d);
    }

    return step(0.0, sdf);
}

fn uniform_to_triangle(v: f32) -> f32 {
    var n = v * 2.0 - 1.0;
    n = sign(n) * (1.0 - sqrt(max(0.0, 1.0 - abs(n)))); // [-1, 1], max prevents NaNs
    // return n + 0.5; // [-0.5, 1.5]
    return n;
}

const RANDOM_SINLESS: bool = true;
const RANDOM_SCALE: vec4f = vec4f(.1031, .1030, .0973, .1099);
fn random12(st: vec2f) -> f32 {
    if (RANDOM_SINLESS) {
        var p3  = fract(vec3(st.xyx) * RANDOM_SCALE.xyz);
        p3 += dot(p3, p3.yzx + 33.33);
        return fract((p3.x + p3.y) * p3.z);
    } else {
        return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453);
    }
}