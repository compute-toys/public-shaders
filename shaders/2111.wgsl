const THUMBNAIL_TIME: f32 = 4.0; 

const NOISE_AMOUNT: f32 = 0.01;
const SCANLINE_FREQUENCY: f32 = 30.0;
const SCANLINE_INTENSITY: f32 = 0.3;
const DITHER_STRENGTH: f32 = 1.0 / 255.0;

const COLOR_BACKGROUND:    vec3f = vec3f(0.01,  0.02,  0.025);
const COLOR_RED_PRIMARY:   vec3f = vec3f(1.0,   0.1,   0.2);
const COLOR_GREEN_PRIMARY: vec3f = vec3f(0.1,   0.0,   0.3);
const COLOR_BLUE_PRIMARY:  vec3f = vec3f(0.2,   0.4,   1.0);
const COLOR_BURN_YELLOW:   vec3f = vec3f(1.0,   1.0,   0.1);
const COLOR_BURN_CYAN:     vec3f = vec3f(0.1,   1.0,   1.0);
const COLOR_BURN_MAGENTA:  vec3f = vec3f(1.0,   0.1,   1.0);


fn rand(co: vec2f) -> f32 {
    return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
}

fn N(p: vec2f, F: f32, T: f32, t: f32) -> f32 {
    return sin(p.x * F + sin(p.y * F + t * T));
}

fn getSwirl(p_in: vec2f, t_global: f32, time_offset: f32) -> f32 {
    var p = p_in;
    let d = length(p);
    let a = (t_global + time_offset) * 0.1 + 2.5 / (d + 0.2);
    let c = cos(a);
    let s = sin(a);
    
    let scale = 1.0 - sin(d * 7.0 - (t_global * 2.0 + time_offset)) * 0.03;
    let transform = mat2x2f(c, -s, s, s) * scale;
    p = transform * p;
    
    let noise_val = (N(p, 3.0, 0.2, t_global) * 0.6 + N(p, 9.0, 0.5, t_global) * 0.4) * 0.5 + 0.5;
    return noise_val * smoothstep(0.1, 0.15, d);
}

fn getSwirl2(p_in: vec2f, t_global: f32, time_offset: f32) -> f32 {
    var p = p_in;
    let d = length(p);
    let a = (t_global + time_offset) * 0.2 + 1.5 / (d + 0.2);
    let c = cos(a);
    let s = d; 

    let scale = 1.0 - cos(a * 7.0 - (t_global * 2.0 + time_offset)) * 0.03;
    let transform = mat2x2f(c, -s, s, c) * scale;
    p = transform * p;

    let noise_val = (N(p, 3.0, 0.2, t_global) * 0.6 + N(p, 9.0, 0.5, t_global) * 0.4) * 0.5 + 0.5;
    return noise_val * smoothstep(0.8, 0.15, d);
}


fn renderScene(uv: vec2f, t: f32) -> vec3f {
    let distance_from_center = cos(t) * 0.01;
    let center = normalize(vec2f(1.0, 0.25)) * distance_from_center;
    let p = uv - center;

    let red_swirl   = getSwirl(p, t, 0.0);
    let green_swirl = getSwirl(p, t, 10.0);
    let blue_swirl  = getSwirl2(p, t, 20.0);

    let r = smoothstep(0.5, 0.8, red_swirl);
    let g = smoothstep(0.5, 0.8, green_swirl);
    let b = smoothstep(0.1, 0.8, blue_swirl);

    var col = COLOR_BACKGROUND;
    col += COLOR_RED_PRIMARY   * red_swirl;
    col += COLOR_GREEN_PRIMARY * green_swirl;
    col += COLOR_BLUE_PRIMARY  * blue_swirl;

    let rg = r * g;
    let gb = g * b;
    let br = b * r;
    let burn_colors = COLOR_BURN_YELLOW  * (rg * rg) +
                      COLOR_BURN_CYAN    * (gb * gb) +
                      COLOR_BURN_MAGENTA * (br * br);
    col += burn_colors / 3.0;

    col *= smoothstep(1.2, 0.5, length(uv));

    return col;
}



@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {

    let screen_dims = textureDimensions(screen);


    if (id.x >= screen_dims.x || id.y >= screen_dims.y) {
        return;
    }


    let fragCoord = vec2f(f32(id.x), f32(id.y));
    let screen_size = vec2f(screen_dims);
    let uv = (fragCoord - screen_size * 0.5) / screen_size.y;

    let t = time.elapsed + THUMBNAIL_TIME;

    var col = renderScene(uv, t);

    let scanline_val = sin(fragCoord.y * SCANLINE_FREQUENCY) * 0.5 + 0.5;
    col *= 1.0 - (scanline_val * scanline_val) * SCANLINE_INTENSITY;

    col += (rand(fragCoord + t) - 0.5) * NOISE_AMOUNT;
    col += (rand(fragCoord) - 0.5) * DITHER_STRENGTH;

    let final_color = vec4f(clamp(col, vec3f(0.0), vec3f(1.0)), 1.0);
    textureStore(screen, id.xy, final_color);
}