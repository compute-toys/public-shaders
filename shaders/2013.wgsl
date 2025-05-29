const COLOR_0: vec3f = vec3f(128, 60, 245) / 255.;
const COLOR_1: vec3f = vec3f(245, 88, 178) / 255.;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    //let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let fragCoord = vec2f(id.xy);

    let screen_size_f = vec2f(screen_size);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / screen_size_f;

    // gradient background
    let t = smoothstep(custom.t_dn, custom.t_up, uv.y); // transition
    var col = mix(COLOR_0, COLOR_1, t);

    var r = (2. * fragCoord - screen_size_f) / screen_size_f.y;
    r = vec2f(fract(atan2(r.y, r.x) / 6.28), length(r) - 0.7);

    let noise  = random12(fragCoord);

    let p = smoothstep(0.1, -0.1, abs(r.y) - 0.2);
    let f = smoothstep(0.1, 0.8, r.x) * smoothstep(0.9, 0.8, r.x);

    let i = int(time.elapsed) % 4;
    var v = 0.0;
    if i == 0 {
        v = step(noise, 1.2 * p * f);
    } else if i == 1 {
        v = smoothstep(-.25, .25, 1.2 * p * f - noise);
    }else if i == 2 {
        v = 1.2 * p * f - noise;
    } else {
        v = 2. * p * f - noise;
    }
    v = saturate(v);

    col = mix(col, vec3f(0.95), v);

    // text
    const ST: vec2f = vec2f(16.);
    const SZ: vec2f = vec2f(54.);

    if all(fragCoord >= ST) && all(fragCoord < ST + SZ) {
        let uv = (fragCoord - ST) / SZ;
        let t = sample_text(48 + u32(i), uv).r;
        col = mix(col, vec3f(1.), t);
    }

    // grain
    col += (noise - 0.5) * custom.grain_strength;
    col = saturate(col);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
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

fn sample_text(i: u32, uv: vec2f) -> vec3f {
    let x = f32(i % 16);
    let y = f32(i / 16);

    let uv0 = (vec2f(x, y) + uv) / 16.;

    return textureSampleLevel(channel0, bilinear, uv0, 0).rgb;
}