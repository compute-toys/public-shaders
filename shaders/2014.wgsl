const COLOR_0: vec3f = vec3f(128, 60, 245) / 255.;
const COLOR_1: vec3f = vec3f(245, 88, 178) / 255.;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    var fragCoord = vec2f(id.xy);
    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / vec2f(screen_size);

    // apply scale
    fragCoord *= custom.scale;

    var col = vec3f(0.);

    // dither without blue noise
    if uv.x < 0.25
    {
        col = mix(COLOR_0, COLOR_1, uv.y);
    }
    // dither with blue noise(binary)
    else if uv.x < 0.5 {
        let tex_size = vec2f(textureDimensions(channel0));
        let tuv = fragCoord / tex_size;

        var bnoise = textureSampleLevel(channel0, nearest_repeat, tuv, 0).r;
        // fix color space(from inverse gamma to linear)
        bnoise = pow(bnoise, 1. / 2.2);
        // Transform distribution to triangle
        bnoise = uniform_to_triangle(bnoise);

        let ratio = floor(uv.y * 2. + bnoise) / 2.;
        col = mix(COLOR_0, COLOR_1, ratio);
    }
    // dither with random noise(binary)
    else if uv.x < 0.75
    {
        var noise = random12(fragCoord);
        // Transform distribution to triangle
        noise = uniform_to_triangle(noise);

        let ratio = floor(uv.y * 2. + noise) / 2.;
        col = mix(COLOR_0, COLOR_1, ratio);
    }
    else
    // dither with Bayer filter(binary)
    {
        let tex_size = vec2f(textureDimensions(channel1));
        let tuv = fragCoord / tex_size;
        var bayer = textureSampleLevel(channel1, nearest_repeat, tuv, 0).r;
        // fix color space(from inverse gamma to linear)
        bayer = pow(bayer, 1. / 2.2);
        // Transform distribution to triangle
        bayer = uniform_to_triangle(bayer);

        let ratio = floor(uv.y * 2. + bayer) / 2.;
        col = mix(COLOR_0, COLOR_1, ratio);
    }

    let screen_size_f = custom.scale * vec2f(screen_size);
    let g = abs(fragCoord.x - 0.25 * screen_size_f.x) < 1.
    || abs(fragCoord.x - 0.50 * screen_size_f.x) < 1.
    || abs(fragCoord.x - 0.75 * screen_size_f.x) < 1.;
    col = mix(col, vec3f(0.08), select(0., 1., g));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn uniform_to_triangle(v: f32) -> f32 {
    var n = v * 2.0 - 1.0;
    n = sign(n) * (1.0 - sqrt(max(0.0, 1.0 - abs(n)))); // [-1, 1], max prevents NaNs
    return n + 0.5; // [-0.5, 1.5]
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