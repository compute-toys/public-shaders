// Compute.Toys port of fused 4x4 Worley fBm with octave-local PSRD curl warp.
// Edit the constants below; Compute.Toys provides `screen` and `time`.
// Or change the values (for some) in the uniform buffer

// Values are normalised before looking up the palette.
// For F2-F1 with L1, start with [0, 1] and tune VALUE_MAX as desired.
const VALUE_MIN: f32 = 0.0;
const VALUE_MAX: f32 = 1.0;

const RED_COEFFS   = array<f32, 5>(6.381e-10, -3.626e-07, 5.265e-05, 0.002549, 0.1565);
const GREEN_COEFFS = array<f32, 5>(2.443e-11, -1.419e-09, 9.979e-06, 0.0007522, 0.05846);
const BLUE_COEFFS  = array<f32, 5>(1.445e-09, -4.338e-07, 1.427e-05, 0.002326, 0.294);

const OCTAVES: u32 = 5u;
const OUTPUT_MODE: u32 = 0u;   // 0=F1, 1=1-F1, 2=F2, 3=F2-F1, 4=F1*F2, 5=F1/F2
const DISTANCE_MODE: u32 = 0u; // 0=L2, 1=L1, 2=Linf, 3=Minkowski p=4

const WORLEY_FREQUENCY: f32 = 0.025;
const PERSISTENCE: f32 = 0.5;
const LACUNARITY: f32 = 2.0;
const JITTER: f32 = 0.928;
const CURL_FREQUENCY: f32 = 0.8;
const WARP_AMPLITUDE: f32 = 0.15;
const MAX_WARP: f32 = 0.25;
const LOOP_SECONDS: f32 = 8.0;
const TAU: f32 = 6.28318530718;

const MAX_JITTER_4X4_F2: f32 = 0.92820323;

// https://www.pcg-random.org/

fn pcg2d(p: vec2i) -> vec2u {
    var v = bitcast<vec2u>(p) * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> vec2u(16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> vec2u(16u);
    return v;
}

fn hash_to_float(h: u32) -> f32 {
    return f32(h) * (1.0 / 4294967295.0);
}

fn distance_metric(delta: vec2f) -> f32 {
    let d = abs(delta);
    let distance_mode = u32(round(custom.distance_mode));
    switch distance_mode {
        case 1u: { return d.x + d.y; }
        case 2u: { return max(d.x, d.y); }
        case 3u: {
            let d2 = d * d;
            return sqrt(sqrt(d2.x * d2.x + d2.y * d2.y));
        }
        default: { return length(delta); }
    }
}

fn worley_f1_f2(pos: vec2f) -> vec2f {
    let base_cell = vec2i(floor(pos + vec2f(0.5))) - vec2i(2);
    let jitter = clamp(custom.jitter, 0.0, MAX_JITTER_4X4_F2);
    var f1 = 1e30;
    var f2 = 1e30;

    for (var dy = 0; dy < 4; dy++) {
        for (var dx = 0; dx < 4; dx++) {
            let cell = base_cell + vec2i(dx, dy);
            let h = pcg2d(cell);
            let random_offset = vec2f(hash_to_float(h.x), hash_to_float(h.y));
            let feature_point = vec2f(cell) + mix(vec2f(0.5), random_offset, jitter);
            let d = distance_metric(pos - feature_point);

            if (d < f1) {
                f2 = f1;
                f1 = d;
            } else if (d < f2) {
                f2 = d;
            }
        }
    }
    return vec2f(f1, f2);
}

fn select_output(f1_f2: vec2f) -> f32 {
    let f1 = f1_f2.x;
    let f2 = f1_f2.y;
    let output_mode = u32(round(custom.output_mode));
    switch output_mode {
        case 1u: { return 1.0 - f1; }
        case 2u: { return f2; }
        case 3u: { return f2 - f1; }
        case 4u: { return f1 * f2; }
        case 5u: { return f1 / max(f2, 1e-6); }
        default: { return f1; }
    }
}

struct NoiseGradient {
    value: f32,
    gradient: vec2f,
}

fn mod289(x: vec3f) -> vec3f {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

// psrdnoise (c) 2021 Stefan Gustavson and Ian McEwan
// Published under the MIT license.
// https://github.com/stegu/psrdnoise/

fn srdnoise2(pos: vec2f, alpha: f32) -> NoiseGradient {
    let skewed = vec2f(pos.x + 0.5 * pos.y, pos.y);
    let i0 = floor(skewed);
    let f0 = skewed - i0;
    let o1 = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), f0.x > f0.y);
    let i1 = i0 + o1;
    let i2 = i0 + vec2f(1.0);

    let v0 = vec2f(i0.x - 0.5 * i0.y, i0.y);
    let v1 = vec2f(v0.x + o1.x - 0.5 * o1.y, v0.y + o1.y);
    let v2 = vec2f(v0.x + 0.5, v0.y + 1.0);
    let x0 = pos - v0;
    let x1 = pos - v1;
    let x2 = pos - v2;

    let iu = vec3f(i0.x, i1.x, i2.x);
    let iv = vec3f(i0.y, i1.y, i2.y);
    var hash = mod289(iu);
    hash = mod289((hash * 51.0 + 2.0) * hash + iv);
    hash = mod289((hash * 34.0 + 10.0) * hash);

    let psi = hash * 0.07482 + alpha;
    let gx = cos(psi);
    let gy = sin(psi);
    let g0 = vec2f(gx.x, gy.x);
    let g1 = vec2f(gx.y, gy.y);
    let g2 = vec2f(gx.z, gy.z);

    var w = 0.8 - vec3f(dot(x0, x0), dot(x1, x1), dot(x2, x2));
    w = max(w, vec3f(0.0));
    let w2 = w * w;
    let w4 = w2 * w2;
    let w3 = w2 * w;
    let gdotx = vec3f(dot(g0, x0), dot(g1, x1), dot(g2, x2));
    let dw = -8.0 * w3 * gdotx;
    let dn0 = w4.x * g0 + dw.x * x0;
    let dn1 = w4.y * g1 + dw.y * x1;
    let dn2 = w4.z * g2 + dw.z * x2;

    return NoiseGradient(10.9 * dot(w4, gdotx), 10.9 * (dn0 + dn1 + dn2));
}

fn clamp_length(v: vec2f, max_length: f32) -> vec2f {
    let length_squared = dot(v, v);
    if (length_squared <= max_length * max_length) {
        return v;
    }
    return v * inverseSqrt(length_squared) * max_length;
}

fn warped_worley_fbm(base_worley_pos: vec2f) -> vec2f {
    var total = vec2f(0.0);
    var amplitude = 1.0;
    var frequency = 1.0;
    var amplitude_sum = 0.0;
    let alpha = TAU * select(fract(time.elapsed / max(custom.loop_seconds, 0.00001)), 0.0, custom.loop_seconds == 0.0);

    for (var octave = 0u; octave < u32(round(custom.octaves)); octave++) {
        let octave_pos = base_worley_pos * frequency;
        let octave_offset = vec2f(19.19, 47.37) * f32(octave);
        let curl_noise = srdnoise2(octave_pos * CURL_FREQUENCY + octave_offset, alpha);
        let curl = vec2f(-curl_noise.gradient.y, curl_noise.gradient.x);
        let warp = clamp_length(curl * custom.warp_amplitude, MAX_WARP);

        total += worley_f1_f2(octave_pos + warp) * amplitude;
        amplitude_sum += amplitude;
        frequency *= LACUNARITY;
        amplitude *= PERSISTENCE;
    }
    return total / max(amplitude_sum, 1e-8);
}

fn evaluate_polynomial(coeffs: array<f32, 5>, x: f32) -> f32 {
    // Horner form: equivalent to polynomial, but cheaper.
    return ((((coeffs[0] * x + coeffs[1]) * x + coeffs[2]) * x
        + coeffs[3]) * x + coeffs[4]);
}

// Don't know how to use a LUT in Compute.Toys so there's this instead.

fn inferno_colormap(t: f32) -> vec3f {
    let x = clamp(t, 0.0, 1.0) * 255.0;

    return clamp(vec3f(
        evaluate_polynomial(RED_COEFFS, x),
        evaluate_polynomial(GREEN_COEFFS, x),
        evaluate_polynomial(BLUE_COEFFS, x)
    ), vec3f(0.0), vec3f(1.0));
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) {
        return;
    }

    // Compute.Toys has bottom-left pixel-coordinate convention.
    let fragCoord = vec2f(f32(id.x) + 0.5, f32(screen_size.y - id.y) - 0.5);
    let f1_f2 = warped_worley_fbm(fragCoord * WORLEY_FREQUENCY);
    let value = select_output(f1_f2);

    let range = max(VALUE_MAX - VALUE_MIN, 1e-6);
    let t = clamp((value - VALUE_MIN) / range, 0.0, 1.0);

    let inferno_srgb = inferno_colormap(t);

    // `screen` expects linear colour
    let col = pow(inferno_srgb, vec3f(2.2));

    textureStore(screen, vec2i(id.xy), vec4f(col, 1.0));
}