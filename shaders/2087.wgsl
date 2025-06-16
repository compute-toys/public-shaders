// --- CRT EFFECT CONTROLS ---
const CRT_CURVATURE: f32 = 0.3;
const SCANLINE_INTENSITY: f32 = 0.2;
const VIGNETTE_STRENGTH: f32 = 0.4;
const CHROMA_ABERRATION_AMOUNT: f32 = 8.5;
const NOISE_INTENSITY: f32 = 0.35;
const NOISE_STRETCH: vec2f = vec2f(1.0, 1.0);
const DESATURATION: f32 = 0.0;
const TINT: vec3f = vec3f(1.0, 1.0, 1.0);

// --- BASE PATTERN CONTROLS ---
const SPIN_ROTATION: f32 = -2.0;
const SPIN_SPEED: f32 = 7.0;
const OFFSET: vec2f = vec2f(0.0, 0.0);
const COLOUR_1: vec4f = vec4f(0.871, 0.267, 0.231, 1.0);
const COLOUR_2: vec4f = vec4f(0.0, 0.42, 0.706, 1.0);
const COLOUR_3: vec4f = vec4f(0.086, 0.137, 0.145, 1.0);
const CONTRAST: f32 = 3.5;
const LIGTHING: f32 = 0.4; 
const SPIN_AMOUNT: f32 = 0.25;
const SPIN_EASE: f32 = 1.0;

// --- OPTIMIZATION CONSTANTS ---
const SQRT2: f32 = 1.41421356237;
const CONTRAST_MOD: f32 = (0.25 * CONTRAST + 0.5 * SPIN_AMOUNT + 1.2);
const SPIN_ANGLE_BASE: f32 = (SPIN_ROTATION * SPIN_EASE * 0.2 + 302.2);
const TWIST_FACTOR: f32 = (SPIN_EASE * 20.0);
const MIX_FACTOR_A: f32 = (0.3 / CONTRAST);
const MIX_FACTOR_B: f32 = (1.0 - MIX_FACTOR_A);


// --- HELPER FUNCTIONS ---
fn rand(co: vec2f) -> f32 {
    return fract(sin(dot(co, vec2f(12.9898, 78.233))) * 43758.5453);
}

// NOTE: This value noise function is the correct WGSL equivalent of the GLSL version.
fn value_noise(p: vec2f) -> f32 {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    // Interpolate along the bottom edge (x-axis)
    let a = mix(rand(i), rand(i + vec2f(1.0, 0.0)), f.x);
    // Interpolate along the top edge (x-axis)
    let b = mix(rand(i + vec2f(0.0, 1.0)), rand(i + vec2f(1.0, 1.0)), f.x);
    // Interpolate between the two edges (y-axis)
    return mix(a, b, f.y);
}

// --- BASE PATTERN FUNCTION ---
fn base_pattern_optimized(screenSize: vec2f, fragCoord: vec2f, invScreenLen: f32, timeOffsets: vec2f) -> vec4f {
    var uv = (fragCoord - 0.5 * screenSize) * invScreenLen - OFFSET;
    let twist_angle = TWIST_FACTOR * mix(1.0, length(uv), SPIN_AMOUNT);
    let total_angle = SPIN_ANGLE_BASE - twist_angle;
    let c = cos(total_angle);
    let s = sin(total_angle);
    uv = mat2x2f(c, s, -s, c) * uv;
    uv *= 30.0;
    
    var uv2 = uv.xx + uv.yy;
    
    // Iteration 1
    uv2 += vec2f(sin(max(uv.x, uv.y))) + uv;
    uv += 0.5 * vec2f(cos(5.1123314 + 0.353 * uv2.y + timeOffsets.x), sin(uv2.x - timeOffsets.y));
    uv -= vec2f(cos(uv.x + uv.y) - sin(uv.x * 0.711 - uv.y));
    // Iteration 2
    uv2 += vec2f(sin(max(uv.x, uv.y))) + uv;
    uv += 0.5 * vec2f(cos(5.1123314 + 0.353 * uv2.y + timeOffsets.x), sin(uv2.x - timeOffsets.y));
    uv -= vec2f(cos(uv.x + uv.y) - sin(uv.x * 0.711 - uv.y));
    // Iteration 3
    uv2 += vec2f(sin(max(uv.x, uv.y))) + uv;
    uv += 0.5 * vec2f(cos(5.1123314 + 0.353 * uv2.y + timeOffsets.x), sin(uv2.x - timeOffsets.y));
    uv -= vec2f(cos(uv.x + uv.y) - sin(uv.x * 0.711 - uv.y));
    // Iteration 4
    uv2 += vec2f(sin(max(uv.x, uv.y))) + uv;
    uv += 0.5 * vec2f(cos(5.1123314 + 0.353 * uv2.y + timeOffsets.x), sin(uv2.x - timeOffsets.y));
    uv -= vec2f(cos(uv.x + uv.y) - sin(uv.x * 0.711 - uv.y));
    // Iteration 5
    uv2 += vec2f(sin(max(uv.x, uv.y))) + uv;
    uv += 0.5 * vec2f(cos(5.1123314 + 0.353 * uv2.y + timeOffsets.x), sin(uv2.x - timeOffsets.y));
    uv -= vec2f(cos(uv.x + uv.y) - sin(uv.x * 0.711 - uv.y));
    
    let paint_res = clamp(length(uv) * 0.035 * CONTRAST_MOD, 0.0, 2.0);
    let c1p = max(0.0, 1.0 - CONTRAST_MOD * abs(1.0 - paint_res));
    let c2p = max(0.0, 1.0 - CONTRAST_MOD * paint_res);
    let c3p = 1.0 - clamp(c1p + c2p, 0.0, 1.0);
    
    let light = (LIGTHING - 0.2) * max(c1p * 5.0 - 4.0, 0.0) + LIGTHING * max(c2p * 5.0 - 4.0, 0.0);
    
    let base_color = COLOUR_1 * c1p + COLOUR_2 * c2p + vec4f(COLOUR_3.xyz, COLOUR_1.a) * c3p;
    
    var final_color = MIX_FACTOR_A * COLOUR_1 + MIX_FACTOR_B * base_color;
    final_color = vec4f(final_color.xyz + vec3f(light), final_color.a);
    return final_color;
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size_u = textureDimensions(screen);
    let screen_size = vec2f(screen_size_u);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size_u.x || id.y >= screen_size_u.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left, to match gl_FragCoord)
    let fragCoord = vec2f(f32(id.x) + 0.5, f32(screen_size_u.y - id.y) - 0.5);

    // --- Pre-calculate values ---
    let invScreenLen = 1.0 / length(screen_size);
    let timeOffsets = time.elapsed * SPIN_SPEED * vec2f(0.131121, 0.113);

    // --- CRT Emulation ---
    let crt_uv = fragCoord / screen_size - vec2f(0.5);
    let dist_sq = dot(crt_uv, crt_uv);
    let dist = sqrt(dist_sq);
    let bend = CRT_CURVATURE * dist_sq;
    let curved_uv = crt_uv * (1.0 - bend);
    
    let aberration_offset_pixels = dist * SQRT2 * CHROMA_ABERRATION_AMOUNT;
    let offset = vec2f(aberration_offset_pixels, 0.0) / screen_size.x;
    
    let g_coord = (curved_uv + vec2f(0.5)) * screen_size;
    let r_coord = (curved_uv + offset + vec2f(0.5)) * screen_size;
    
    let color_g = base_pattern_optimized(screen_size, g_coord, invScreenLen, timeOffsets);
    let color_r = base_pattern_optimized(screen_size, r_coord, invScreenLen, timeOffsets);
    
    let color_b_approx = color_g.xyz - (color_r.xyz - color_g.xyz);
    var final_color = vec4f(color_r.r, color_g.g, color_b_approx.b, 1.0);

    // --- Signal Noise ---
    var noise_uv = fragCoord / NOISE_STRETCH;
    noise_uv.x += noise_uv.y * 0.3;
    let noise = mix(0.5, value_noise(noise_uv), NOISE_INTENSITY);
    let base = final_color.xyz;
    let blend_dark = 2.0 * base * noise + base * base * (1.0 - 2.0 * noise);
    let blend_light = sqrt(max(vec3f(0.0), base)) * (2.0 * noise - 1.0) + 2.0 * base * (1.0 - noise);
    // NOTE: Using `step(vec3f(0.5), vec3f(noise))` is the WGSL equivalent of GLSL's `step(0.5, noise)` in a `mix` with vector components.
    let new_rgb_noise = mix(blend_dark, blend_light, step(vec3f(0.5), vec3f(noise)));
    final_color = vec4f(new_rgb_noise, final_color.a);

    // --- Color Grading & Final Touches ---
    let luminance = dot(final_color.xyz, vec3f(0.2126, 0.7152, 0.0722));
    let desaturated_rgb = mix(final_color.xyz, vec3f(luminance), DESATURATION);
    final_color = vec4f(desaturated_rgb * TINT, final_color.a);
    
    let scanline = sin(fragCoord.y * 2.094395); // approx 2*PI/3
    let scanline_abs = abs(scanline);
    let scanline_effect = scanline_abs * scanline_abs * scanline_abs; // Fast pow(x, 3)
    final_color = vec4f(final_color.xyz * (1.0 - SCANLINE_INTENSITY * scanline_effect), final_color.a);

    let vignette = 1.0 - VIGNETTE_STRENGTH * dist * SQRT2;
    final_color = vec4f(final_color.xyz * vignette, final_color.a);

    let clamped_color = clamp(final_color, vec4f(0.0), vec4f(1.0));
    let corrected_color = pow(clamped_color.xyz, vec3f(1.0 / 0.5));

    textureStore(screen, id.xy, vec4f(corrected_color, 1.0));
}