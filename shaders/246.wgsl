#include <string>

const pi = acos(-1.0);

#storage write_totals array<atomic<u32>>
#storage read_totals array<u32>

fn print(text: String, coord: vec2f, size: f32, bold: f32) -> f32 {
    let zoom = 1.42;

    if any(coord < vec2f(0, -zoom)) || any(coord > size * vec2f(f32(text.len), 1.2)) {
        return 1;
    }

    let p = coord / size;
    let ch = text.chars[u32(p.x)];
    if ch < 0x21 || ch > 0x7f {
        return 1;
    }

    let off = vec2f(vec2u(ch % 16u, ch / 16u));
    let uv = ((vec2f(fract(p.x), p.y) - 0.5) / zoom + 0.5 + off) / 16.0;
    let sdf = textureSampleLevel(channel0, trilinear, uv, 0).a;

    let aa = 0.5 / (size * zoom);
    let base = 0.5 + bold;
    return smoothstep(base - aa, base + aa, sdf);
}

fn panel_one(coord: vec2f, R: vec2f) -> f32 {
    let v = coord - vec2f(R.x - R.y / 2.0, R.y / 2.0);
    let r = length(v);
    let a = acos(dot(normalize(v), normalize(vec2f(-3, 2))) + 1e-6);

    let d = r - 0.23 * R.x;
    var col = smoothstep(0, 1, abs(d) - 1.0);

    if d < 0 {
        let sum = read_totals[0]
            + read_totals[1]
            + read_totals[2]
            + read_totals[3];

        let value = f32(sum) / f32(SCREEN_WIDTH * SCREEN_HEIGHT);
        col *= smoothstep(0, 1 / r, a - pi * value);
    }

    let s = R.y / 11;
    let g = R.y / 8;
    var p = coord;
    col *= print("Fraction of", p, s, 0); p.y -= g;
    col *= print("shader which", p, s, 0); p.y -= g;
    col *= print("is white", p, s, 0); p.y -= 3.2 * g;
    col *= print("Fraction of", p, s, 0); p.y -= g;
    col *= print("shader which", p, s, 0); p.y -= g;
    col *= print("is black", p, s, 0);

    return col;
}

fn panel_two(coord: vec2f, R: vec2f) -> f32 {
    var col = 1.0;

    if any(coord < vec2f(0)) || any(coord > R) {
        return 1.0;
    }

    let s = R.y / 11;
    let g = R.y / 8;
    var p = coord;
    col *= print("Number of black", p, s, 0); p.y -= g;
    col *= print("pixels by panel:", p, s, 0);

    let d = abs(fract(6.0 * coord.y / R.y + 0.5) - 0.5);
    if d < 0.2 { return col; }

    let k = i32(floor(6.0 * coord.y / R.y)) - 2;
    if k < 0 { return col; }

    let sum = read_totals[k];
    let top = max(max(read_totals[0], read_totals[1]), max(read_totals[2], read_totals[3]));

    let val = f32(sum) / f32(top);
    if val > coord.x / R.x {
        col = 0;
    }

    return col;
}

fn panel_three(coord: vec2f, R: vec2f) -> f32 {
    var col = 1.0;

    let s = R.y / 11;
    let g = R.y / 8;
    var p = coord;
    col *= print("Location of black", p, s, 0); p.y -= g;
    col *= print("pixels in", p, s, 0); p.x -= 10 * s;
    col *= print("this shader:", p, s, 0);

    let RR = vec2f(SCREEN_WIDTH, SCREEN_HEIGHT);
    let tex = 1.2 * RR.y * (coord - R / 2) / R.y + RR / 2 - vec2f(0, 4.3 * g);
    const SAMPLES = 5;

    if all(tex > vec2f(SAMPLES)) && all(tex < RR - SAMPLES) {
        var acc = 0.0;
        for(var k = 0; k < SAMPLES * SAMPLES; k++) {
            let off = vec2f(vec2i(k % SAMPLES, k / SAMPLES)) - SAMPLES / 2.0;
            acc += passLoad(0, vec2i(tex + off), 0).x;
        }
        col *= acc / (SAMPLES * SAMPLES);
    }

    return col;
}

fn panel_four(coord: vec2f, R: vec2f) -> f32 {
    if any(coord < vec2f(0)) || any(coord > R) {
        return 1.0;
    }

    var col = 1.0;

    let s = R.y / 11;
    let g = R.y / 8;
    let G = R.y / 5;
    var p = coord;
    let b = 0.022;
    col *= print("Self-Descriptive", p, s, b);
    col *= print("Shader", p - vec2f(17 * s, 0), s, b); p.y -= g;
    col *= print("by slerpy", p, s, 0); p.y -= G;
    col *= print("compute.toys", p, s, b);
    col *= print("/view/246", p - vec2f(12 * s, 0), s, 0);
    p.y -= G;

    if p.y < 0 {
        return col;
    }

    return step(0.5, (fract(11 * (p.x + p.y) / R.x)));
}

const buckets = 64u;

#workgroup_count init 1 1 1
@compute @workgroup_size(4)
fn init(@builtin(global_invocation_id) id: vec3u) {
    var acc = 0u;
    for (var k = 0u; k < buckets; k++) {
        let idx = id.x + 4 * k;
        acc += atomicExchange(&write_totals[idx], 0);
    }

    let val = &read_totals[id.x];
    *val += acc;
    *val /= 2;
}

@compute @workgroup_size(16, 16)
fn render(@builtin(global_invocation_id) id: vec3u) {
    let R = textureDimensions(screen);
    if any(id.xy >= R) { return; }

    let margin = round(f32(R.y) / 16.0);
    let base = vec2f(id.xy) - margin / 2;
    let res = vec2f(R) / 2.0 - margin / 2;

    let b = base > res;
    let k = 2 * u32(b.y) + u32(b.x);
    let coord = base - vec2f(b) * res;

    var col = 1.0;
    switch k {
        case 0 { col *= panel_one(coord - margin, res - 2 * margin); }
        case 1 { col *= panel_two(coord - margin, res - 2 * margin); }
        case 2 { col *= panel_three(coord - margin, res - 2 * margin); }
        case 3 { col *= panel_four(coord - margin, res - 2 * margin); }
        default {}
    }

    let s = (res - margin) / 2;
    let v = abs(coord - margin / 2.0 - s) - s;
    let d = abs(max(v.x, v.y)) - 1.0;
    col *= smoothstep(0, 1, d);

    if col < 0.5 {
        let off = (id.x + id.y) % buckets;
        let idx = k + 4 * off;
        atomicAdd(&write_totals[k], 1u);
    }

    passStore(0, vec2i(id.xy), vec4f(col));
    textureStore(screen, id.xy, vec4f(col));
}
