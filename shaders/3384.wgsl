#define BUFFER_A 1
#define RES 100.0
#calcdefine RES_UV RES / SCREEN_WIDTH

fn sdCircle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdRhombus(p: vec2f, b: vec2f) -> f32 {
    let n_b = vec2f(b.x, -b.y);
    var n_p = abs(p);

    let h = clamp(
        (dot(n_b, n_p) + n_b.y * n_b.y) / dot(n_b, n_b),
        0.0,
        1.0
    );

    n_p -= n_b * vec2f(h, h - 1.0);
    return length(n_p) * sign(n_p.x);
}

fn op_smooth_union(d1: f32, d2: f32, k: f32) -> f32 {
    let h = max(k - abs(d1 - d2), 0.0) / k;
    return min(d1, d2) - h * h * k * 0.25;
}

fn bilinear_interpolation(uv: vec2f) -> f32 {

    let grid_pos = uv * f32(SCREEN_WIDTH) / RES;

    let p = floor(grid_pos);
    let f = fract(grid_pos);

    let p00 = vec2i(p) * i32(RES);
    let p10 = p00 + vec2i(i32(RES), 0);
    let p01 = p00 + vec2i(0, i32(RES));
    let p11 = p00 + vec2i(i32(RES), i32(RES));

    let d00 = textureLoad(pass_in, p00, BUFFER_A, 0).r;
    let d10 = textureLoad(pass_in, p10, BUFFER_A, 0).r;
    let d01 = textureLoad(pass_in, p01, BUFFER_A, 0).r;
    let d11 = textureLoad(pass_in, p11, BUFFER_A, 0).r;

    let d0 = mix(d00, d10, f.x);
    let d1 = mix(d01, d11, f.x);

    return mix(d0, d1, f.y);
}

@compute @workgroup_size(16, 16)
fn grid_pass(@builtin(global_invocation_id) id: vec3u) {

    let grid_point =
        (id.x % u32(RES) == 0u) &&
        (id.y % u32(RES) == 0u);

    if !grid_point {
        return;
    }

    let uv = vec2f(id.xy) / f32(SCREEN_WIDTH);

    let d_circle = sdCircle(uv - vec2f(0.2, 0.28), 0.1);

    let d_rhombus = sdRhombus(
        uv - vec2f(0.7, 0.28),
        vec2f(0.15, 0.15)
    );

    let k = max(
        (1.0 + sin(time.elapsed)) / 2.0 * 0.6,
        0.001
    );

    let d = op_smooth_union(d_circle, d_rhombus, k);

    textureStore(
        pass_out,
        vec2i(id.xy),
        BUFFER_A,
        vec4f(d, 0.0, 0.0, 1.0)
    );
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {

    let uv = vec2f(id.xy) / f32(SCREEN_WIDTH);


    let d = bilinear_interpolation(uv);

    // Color
    var col = select(
        vec3f(0.65, 0.85, 1.0),
        vec3f(0.9, 0.6, 0.3),
        d > 0.0
    );

    col *= 1.0 - exp(-8.0 * abs(d));
    col *= 0.5 + 0.2 * select(
        0.0,
        1.0,
        cos(400.0 * d) > 0.0
    );
    col = mix(
        col,
        vec3f(1.0),
        1.0 - smoothstep(0.0, 0.005, abs(d))
    );

    textureStore(screen, vec2i(id.xy), vec4f(col, 1.0));
}