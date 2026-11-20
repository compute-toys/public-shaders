#define BUFFER_A 1
#define BUFFER_B 2
#define RES 100
#define WORKGROUP_SIZE_GRID 16
#calcdefine INVOCATION_X SCREEN_WIDTH / RES / WORKGROUP_SIZE_GRID + 1
#calcdefine INVOCATION_Y SCREEN_HEIGHT / RES / WORKGROUP_SIZE_GRID +1

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

fn rotate2D(p: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);

    let rot = mat2x2<f32>(
        vec2<f32>(c, -s),
        vec2<f32>(s, c)
    );
    return rot * p;
}

fn bilinear_interpolation(uv: vec2f) -> f32 {

    let grid_pos = uv * f32(SCREEN_WIDTH) / f32(RES);

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

#workgroup_count grid_pass INVOCATION_X INVOCATION_Y 1
@compute @workgroup_size(WORKGROUP_SIZE_GRID, WORKGROUP_SIZE_GRID)
fn grid_pass(@builtin(global_invocation_id) id: vec3u) {

    let pixel = id.xy * u32(RES);

    let uv = vec2f(pixel) / f32(SCREEN_WIDTH);

    let d_circle = sdCircle(uv - vec2f(0.2, 0.28), 0.1);

    let d_rhombus = sdRhombus(
        rotate2D(uv - vec2f(0.7, 0.28),time.elapsed),
        vec2f(0.15, 0.1)
    );

    let k = max(
        (1.0 + sin(time.elapsed)) / 2.0 * 0.7,
        0.001
    );

    let d = op_smooth_union(d_circle, d_rhombus, k);

    textureStore(
        pass_out,
        vec2i(pixel),
        BUFFER_A,
        vec4f(d, 0.0, 0.0, 1.0)
    );
}

@compute @workgroup_size(16,16)
fn normal_raymarching(@builtin(global_invocation_id) id: vec3u) {


    let uv = vec2f(id.xy) / f32(SCREEN_WIDTH);

    let d_circle = sdCircle(uv - vec2f(0.2, 0.28), 0.1);

    let d_rhombus = sdRhombus(
        rotate2D(uv - vec2f(0.7, 0.28),time.elapsed),
        vec2f(0.15, 0.1)
    );

    let k = max(
        (1.0 + sin(time.elapsed)) / 2.0 * 0.75,
        0.001
    );

    let d = op_smooth_union(d_circle, d_rhombus, k);

    textureStore(
        pass_out,
        vec2i(id.xy),
        BUFFER_B,
        vec4f(d, 0.0, 0.0, 1.0)
    );
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {

    let uv = vec2f(id.xy) / f32(SCREEN_WIDTH);


    let d = bilinear_interpolation(uv);

    let d_comp = textureLoad(pass_in, id.xy, BUFFER_B, 0).r;

    // Color
    var col = select(
        vec3f(0.65, 0.85, 1.0),
        vec3f(0.9, 0.6, 0.3),
        d_comp > 0.0
    );

    col *= 1.0 - exp(-8.0 * abs(d_comp));
    col *= 0.5 + 0.2 * select(
        0.0,
        1.0,
        cos(400.0 * d_comp) > 0.0
    );
    col = mix(
        col,
        vec3f(1.0),
        1.0 - smoothstep(0.0, 0.001, abs(d_comp))
    );

        col = mix(
        col,
        vec3f(1.0, 1.0, 0.0),
        1.0 - smoothstep(0.0, 0.002, abs(d))
    );

    textureStore(screen, vec2i(id.xy), vec4f(col, 1.0));
    // textureStore(screen, vec2i(id.xy), vec4f(vec3f(d), 1.0));

}