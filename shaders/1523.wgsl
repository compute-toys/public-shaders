fn sRGB(linear: vec4<f32>) -> vec4<f32> {
    return select(
        12.92 * linear,
        1.055 * pow(linear, vec4<f32>(1.0 / 2.4)) - 0.055,
        linear > vec4<f32>(0.04045 / 12.92)
    );
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
    var color = vec3<f32>(0.0);

    var u = f32(id.x) / f32(SCREEN_WIDTH);
    var expectedValue = i32(trunc(u * 256.0));
    var false_linear = textureLoad(channel0, vec2<i32>(expectedValue, 0), 0);
    var false_sRGB = sRGB(false_linear);
    var actualValue = i32(round(false_sRGB.r * 255.0));

    if (actualValue == expectedValue) {
        color.g = 1.0;
    } else {
        color.r = 1.0;
    }

    textureStore(screen, id.xy, vec4<f32>(color, 1.0));
}