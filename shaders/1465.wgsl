
fn applyGamma(color: vec4<f32>, gamma: f32) -> vec4<f32> {
    return vec4<f32>(pow(color.rgb, vec3<f32>(1.0 / gamma)), color.a);
}
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let R = vec2<f32>(textureDimensions(screen));
    if (id.x >= u32(R.x) || id.y >= u32(R.y)) { return; }
    let u = vec2<f32>(f32(id.x) + 0.5, f32(id.y) + 0.5);
    let uuu = 5.0 * (u + u - R) / R.y;
    let f = fract(uuu) - 0.5;
    var O = vec4<f32>(0.7, 0.8, 0.1, 1.0);
    let R_ceil = ceil(uuu);
    let star = abs(f.x) + abs(f.y) - 0.15 * abs(sin(atan2(f.y, f.x) * 4.0));
    if (star < 0.4) {
        O += vec4<f32>(4.0 * dot(cos(R_ceil.x + R_ceil.y + vec2<f32>(0.0, 11.0)), f));
    }
    if (star < 0.25) {
        O = vec4<f32>(0.3, 0.3, 1.0, 1.0);
    }
    O = applyGamma(O, 0.4); 
    textureStore(screen, vec2<i32>(id.xy), O);
}