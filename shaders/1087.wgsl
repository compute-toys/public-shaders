const PI: f32 = 3.14;
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = vec2<f32>(896.0, 504.0); 

    if (id.x >= u32(screen_size.x) || id.y >= u32(screen_size.y)) { return; }

    let fragCoord = vec2<f32>(f32(id.x) + 0.5, screen_size.y - f32(id.y) - 0.5);

    var uv = 15.0 * (fragCoord / screen_size);

    uv = uv - vec2<f32>(10.0, 10.0); 
    uv = uv / max(screen_size.x / screen_size.y, screen_size.y / screen_size.x);

    var col: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    let frequency: f32 = 0.0;

    for (var j: f32 = 0.0; j < 5.2; j += 1.0) {
        for (var i: f32 = 1.0; i < 5.0; i += 1.0) {
            uv.x = uv.x + (0.2 / (i + j) * sin(i * atan(time.elapsed) * 2.0 * uv.y + (time.elapsed * 0.1) + i * j));
            uv.y = uv.y + (1.0 / (i + j) * cos(i * 0.6 * uv.x + (time.elapsed * 0.12) + i * j));
            let frequency: f32 = 0.0; 
            var angle: f32 = time.elapsed * 0.01; 
            let rotation: mat2x2<f32> = mat2x2<f32>(cos(angle), -sin(angle), sin(angle), cos(angle));
        uv = rotation * uv;
        }
        var texColor = textureSampleLevel(channel0, bilinear, uv.xy, 5).rgb;
        let lenSq: f32 = atan(uv.x);
        let col1: vec3<f32> = 0.1 + 3.5 * cos(frequency * (1.0 + time.elapsed) + vec3<f32>(0.0, 0.5, 5.0) + PI * vec3<f32>(1.0 * lenSq));
        let col2: vec3<f32> = 1.2 + 2.1 * cos(frequency * (1.1 + time.elapsed) + PI * vec3<f32>(lenSq));
        let col3: vec3<f32> = 0.2 + 3.1 * cos(frequency * (1.0 + time.elapsed) + vec3<f32>(1.0, 0.5, 0.0) + PI * vec3<f32>(1.0 * sin(lenSq)));
        col += texColor * (col1 + col2 + col3 + col3);
    }
    col /= 9.0;

    let bg: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
    col = mix(col, bg, 1.0 - smoothstep(0.0, 1.0, length(uv) - 0.1));

    textureStore(screen, vec2<i32>(id.xy), vec4<f32>(col, 1.0));
}