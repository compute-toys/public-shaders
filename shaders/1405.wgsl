const PI: f32 = 3.14159265359;
const TAU: f32 = 2.0 * PI;
fn applyGamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}
fn oscillate(minValue: f32, maxValue: f32, interval: f32, currentTime: f32) -> f32 {
    return minValue + (maxValue - minValue) * 0.5 * (sin(2.0 * PI * currentTime / interval) + 1.0);
}

fn cliffordAttractor(p: vec2<f32>, a: f32, b: f32, c: f32, d: f32) -> vec2<f32> {
    let x = sin(a * p.y) + c * cos(a * p.x);
    let y = sin(b * p.x) + d * cos(b * p.y);
    return vec2<f32>(x, y);
}

@compute @workgroup_size(16, 16)
fn buffer_a(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2<f32>(f32(id.x) + 0.5, f32(screen_size.y - id.y) - 0.5);
    let uv = 7.0 * (2.0 * fragCoord - vec2<f32>(screen_size)) / f32(screen_size.y);

    let numCircles = 12;
    let numPoints = 15;
    let xVal4444 = oscillate(4.5, 4.5, 5.0, time.elapsed);
    let ddsad = oscillate(0.02, 0.01, 12.0, time.elapsed);
    let cR = xVal4444 / f32(numCircles);
    var color = vec3<f32>(0.0);

    // Clifford attractor parameters
    let a = -1.8;
    let b = -2.0;
    let c = -0.5;
    let d = -0.9;
    let scale = 3.0;

    for(var i = 0; i < numCircles; i++) {
        for(var j = 0; j < numPoints; j++) {
            let t = f32(j) / f32(numPoints) * TAU + time.elapsed * 0.1;
            let initialPoint = vec2<f32>(cos(t), sin(t)) * f32(i+1) * cR * 0.2;

            var attractorPoint = initialPoint;
            for(var k = 0; k < 10; k++) {
                attractorPoint = cliffordAttractor(attractorPoint, a, b, c, d);
            }

            let circlePoint = attractorPoint * scale;
            let pointColor = 0.5 + 0.5 * sin(vec3<f32>(1.0, TAU/3.0, TAU*2.0/3.0) + f32(i) * 0.87);
            let dist = length(uv - circlePoint);
            color += pointColor * ddsad / dist;
        }
    }

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    color = sqrt(color) * 2.0 - 1.0;

    let feedback = textureLoad(pass_in, vec2<i32>(id.xy), 1, 0);
    color = mix(color, feedback.rgb, 0.5);
    color = applyGamma(color, 0.5);

    textureStore(pass_out, vec2<i32>(id.xy), 0, vec4<f32>(color, 1.0));
}

@compute @workgroup_size(16, 16)
fn buffer_b(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let uv = vec2<f32>(id.xy) / vec2<f32>(screen_size);

    let prevFrame = textureLoad(pass_in, vec2<i32>(id.xy), 1, 0);

    let currentFrame = textureLoad(pass_in, vec2<i32>(id.xy), 0, 0);

    let decay = 0.95;

    let result = mix(prevFrame * decay, currentFrame, 0.5);

    textureStore(pass_out, vec2<i32>(id.xy), 1, result);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let result = textureLoad(pass_in, vec2<i32>(id.xy), 1, 0);
    textureStore(screen, vec2<i32>(id.xy), result);
}