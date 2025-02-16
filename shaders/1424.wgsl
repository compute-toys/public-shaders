const PI: f32 = 3.14159265359;
const TAU: f32 = 2.0 * PI;
fn applyGamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}

fn oscillate(minValue: f32, maxValue: f32, interval: f32, currentTime: f32) -> f32 {
    return minValue + (maxValue - minValue) * 0.5 * (sin(2.0 * PI * currentTime / interval) + 1.0);
}

@compute @workgroup_size(16, 16)
fn buffer_a(@builtin(global_invocation_id) id: vec3<u32>) {
    let resolution = vec2<f32>(textureDimensions(screen));
    let fragCoord = vec2<f32>(id.xy) + 0.5;
    let uv = 32.0 * (2.0 * fragCoord - resolution) / resolution.y;

    let numCircles = 6;
    let numPoints = 12;
    let xVal1 = oscillate(0.0, 1.0, 15.0, time.elapsed);
    let xVal2 = oscillate(12.0, 12.0, 5.0, time.elapsed);
    let xVal3 = oscillate(0.0, 2.0, 12.0, time.elapsed);
    let xVal4444 = oscillate(4.5, 4.5, 5.0, time.elapsed);
    let ddsad = oscillate(0.08, 0.08, 12.0, time.elapsed);
    let frequency = xVal1;
    let amplitude = xVal2;
    let phase = time.elapsed * 1.1;
    let CR = xVal4444 / f32(numCircles);
    var color = vec3<f32>(0.0);

    let sf = oscillate(0.5, 0.5, 8.0, time.elapsed);
    let wf = oscillate(1.0, 1.0, 10.0, time.elapsed);

    for(var i: i32 = 0; i < numCircles; i++) {
        for(var j: i32 = 0; j < numPoints; j++) {
            let radius = f32(i) * CR;
            var CPATH: vec2<f32>;

            let angle = phase + TAU * f32(j) / f32(numPoints);
            CPATH.x = sin(angle) * radius;
            CPATH.y = cos(angle) * radius;

            CPATH.x += sin(4.0 * PI * frequency * f32(j) / f32(numPoints) + phase) * amplitude;
            CPATH.y += cos(4.0 * xVal3 * PI * frequency * f32(j) / f32(numPoints) + phase) * amplitude;

            let angleeeeeeee = atan2(CPATH.y, CPATH.x);
            let sRadi = length(CPATH);
            let spiralOffset = vec2<f32>(
                cos(angleeeeeeee + sRadi * sf),
                sin(angleeeeeeee + sRadi * sf)
            ) * amplitude * 0.3;
            CPATH += spiralOffset;

            let twist = sin(wf * sRadi + time.elapsed);
            let MATRIX = mat2x2<f32>(
                cos(twist), -sin(twist),
                sin(twist), cos(twist)
            );
            CPATH = MATRIX * CPATH;
            let heyy = oscillate(0.55, 0.89, 4.0, time.elapsed);

            let poCAaaa = 0.7 + 0.7 * sin(vec3<f32>(1.0, TAU/3.0, TAU*2.0/3.0) + f32(i) * heyy);
            let dist = length(uv - CPATH);
            color += poCAaaa * ddsad / dist;

        }
    }

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));


    let feedback = textureLoad(pass_in, vec2<i32>(id.xy), 1, 0);
    color = applyGamma(color, 0.5);

    color = mix(color, feedback.rgb, 0.4);

    textureStore(pass_out, vec2<i32>(id.xy), 0, vec4<f32>(color, 1.0));

}

@compute @workgroup_size(16, 16)
fn buffer_b(@builtin(global_invocation_id) id: vec3<u32>) {
    let resolution = vec2<f32>(textureDimensions(screen));
    let uv = vec2<f32>(id.xy) / resolution;
    let prevFrame = textureLoad(pass_in, vec2<i32>(id.xy), 1, 0);
    let currentFrame = textureLoad(pass_in, vec2<i32>(id.xy), 0, 0);
    let decay = 0.95;
    
    let result = mix(prevFrame * decay, currentFrame, 0.2);

    textureStore(pass_out, vec2<i32>(id.xy), 1, result);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let result = textureLoad(pass_in, vec2<i32>(id.xy), 1, 0);
    textureStore(screen, vec2<i32>(id.xy), result);
}