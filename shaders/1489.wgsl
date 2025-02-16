const AA: i32 = 4;
const PI: f32 = 3.14159265;
const MAX_ITER: i32 =355;

fn osc(minValue: f32, maxValue: f32, interval: f32, currentTime: f32) -> f32 {
    return minValue + (maxValue - minValue) * 0.5 * (sin(2.0 * PI * currentTime / interval) + 1.0);
}
fn implicit(c: vec2<f32>, trap1: vec2<f32>, trap2: vec2<f32>, currentTime: f32) -> vec4<f32> {
    var z: vec2<f32> = vec2<f32>(0.0, 0.0);
    var dz: vec2<f32> = vec2<f32>(1.0, 0.0);
    var trap1_min: f32 = 1e20;
    var trap2_min: f32 = 1e20;
    var i: i32 = 0;
    for (i = 0; i < MAX_ITER; i = i + 1) {
        dz = 2.0 * vec2<f32>(z.x * dz.x - z.y * dz.y, z.x * dz.y + z.y * dz.x) + vec2<f32>(1.0, 0.0);
        let xnew: f32 = z.x * z.x - z.y * z.y + c.x;
        z.y = 2.0 * z.x * z.y + c.y;
        z.x = xnew;
        let dampenedTime: f32 = currentTime * 0.001;
        z = z + 0.1 * vec2<f32>(sin(0.001 * dampenedTime), cos(0.001 * dampenedTime));
        // Orbit traps: https://iquilezles.org/articles/ftrapsgeometric/
        trap1_min = min(trap1_min, length(z - trap1));
        trap2_min = min(trap2_min, dot(z - trap2, z - trap2));

        if (dot(z, z) > 3.14) {
            break;
        }
    }
    let d: f32 = sqrt(dot(z, z) / dot(dz, dz)) * log(dot(z, z));
    return vec4<f32>(f32(i), d, trap1_min, trap2_min);
}
fn gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size: vec2<u32> = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) {
        return;
    }
    let fragCoord: vec2<f32> = vec2<f32>(f32(id.x) + 0.5, f32(screen_size.y) - f32(id.y) - 0.5);
    let screen_size_f32: vec2<f32> = vec2<f32>(f32(screen_size.x), f32(screen_size.y));
    let uv_base: vec2<f32> = 0.4 * (fragCoord - 0.5 * screen_size_f32) / screen_size_f32.y;
    let camSpeed: vec2<f32> = vec2<f32>(0.0002, 0.0002);
    let camPath: vec2<f32> = vec2<f32>(sin(camSpeed.x * time.elapsed / 10.0), cos(camSpeed.y * time.elapsed / 10.0));
    var pan: vec2<f32> = vec2<f32>(0.8076, 0.2606);
    if (time.elapsed > 2.0) {
        let timeSince14: f32 = time.elapsed - 45.0;
        pan.y = pan.y + 0.00002 * timeSince14;
    }
    let zoomLevel: f32 = osc(0.0005, 0.0005, 10.0,time.elapsed);
    let trap1: vec2<f32> = vec2<f32>(0.0, 1.0);
    let trap2: vec2<f32> = vec2<f32>(-0.5, 2.0) + 0.5 * vec2<f32>(cos(0.13 * time.elapsed), sin(0.13 * time.elapsed));
    var col: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    for (var m: i32 = 0; m < AA; m = m + 1) {
        for (var n: i32 = 0; n < AA; n = n + 1) {
            let sample_offset: vec2<f32> = vec2<f32>(f32(m), f32(n)) / f32(AA);
            let min_res: f32 = min(screen_size_f32.x, screen_size_f32.y);
            let uv_sample: vec2<f32> = ((fragCoord + sample_offset - 0.5 * screen_size_f32) / min_res * zoomLevel + pan + camPath) * 2.033 - vec2<f32>(2.14278, 2.14278);
            let z_data: vec4<f32> = implicit(uv_sample, trap1, trap2, time.elapsed);
            let iter_ratio: f32 = z_data.x / f32(MAX_ITER);
            let d: f32 = z_data.y;
            let trap1_dist: f32 = z_data.z;
            let trap2_dist: f32 = z_data.w;
            if (iter_ratio < 1.0) {
                let c1: f32 = pow(clamp(2.00 * d / zoomLevel, 0.0, 1.0), 0.5);
                let c2: f32 = pow(clamp(1.5 * trap1_dist, 0.0, 1.0), 2.0);
                let c3: f32 = pow(clamp(0.4 * trap2_dist, 0.0, 1.0), 0.25);
                let col1: vec3<f32> = 0.5 + 0.5 * sin(vec3<f32>(3.0) + 4.0 * c2 + vec3<f32>(0.0, 0.5, 1.0));
                let col2: vec3<f32> = 0.5 + 0.5 * sin(vec3<f32>(4.1) + 2.0 * c3 + vec3<f32>(1.0, 0.5, 0.0));
                let osc_val: f32 = osc(1.0, 12.0, 10.0,time.elapsed);
                let exteriorColor: vec3<f32> = 0.5 + 0.5 * sin(2.0 *trap2_dist + vec3<f32>(0.0, 0.5, 1.0) + PI * vec3<f32>(3.0 * iter_ratio) + osc_val);
                col = col +col1 +  exteriorColor;
            }
        }
    }

    col = gamma(col / f32(AA * AA),0.5);


    textureStore(screen, vec2<i32>(id.xy), vec4<f32>(col, 1.0));
}
