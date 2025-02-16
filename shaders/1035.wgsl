const PI: f32 = 3.141592653589793;
fn oscillate(mint: f32, maxt: f32, interval: f32, ct: f32) -> f32 {
    return mint + (maxt - mint) * 0.5 * (sin(2.0 * PI * ct / interval) + 1.0);
}
fn rotate2D(r: f32) -> mat2x2<f32> {
    return mat2x2<f32>(cos(r), -sin(r), sin(r), cos(r));
}
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);

    let t: f32 = time.elapsed/8.0;
    let xVal4: f32 =  oscillate(5.51, 5.51, 15.0, t);
    let xVal6: f32 = oscillate(5.51, 5.51, 15.0, t);
    let xVal7: f32 = oscillate(0.8, 0.8, 15.0, t);

    var n: vec2<f32> = vec2<f32>(0.0, 0.0);
    var N: vec2<f32> = vec2<f32>(0.2, 0.5) * 2.2;
    var p: vec2<f32> = uv+t ;
    var S: f32 = xVal6;
    let m: mat2x2<f32> = rotate2D(xVal4);
    var bf: f32 = 1.45;

    for (var j: f32 = 1.0; j < 45.0; j += 1.0) {
        p = m * p;
        n = m * n;
        let q: vec2<f32> = m * p * S *j + n + vec2<f32>(t, t);
        
        n += bf * sin(q)+t;
        N += bf * cos(q) / S * xVal7;
        bf *= 1.3 * tanh(0.975);
        S *= 1.22 * tanh(1.975);
    }
    let c: vec3<f32> = vec3<f32>(
    1.5 * cos(1.0 *N.x), 
    0.5 * sin(2.0 * N.y), 
    1.5 * cos(1.0 *N.y)
    );

    var col: vec3<f32> = (vec3<f32>(0.0 , 25.5, 0.0) *
    0.01 + c) * 
    ((1.0 * N.x * 0.5 * N.y* N.y* N.y* N.y + 0.0001) + .0001 / length(1.0 * N));

    textureStore(screen, id.xy, vec4f(col, 1.0));
}