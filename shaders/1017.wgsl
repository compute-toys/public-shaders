
//based: https://www.shadertoy.com/view/DtXyWM
const PI: f32 = 3.141592653589793;
fn oscillate(minValue: f32, maxValue: f32, interval: f32, currentTime: f32) -> f32 {
    return minValue + (maxValue - minValue) * 0.5 * (sin(2.0 * PI * currentTime / interval) + 1.0);
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

   let t: f32 = time.elapsed;
   //adjustable params:
    let xVal4: f32 = oscillate(0.5, 0.5, 15.0, t);
    //adjust this for osc. zoom
    let xVal6: f32 = oscillate(0.8, 3.8, 15.0, t);
    let xVal7: f32 = oscillate(0.5, 0.51, 10.0, t);

    var p: vec2<f32> = uv;
    var n: vec2<f32> = vec2<f32>(0.0, 0.0);
    var N: vec2<f32> = vec2<f32>(1.5, 1.5);
    let m: mat2x2<f32> = rotate2D(xVal4);
    var S: f32 = xVal6;
    var branchFactor: f32 = 1.78;

    for (var j: f32 = 0.0; j < 45.0; j += 2.0) {
        p = p * m;
        n = n * m;
        let q: vec2<f32> = p * S * j + n + vec2<f32>(t/2.0, t/1.5); //static view: (change t values)

        n += branchFactor * sin(q);
        N += branchFactor * cos(q) / S * xVal7;
        S *= 1.45 * tanh(2.975);
    }
    let baseColor: vec3<f32> = vec3<f32>(0.1, 0.2, 0.5); 
    let colorVariation: vec3<f32> = vec3<f32>(
        0.4 * sin(0.0 + N.x*N.y),
        0.5 * sin(0.4 + N.x*N.y),
        0.8 * cos(0.2 + N.x*N.y)
    );
    var col: vec3<f32> = baseColor + colorVariation;
    let complementaryColor: vec3<f32> = vec3<f32>(0.1, 0.2, 0.1); 
    let complementaryVariation: f32 = 1.5 + 0.0 * sin(2.*PI * uv.x * uv.y + t); //0.0 * (the custom...)
    //let xVal1: f32 = oscillate(0.2, 0.001, 5.0, t);
    //remove if you feel awkard :-P (update: I removed :d)
    //let distanceFromCenter: f32 = length(uv - vec2<f32>(0.5, 0.5));
    let complementaryIntensity: f32 = smoothstep(0.1, 0.35, 1.0);
    col = mix(col, complementaryColor, complementaryVariation * complementaryIntensity);


    textureStore(screen, id.xy, vec4f(col, 1.0));

}