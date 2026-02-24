@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    let uv = fragCoord / vec2f(screen_size.yy);

    var noise_value = remap(
        pattern(vec3f((uv.xy + vec2((uv.y)*snoise3(vec3f(time.elapsed/4, 0, 0)), -time.elapsed*1.125))*1.5, time.elapsed/1.5)),
        -1,
        1,
        0,
        1
    );

    noise_value = noise_value * pow(remap(uv.y, 0, 1, 1, 0), 0.65);
    noise_value = noise_value * remap(distance(uv.x, f32(screen_size.x)/f32(screen_size.y)/2), 0.25, 0.5, 1, 0);

    noise_value = pow(noise_value, 5);

    // Time varying pixel colour
    var col = colorRamp(pow(noise_value, 2.2));

    // Convert from gamma-encoded to linear colour space
    // col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn colorRamp(t: f32) -> vec3<f32> {
    let clamped_t = clamp(t, 0.0, 1.0);

    // Define color stops and positions (e.g., black at 0.0, red at 0.5, white at 1.0)
    let stops = array<f32, 3>(0.0, 0.5, 1.0);
    let colors = array<vec4<f32>, 4>(
        vec4<f32>(0.0, 0.0, 0.0, 1.0), // Black
        vec4<f32>(247.0/255.0, 25.0/255.0, 37.0/255.0, 1.0),
        vec4<f32>(247.0/255.0, 170.0/255.0, 37.0/255.0, 1.0), // Red
        vec4<f32>(1.0, 1.0, 1.0, 1.0)  // White
    );

    for (var i = 0u; i < 3u; i = i + 1u) {
        if (clamped_t >= stops[i] && clamped_t <= stops[i+1]) {
            // Calculate interpolation factor (0.0 to 1.0) for the current segment
            let segment_start = stops[i];
            let segment_end = stops[i+1];
            let segment_t = (clamped_t - segment_start) / (segment_end - segment_start);
            // Interpolate between the two adjacent colors
            return mix(colors[i], colors[i+1], segment_t).xyz;
        }
    }
    // Fallback for t=1.0 edge case due to loop condition
    return colors[2].xyz;
}

fn pattern( v: vec3f ) -> float
{
    let p = v.xy;
    let q = vec2f( snoise3_fract( vec3f(p + vec2(0.0,0.0), v.z) ),
                   snoise3_fract( vec3f(p + vec2(5.2,1.3), v.z ) ) );

    return snoise3_fract( vec3f(p + 4.0*q,v.z) );
}

fn remap(value: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    return ((value - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min;
}

fn mod289_3(x: vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_4(x: vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute(x: vec4<f32>) -> vec4<f32> {
    return mod289_4(((x * 34.0) + 1.0) * x);
}

fn taylorInvSqrt(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn snoise3(v: vec3<f32>) -> f32 {
    let C = vec2<f32>(1.0/6.0, 1.0/3.0);
    let D = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    // First corner
    var i  = floor(v + dot(v, C.yyy));
    let x0 = v - i + dot(i, C.xxx);

    // Other corners
    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);

    let x1 = x0 - i1 + C.xxx;
    let x2 = x0 - i2 + C.yyy;
    let x3 = x0 - D.yyy;

    // Permutations
    i = mod289_3(i);
    let p = permute(permute(permute(
        i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
        + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0))
        + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));

    // Gradients
    let n_ = 0.142857142857; // 1.0/7.0
    let ns = n_ * D.wyz - D.xzx;

    let j = p - 49.0 * floor(p * ns.z * ns.z);

    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_);

    let x = x_ * ns.x + ns.yyyy;
    let y = y_ * ns.x + ns.yyyy;
    let h = 1.0 - abs(x) - abs(y);

    let b0 = vec4<f32>(x.xy, y.xy);
    let b1 = vec4<f32>(x.zw, y.zw);

    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, vec4<f32>(0.0));

    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;

    var p0 = vec3<f32>(a0.xy, h.x);
    var p1 = vec3<f32>(a0.zw, h.y);
    var p2 = vec3<f32>(a1.xy, h.z);
    var p3 = vec3<f32>(a1.zw, h.w);

    // Normalise gradients
    let norm = taylorInvSqrt(vec4<f32>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    var m = max(0.6 - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4<f32>(0.0));
    m = m * m;
    return 42.0 * dot(m * m, vec4<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

fn snoise3_fract(m: float3) -> float
{
    return   0.5333333*snoise3(m*rot1)
            +0.2666667*snoise3(2.0*m*rot2)
            +0.1333333*snoise3(4.0*m*rot3)
            +0.0666667*snoise3(8.0*m);
}

const rot2 = mat3x3<f32>(-0.55,-0.39, 0.74, 0.33,-0.91,-0.24,0.77, 0.12,0.63);
const rot1 = mat3x3<f32>(-0.37, 0.36, 0.85,-0.14,-0.93, 0.34,0.92, 0.01,0.4);
const rot3 = mat3x3<f32>(-0.71, 0.52,-0.47,-0.08,-0.72,-0.68,-0.7,-0.45,0.56);