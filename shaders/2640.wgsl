fn hash3(p: vec3f) -> vec3f {
    var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    p3 = p3 + dot(p3, p3.yzx + 19.19);
    return fract(vec3f(p3.x + p3.y, p3.y + p3.z, p3.z + p3.x) * p3.zxy);
}

// Ultra-fast hash (fewer operations)
fn hash_fast(p: vec3<f32>) -> f32 {
    let p3 = fract(p * 0.1031);
    return fract(dot(p3, p3.yzx + 33.33));
}

// Simplified noise using the fast hash
fn noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    
    // Reduce to 4 corner samples (2D-like approach)
    let n00 = hash_fast(i);
    let n10 = hash_fast(i + vec3<f32>(1.0, 0.0, 0.0));
    let n01 = hash_fast(i + vec3<f32>(0.0, 1.0, 0.0));
    let n11 = hash_fast(i + vec3<f32>(1.0, 1.0, 0.0));
    
    return (mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y) * 2.0 - 1.0);
}

// Fractal Brownian Motion - pure function
fn fbm(p: vec3<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    
    // 6 octaves (hardcoded for pure function)
    // Persistence = 0.5, Lacunarity = 2.0
    
    value += amplitude * noise(p * frequency);
    amplitude *= 0.5;
    frequency *= 2.0;
    
    value += amplitude * noise(p * frequency);
    amplitude *= 0.5;
    frequency *= 2.0;
    
    value += amplitude * noise(p * frequency);
    amplitude *= 0.5;
    frequency *= 2.0;
    
    value += amplitude * noise(p * frequency);
    amplitude *= 0.5;
    frequency *= 2.0;
    
    value += amplitude * noise(p * frequency);
    amplitude *= 0.5;
    frequency *= 2.0;
    
    value += amplitude * noise(p * frequency);
    
    return value;
}

fn rot2D(a: f32) -> mat2x2<f32> {
    let s: f32 = sin(a);
    let c: f32 = cos(a);

    return mat2x2(c, -s, s, c);
}

fn sdSphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

// Displace your sphere's SDF with noise that decreases with height
fn flameSDF(p: vec3f) -> f32 {
    let sphereDist = length(p) - 1.0;
    let flameHeight = max(0.0, -p.y);
    let noiseStrength: f32 = exp(-flameHeight * 2.0);
    let displacement = fbm(p * 2.0 + vec3f(0.0, time.elapsed * 2.0, 0.0)) * noiseStrength;
    return sphereDist - displacement * 0.3;
}

fn map(p: vec3<f32>) -> f32 {
    let spherePos = vec3<f32>(0.0, 0.0, 0.0);
    let sphereScale = 1.0; // ((sin(time.elapsed) + 1.0) / 2.0);
    let sphere = flameSDF(p);
    return sphere;

    // translated and scaled single sphere:
    // return sdSphere((q - spherePos) * sphereScale, 0.2) / sphereScale;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    
    // Normalised pixel coordinates (from 0 to 1)
    let uv = (fragCoord * 2.0 - vec2f(f32(screen_size.x), f32(screen_size.y))) / f32(screen_size.y);

    let fov = 3.0 * (sin(time.elapsed * 0.2) + 2.0) / 3.0;
    let ro = vec3f(0, 0, -3);
    let rd = normalize(vec3f(uv * fov, 1));
    var t = 0.0; // total distance traveled
    var col = vec3<f32>(0); // starts at black
    let surface_color = vec3f(1.0, 0.9, 0.8) * 1.5; // > 1.0 to be brighter
    let glow_color = vec3f(1.0, 0.4, 0.1); // orange 

    // Raymarching
    for (var i: i32 = 0; i < 100; i++) {
        var p = ro + rd * t; // position along the ray
        var d = map(p); // current distance to the scene

        // accumulate volumetric glow
        // add a bit of glowing color at each step
        // the exp() function gives a soft falloff
        // adjust the values 0.03 (density) and 2.0 (speed of falloff)
        let density = 0.03;
        let falloffSpeed = 5.0;
        col += glow_color * density * exp(-d * falloffSpeed);

        // did the ray hit the surface?
        if (d < 0.001) {
            // hit, add the surface color to the accumulation
            col += surface_color;
            break;
        }

        t += d * 0.8;

        if (t > 25.0) {
            // Add some background stars or cosmic dust
            let bg_noise = hash3(normalize(vec3f(uv, 1)) * 100.0).x;
            if (bg_noise > 0.998) {
                col += vec3f(1.0) * (bg_noise - 0.998) * 50.0;
            }
            break;
        }
        // Stop marching if distance is too far away
        if ( d > 100.0) {
            break;
        }
    }

    // tonemapping and gamma correction

    // 'col' is in linear light space and can become very bright (HDR)
    // this simple tonemapping prevents values > 1.0 to be blown out to pure white
    col = col / (col + 1.0); // (Reinhard tonemapping)

    // gamma correction (Linear -> sRGB/Gamma space)
    col = pow(col, vec3f(1.0 / 2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
