// Classic Simplex noise 2D implementation
fn hash2(p: float2) -> float2 {
    let p2 = float2(dot(p, float2(127.1, 311.7)), dot(p, float2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p2) * 43758.5453123);
}

/// 2D Simplex noise implementation
/// Returns noise values in range [-1, 1]
/// 
/// Implementation steps:
/// 1. Skew input space to simplex grid
/// 2. Determine simplex cell and offsets
/// 3. Calculate contributions from all 3 corners
/// 4. Sum contributions with gradient dot products
fn simplex2d(p: float2) -> float {
    // Skew factors for 2D simplex grid
    let F2 = 0.366025403; // (sqrt(3)-1)/2
    let G2 = 0.211324865; // (3-sqrt(3))/6
    
    // Skew the input space
    let s = dot(p, float2(0.5, 0.5));
    let i = floor(p + s);
    let t = dot(i, float2(G2, G2));
    let x0 = p - i + t;
    
    // Determine simplex cell
    var i1: float2;
    if (x0.x > x0.y) {
        i1 = float2(1, 0);
    } else {
        i1 = float2(0, 1);
    }
    let x1 = x0 - i1 + G2;
    let x2 = x0 - 1.0 + 2.0 * G2;
    
    // Calculate gradients
    let j = i % 256.0;
    let k = (i + i1) % 256.0;
    let l = (i + 1.0) % 256.0;
    
    let gi0 = normalize(hash2(j));
    let gi1 = normalize(hash2(k));
    let gi2 = normalize(hash2(l));
    
    // Calculate contribution weights
    var n0: f32;
    var n1: f32;
    var n2: f32;
    
    var t0 = 0.5 - dot(x0, x0);
    if (t0 < 0.0) {
        n0 = 0.0;
    } else {
        t0 *= t0;
        n0 = t0 * t0 * dot(gi0, x0);
    }
    
    var t1 = 0.5 - dot(x1, x1);
    if (t1 < 0.0) {
        n1 = 0.0;
    } else {
        t1 *= t1;
        n1 = t1 * t1 * dot(gi1, x1);
    }
    
    var t2 = 0.5 - dot(x2, x2);
    if (t2 < 0.0) {
        n2 = 0.0;
    } else {
        t2 *= t2;
        n2 = t2 * t2 * dot(gi2, x2);
    }
    
    // Scale output to [-1,1]
    return 70.0 * (n0 + n1 + n2);
}

/* 3D hash function matching our 2D implementation */
fn hash3(p: float3) -> float3 {
    let p2 = float3(
        dot(p, float3(127.1, 311.7, 241.1)),
        dot(p, float3(269.5, 183.3, 391.5)),
        dot(p, float3(419.2, 371.9, 149.7))
    );
    return -1.0 + 2.0 * fract(sin(p2) * 43758.5453123);
}

/// 3D Simplex noise implementation
/// Returns noise values in range [-1, 1]
fn simplex3d(p: float3) -> float {
    // Skew factors for 3D simplex grid
    let F3 = 0.333333333;
    let G3 = 0.166666667;
    
    // Skew input space
    let s = dot(p, float3(F3, F3, F3));
    let i = floor(p + s);
    let t = dot(i, float3(G3, G3, G3));
    let x0 = p - i + t;
    
    // Determine simplex cell
    var i1: float3;
    var i2: float3;
    if (x0.x >= x0.y) {
        if (x0.y >= x0.z) {
            i1 = float3(1, 0, 0);
            i2 = float3(1, 1, 0);
        } else if (x0.x >= x0.z) {
            i1 = float3(1, 0, 0);
            i2 = float3(1, 0, 1);
        } else {
            i1 = float3(0, 0, 1);
            i2 = float3(1, 0, 1);
        }
    } else {
        if (x0.y < x0.z) {
            i1 = float3(0, 0, 1);
            i2 = float3(0, 1, 1);
        } else if (x0.x < x0.z) {
            i1 = float3(0, 1, 0);
            i2 = float3(0, 1, 1);
        } else {
            i1 = float3(0, 1, 0);
            i2 = float3(1, 1, 0);
        }
    }
    
    let x1 = x0 - i1 + G3;
    let x2 = x0 - i2 + 2.0 * G3;
    let x3 = x0 - 1.0 + 3.0 * G3;
    
    // Calculate gradients
    let j = i % 256.0;
    let k = (i + i1) % 256.0;
    let l = (i + i2) % 256.0;
    let m = (i + 1.0) % 256.0;
    
    let gj0 = normalize(hash3(j));
    let gj1 = normalize(hash3(k));
    let gj2 = normalize(hash3(l));
    let gj3 = normalize(hash3(m));
    
    // Calculate contribution weights
    var n0: f32;
    var n1: f32;
    var n2: f32;
    var n3: f32;
    
    var t0 = 0.6 - dot(x0, x0);
    if (t0 < 0.0) {
        n0 = 0.0;
    } else {
        t0 *= t0;
        n0 = t0 * t0 * dot(gj0, x0);
    }
    
    var t1 = 0.6 - dot(x1, x1);
    if (t1 < 0.0) {
        n1 = 0.0;
    } else {
        t1 *= t1;
        n1 = t1 * t1 * dot(gj1, x1);
    }
    
    var t2 = 0.6 - dot(x2, x2);
    if (t2 < 0.0) {
        n2 = 0.0;
    } else {
        t2 *= t2;
        n2 = t2 * t2 * dot(gj2, x2);
    }
    
    var t3 = 0.6 - dot(x3, x3);
    if (t3 < 0.0) {
        n3 = 0.0;
    } else {
        t3 *= t3;
        n3 = t3 * t3 * dot(gj3, x3);
    }
    
    // Scale output to [-1,1]
    return 32.0 * (n0 + n1 + n2 + n3);
}

/// Fractal Brownian Motion (fBm) using Simplex noise
fn simplex_fractal(m: float3) -> float {
    var amplitude = 1.0;
    var frequency = 1.0;
    var total = 0.0;
    var max_value = 0.0;
    
    for (var i: u32 = 0; i < u32(custom.octaves); i++) {
        total += simplex3d(m * frequency) * amplitude;
        max_value += amplitude;
        
        amplitude *= custom.persistence * custom.gain;
        frequency *= custom.lacunarity;
        amplitude = clamp(amplitude, 0.001, 100.0);
    }
    
    var value = total / max_value;
    value = pow(abs(value), custom.exponent) * sign(value);
    return mix(value, smoothstep(custom.range, custom.range, value), custom.smoothing);
}

/// 3D Perlin noise implementation
/// Follows similar structure to 2D version but with 8 grid cell corners
/// Returns noise values in range [-1, 1]
fn perlin3d(p: float3) -> float {
    let Pi = floor(p);
    let Pf = fract(p);
    
    // Grid cell coordinates
    let i000 = Pi;
    let i100 = Pi + float3(1, 0, 0);
    let i010 = Pi + float3(0, 1, 0);
    let i110 = Pi + float3(1, 1, 0);
    let i001 = Pi + float3(0, 0, 1);
    let i101 = Pi + float3(1, 0, 1);
    let i011 = Pi + float3(0, 1, 1);
    let i111 = Pi + float3(1, 1, 1);
    
    // Gradient vectors
    let g000 = normalize(hash3(i000));
    let g100 = normalize(hash3(i100));
    let g010 = normalize(hash3(i010));
    let g110 = normalize(hash3(i110));
    let g001 = normalize(hash3(i001));
    let g101 = normalize(hash3(i101));
    let g011 = normalize(hash3(i011));
    let g111 = normalize(hash3(i111));
    
    // Distance vectors
    let d000 = Pf - float3(0, 0, 0);
    let d100 = Pf - float3(1, 0, 0);
    let d010 = Pf - float3(0, 1, 0);
    let d110 = Pf - float3(1, 1, 0);
    let d001 = Pf - float3(0, 0, 1);
    let d101 = Pf - float3(1, 0, 1);
    let d011 = Pf - float3(0, 1, 1);
    let d111 = Pf - float3(1, 1, 1);
    
    // Dot products
    let n000 = dot(g000, d000);
    let n100 = dot(g100, d100);
    let n010 = dot(g010, d010);
    let n110 = dot(g110, d110);
    let n001 = dot(g001, d001);
    let n101 = dot(g101, d101);
    let n011 = dot(g011, d011);
    let n111 = dot(g111, d111);
    
    // 3D quintic interpolation parameters
    let u = Pf * Pf * Pf * (Pf * (Pf * 6.0 - 15.0) + 10.0);
    return mix(
        mix(
            mix(n000, n100, u.x),
            mix(n010, n110, u.x),
            u.y
        ),
        mix(
            mix(n001, n101, u.x),
            mix(n011, n111, u.x),
            u.y
        ),
        u.z
    ) * 1.1547; // scale to [-1,1]
}

/// Fractal Brownian Motion (fBm) using Perlin noise
/// Combines multiple octaves of noise at increasing frequencies
/// - persistence: Amplitude multiplier between octaves (0.0-1.0)
/// - lacunarity: Frequency multiplier between octaves (typically 2.0)
/// Fractal Brownian Motion with enhanced controls
/// Applies to both 2D and 3D noise through shared parameters
fn perlin_fractal(m: float3) -> float {
    var amplitude = 1.0;
    var frequency = 1.0;
    var total = 0.0;
    var max_value = 0.0;
    
    // Main octave loop with flexible controls
    for (var i: u32 = 0; i < u32(custom.octaves); i++) {
        // Sample noise and accumulate
        total += perlin3d(m * frequency) * amplitude;
        max_value += amplitude;
        
        // Update amplitude and frequency using customizable rates
        amplitude *= custom.persistence * custom.gain;  // Combined persistence and gain
        frequency *= custom.lacunarity;  // Remove floor() for smooth transitions
        
        // Apply amplitude clamping to prevent overfloww
        amplitude = clamp(amplitude, 0.001, 100.0);
    }
    
    // Normalize and apply final transformations
    var value = total / max_value;
    value = pow(abs(value), custom.exponent) * sign(value); // Exponent for contrast control
    return mix(value, smoothstep(custom.range, custom.range, value), custom.smoothing);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = float2(id.xy) + .5;
    let resolution = float2(screen_size);
    let px = fragCoord / resolution.x;
    let py = fragCoord / resolution.y;
    let p3 = float3(px, time.elapsed*0.025);

    var f = 0.;
    // Visualize different noise types side-by-side
    if (px.x < .5 && py.y < .5) { 
        // Left side: Raw 3D Simplex noise
        f = simplex3d(p3*custom.scale);
    } else if (px.x >= .5 && py.y < .5){ 
        // Right side: Fractal noise (4 octaves combined)
        f = simplex_fractal(p3*custom.scale);
    }
    else if (px.x < .5 && py.y >= .5){ 
        // Right side: Fractal noise (4 octaves combined)
        f = perlin3d(p3*custom.scale);
    }
    else {
        f = perlin_fractal(p3*custom.scale);
    }
    
    // Remap noise from [-1,1] to [0,1] for display
    f = 0.5 + 0.5*f;
    
    // Smooth transition between left/right sections
    f *= smoothstep(0.0, 0.005, abs(px.x - 0.5));
    f *= smoothstep(0.0, 0.005, abs(py.y - 0.5));

    f = pow(f, 2.2); // perceptual gradient to linear colour space
    textureStore(screen, int2(id.xy), float4(f, f, f, 1.));
}
